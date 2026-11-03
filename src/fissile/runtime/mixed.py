"""Engine-independent binding, artifact identity and execution coverage."""

from collections import Counter
from fissile.config import Config, read_json, contained, sha256


class MixedRuntime:
    """Adapters bind canonical weights; providers supply numerical closures."""

    def __init__(self, config):
        self.config = (
            config.validate() if isinstance(config, Config) else Config.load(config)
        )
        self.plan = self.config.plan
        self.profile = self.plan.profile
        self.metadata = read_json(self.config.manifest)["projections"]
        self.provider = self.config.provider.create(self.config)
        self.expected_routes = self.plan.routes
        self.calls = {}
        self.routes = {}
        self.paired_mlps = set()

    def check(self, key, weight):
        from fissile.kernels.packs import weight_hash

        if key in self.calls:
            raise ValueError("Projection already installed: " + key)
        meta = self.metadata[key]
        if (
            tuple(weight.shape) != tuple(meta["shape"])
            or weight_hash(weight) != meta["effective_weight_sha256"]
        ):
            raise ValueError("Effective weight identity mismatch: " + key)
        self.calls[key] = 0
        self.routes[key] = self.expected_routes[key]
        return meta

    def pack(self, key, weight):
        from fissile.kernels.packs import load_pack

        meta = self.check(key, weight)
        return load_pack(
            contained(self.config.artifacts_dir, meta["path"]),
            meta,
            device=weight.device,
        )

    def projection(self, key, weight):
        selected = self.expected_routes[key]
        if selected in self.config.provider.packed_routes:
            packed = self.pack(key, weight)
        else:
            self.check(key, weight)
            packed = None
        compute = self.provider.projection(selected, weight, packed)

        def apply(x):
            shape = x.shape
            result = compute(self.provider.matrix(x))
            self.calls[key] += 1
            return result.reshape(*shape[:-1], self.metadata[key]["shape"][0])

        return apply

    def candidate_readout(self, key, weight):
        """Readout of only the requested vocabulary rows (others -inf).

        Same weight identity and coverage accounting as ``projection``; the
        engine adapter must supply the request's candidate token IDs.
        """
        factory = getattr(self.provider, "candidate_readout", None)
        if factory is None:
            raise ValueError("Kernel provider does not implement candidate readout")
        selected = self.expected_routes[key]
        packed = (
            self.pack(key, weight)
            if selected in self.config.provider.packed_routes
            else (self.check(key, weight), None)[1]
        )
        compute = factory(selected, weight, packed)

        def apply(x, candidates):
            result = compute(self.provider.matrix(x), candidates)
            self.calls[key] += 1
            return result.reshape(*x.shape[:-1], self.metadata[key]["shape"][0])

        return apply

    def _validate_group(self, keys, weights):
        keys, weights = tuple(keys), tuple(weights)
        if not keys or len(keys) != len(weights) or len(set(keys)) != len(keys):
            raise ValueError("Require nonempty, distinct projection keys and weights")
        if any(k not in self.expected_routes or k in self.calls for k in keys):
            raise ValueError("Unknown or already installed projection in group")
        if len({self.metadata[k]["shape"][1] for k in keys}) != 1:
            raise ValueError("Grouped projections must have the same input width")
        return keys, weights

    def _bind_group(self, keys, weights):
        keys, weights = self._validate_group(keys, weights)
        projections = []
        for key, weight in zip(keys, weights):
            route = self.expected_routes[key]
            if route in self.config.provider.packed_routes:
                packed = self.pack(key, weight)
            else:
                self.check(key, weight)
                packed = None
            projections.append((route, weight, packed))
        return keys, tuple(projections)

    def projection_group(self, keys, weights):
        """Bind sibling projections of the same input without joining GEMMs.

        Providers may offer call-local activation reuse; existing providers
        retain their independent projection functions. Engines own concatenation.
        """
        factory = getattr(self.provider, "projection_group", None)
        if factory is None:
            keys, weights = self._validate_group(keys, weights)
            functions = tuple(self.projection(k, w) for k, w in zip(keys, weights))
            return lambda x: tuple(fn(x) for fn in functions)
        keys, projections = self._bind_group(keys, weights)
        compute = factory(projections)

        def apply(x):
            results = compute(self.provider.matrix(x))
            if len(results) != len(keys):
                raise RuntimeError("Provider returned incomplete projection group")
            shaped = tuple(
                result.reshape(*x.shape[:-1], self.metadata[key]["shape"][0])
                for key, result in zip(keys, results)
            )
            for key in keys:
                self.calls[key] += 1
            return shaped

        return apply

    def joined_projection_group(self, keys, weights, concatenate):
        """Sibling projections whose outputs the engine consumes concatenated.

        Providers with ``joined_projection_group`` write each unchanged GEMM
        into its column slice of one buffer. Otherwise the engine-supplied
        ``concatenate`` joins the ``projection_group`` outputs in key order.
        """
        factory = getattr(self.provider, "joined_projection_group", None)
        if factory is None:
            group = self.projection_group(keys, weights)
            return lambda x: concatenate(group(x))
        keys, projections = self._bind_group(keys, weights)
        width = sum(self.metadata[key]["shape"][0] for key in keys)
        compute = factory(projections)

        def apply(x):
            result = compute(self.provider.matrix(x))
            rows = 1
            for n in x.shape[:-1]:
                rows *= n
            if tuple(result.shape) != (rows, width):
                raise RuntimeError("Provider returned incomplete joined projection")
            shaped = result.reshape(*x.shape[:-1], width)
            for key in keys:
                self.calls[key] += 1
            return shaped

        return apply

    def mlp(self, prefix, gate, up, down, *, long_silu):
        groups = {m.prefix: m for m in self.plan.fused_mlps}
        if prefix not in groups or prefix in self.paired_mlps:
            raise ValueError("Unknown or duplicate fused MLP binding: " + prefix)
        keys = groups[prefix].keys
        packs = tuple(
            self.pack(k, w) for k, w in zip(keys, (gate, up, down), strict=True)
        )
        compute = self.provider.mlp(packs, long_silu=long_silu)
        self.paired_mlps.add(prefix)

        def apply(x):
            shape = x.shape
            result = compute(self.provider.matrix(x))
            for key in keys:
                self.calls[key] += 1
            return result.reshape(*shape[:-1], self.metadata[keys[-1]]["shape"][0])

        return apply

    def validate_coverage(self):
        if set(self.calls) != set(self.expected_routes) or self.paired_mlps != {
            m.prefix for m in self.plan.fused_mlps
        }:
            raise ValueError("Incomplete selected-model projection coverage")
        if self.routes != dict(self.expected_routes):
            raise ValueError("Installed routes differ from the selected recipe")
        return {
            "complete": True,
            "profile": self.profile.name,
            "recipe": self.plan.recipe.name,
            "kernel_provider": self.config.kernel_provider,
            "routes": dict(
                Counter(self.routes.values()), paired_mlp=len(self.paired_mlps)
            ),
            "effective_weight_identities_verified": len(self.calls),
            "weights_requantized": False,
            "prepared_manifest_sha256": sha256(self.config.manifest),
            "resource_sha256": {
                n: sha256(r.path) for n, r in self.config.resources.items()
            },
            "full_projection_coverage": True,
            "full_vocab": self.profile.vocab_size,
            "bf16_parameters_retained": True,
            "memory_compression_claim": False,
        }

    def validate_counts(self, before, *, head, body):
        """Check one engine call: output head ran ``head`` times, every body
        projection ``body`` times, against counters taken just before it."""
        if set(self.calls) != set(self.expected_routes) or set(before) != set(self.calls):
            raise RuntimeError("Projection bypass or unexpected extra execution")
        delta = {k: self.calls[k] - before[k] for k in self.calls}
        if delta.pop("lm_head") != head or set(delta.values()) != {body}:
            raise RuntimeError("Projection bypass or unexpected extra execution")

    def validate_pass(self, before):
        """Check one outer forward against counters taken just before it.

        The output head runs eagerly exactly once. The transformer body either ran
        eagerly (every body projection once) or was replayed from a captured
        prefill graph (no Python execution); a replay is accepted only if every
        Python-observed body pass, including capture, covered all projections.
        Returns "eager" or "graph_replay".
        """
        if set(self.calls) != set(self.expected_routes) or set(before) != set(self.calls):
            raise RuntimeError("Projection bypass or unexpected extra execution")
        delta = {k: self.calls[k] - before[k] for k in self.calls}
        if delta.pop("lm_head") != 1:
            raise RuntimeError("Projection bypass or unexpected extra execution")
        body = set(delta.values())
        if body not in ({1}, {0}) or len({self.calls[k] for k in delta}) != 1:
            raise RuntimeError("Projection bypass or unexpected extra execution")
        return "eager" if body == {1} else "graph_replay"

    def validate_forward(self, count):
        if set(self.calls) != set(self.expected_routes) or set(self.calls.values()) != {
            count
        }:
            raise RuntimeError("Projection bypass or unexpected extra execution")
