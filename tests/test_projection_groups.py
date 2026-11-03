"""CPU-only tests for same-input projection groups and call-local reuse."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from fissile.runtime.mixed import MixedRuntime


class Matrix:
    shape = (1, 3, 128)
    def reshape(self, *shape):
        return shape
    def __len__(self):
        return 3


def make_runtime(grouped=True):
    runtime = MixedRuntime.__new__(MixedRuntime)
    runtime.calls = {}
    runtime.expected_routes = {'q': 'fp4', 'z': 'fp4'}
    runtime.metadata = {'q': {'shape': (256, 128)}, 'z': {'shape': (128, 128)}}
    runtime.config = SimpleNamespace(provider=SimpleNamespace(packed_routes={'fp4'}))
    runtime.pack = lambda key, weight: runtime.calls.setdefault(key, 0)
    runtime.provider = SimpleNamespace(matrix=lambda x: x)
    if grouped:
        runtime.provider.projection_group = lambda parts: lambda x: (Matrix(), Matrix())
    else:
        runtime.provider.projection = lambda route, weight, packed: lambda x: Matrix()
    return runtime


class ProjectionGroups(unittest.TestCase):
    def runtime(self, grouped=True):
        return make_runtime(grouped)

    def test_group_shape_and_coverage(self):
        runtime = self.runtime()
        fn = runtime.projection_group(['q', 'z'], [object(), object()])
        self.assertEqual(fn(Matrix()), ((1, 3, 256), (1, 3, 128)))
        self.assertEqual(runtime.calls, {'q': 1, 'z': 1})
        fn(Matrix())
        self.assertEqual(runtime.calls, {'q': 2, 'z': 2})

    def test_existing_provider_fallback(self):
        runtime = self.runtime(grouped=False)
        fn = runtime.projection_group(['q', 'z'], [object(), object()])
        self.assertEqual(fn(Matrix()), ((1, 3, 256), (1, 3, 128)))
        self.assertEqual(runtime.calls, {'q': 1, 'z': 1})

    def test_reject_invalid_binding_before_loading(self):
        for keys, weights in [([], []), (['q'], []), (['q', 'q'], [1, 2]), (['missing'], [1])]:
            runtime = self.runtime()
            with self.assertRaises(ValueError):
                runtime.projection_group(keys, weights)
            self.assertEqual(runtime.calls, {})
        runtime = self.runtime()
        runtime.metadata['z']['shape'] = (128, 256)
        with self.assertRaises(ValueError):
            runtime.projection_group(['q', 'z'], [1, 2])
        self.assertEqual(runtime.calls, {})

    def test_incomplete_output_does_not_record_coverage(self):
        runtime = self.runtime()
        runtime.provider.projection_group = lambda parts: lambda x: (Matrix(),)
        fn = runtime.projection_group(['q', 'z'], [1, 2])
        with self.assertRaises(RuntimeError):
            fn(Matrix())
        self.assertEqual(runtime.calls, {'q': 0, 'z': 0})

    def test_reject_duplicate_installation(self):
        runtime = self.runtime()
        runtime.projection_group(['q', 'z'], [1, 2])
        with self.assertRaises(ValueError):
            runtime.projection_group(['q', 'z'], [1, 2])
        self.assertEqual(runtime.calls, {'q': 0, 'z': 0})

    def test_failed_compute_does_not_record_coverage(self):
        runtime = self.runtime()
        def fail(x):
            raise RuntimeError('simulated kernel failure')
        runtime.provider.projection_group = lambda parts: fail
        fn = runtime.projection_group(['q', 'z'], [1, 2])
        with self.assertRaises(RuntimeError):
            fn(Matrix())
        self.assertEqual(runtime.calls, {'q': 0, 'z': 0})

    def test_activation_reuse_is_route_and_call_local(self):
        # Load just the orchestration method; no torch/CUDA import in CPU tests.
        path = Path(__file__).parents[1] / 'src/fissile/providers/gb10.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'projection_group')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        calls = []
        serial = []
        def activation(route, x):
            token = object();calls.append((route, x, token));return token
        def prepare(route, weight, packed):
            def apply(a, rows):
                serial.append(weight);return a
            return apply
        provider = SimpleNamespace(_activation=activation, _prepared_projection=prepare)
        make = namespace['projection_group']
        with self.assertRaises(ValueError):make(provider, [])
        fn = make(provider, [('fp4', 'q', None), ('fp8', 'k', None), ('fp8', 'v', None)])
        x = Matrix();first = fn(x);second = fn(x)
        self.assertIs(first[1], first[2])
        self.assertIsNot(first[0], first[1])
        self.assertIsNot(first[1], second[1])
        self.assertEqual([c[0] for c in calls], ['fp4', 'fp8', 'fp4', 'fp8'])
        self.assertEqual(serial, ['q', 'k', 'v', 'q', 'k', 'v'])


class Joined:
    def __init__(self, rows, width):
        self.shape = (rows, width)
    def reshape(self, *shape):
        return shape


class JoinedProjectionGroups(unittest.TestCase):
    def runtime(self, joined=True, result=None):
        runtime = make_runtime(grouped=True)
        if joined:
            runtime.provider.joined_projection_group = (
                lambda parts: lambda x: result or Joined(3, 384))
        return runtime

    def test_direct_output_shape_and_coverage(self):
        runtime = self.runtime()
        fn = runtime.joined_projection_group(['q', 'z'], [1, 2], self.fail)
        self.assertEqual(fn(Matrix()), (1, 3, 384))
        self.assertEqual(runtime.calls, {'q': 1, 'z': 1})

    def test_engine_concatenation_without_provider_support(self):
        runtime = self.runtime(joined=False)
        joined = []
        fn = runtime.joined_projection_group(['q', 'z'], [1, 2], lambda parts: joined.append(parts) or 'joined')
        self.assertEqual(fn(Matrix()), 'joined')
        self.assertEqual(joined, [((1, 3, 256), (1, 3, 128))])
        self.assertEqual(runtime.calls, {'q': 1, 'z': 1})

    def test_wrong_width_does_not_record_coverage(self):
        runtime = self.runtime(result=Joined(3, 256))
        fn = runtime.joined_projection_group(['q', 'z'], [1, 2], self.fail)
        with self.assertRaises(RuntimeError):
            fn(Matrix())
        self.assertEqual(runtime.calls, {'q': 0, 'z': 0})

    def test_reject_invalid_binding_before_loading(self):
        for keys, weights in [([], []), (['q', 'q'], [1, 2]), (['missing'], [1])]:
            runtime = self.runtime()
            with self.assertRaises(ValueError):
                runtime.joined_projection_group(keys, weights, self.fail)
            self.assertEqual(runtime.calls, {})

    @staticmethod
    def fail(parts):
        raise AssertionError('engine concatenation must not run')

    def test_provider_writes_column_slices_in_key_order(self):
        path = Path(__file__).parents[1] / 'src/fissile/providers/gb10.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'joined_projection_group')
        writes, packs = [], []

        class Buffer:
            def __init__(self, shape):
                self.shape = shape
            def __getitem__(self, index):
                return index

        class Torch:
            bfloat16 = 'bf16'
            @staticmethod
            def empty(shape, dtype, device):
                buffers.append((shape, dtype, device))
                return Buffer(shape)

        buffers = []
        namespace = {'torch': Torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)

        def activation(route, x):
            packs.append(route)
            return (SimpleNamespace(shape=(16, 64)),)

        def prepare(route, weight, packed):
            def apply(a, rows, out=None):
                writes.append((weight.name, rows, out))
            return apply

        def weight(name, rows):
            return SimpleNamespace(name=name, shape=(rows, 128))

        provider = SimpleNamespace(_activation=activation, _prepared_projection=prepare)
        make = namespace['joined_projection_group']
        with self.assertRaises(ValueError):
            make(provider, [])
        with self.assertRaises(ValueError):
            make(provider, [('w4a16', weight('h', 128), None)])
        fn = make(provider, [('fp4', weight('q', 256), None), ('fp4', weight('z', 128), None)])
        x = type('X', (), {'device': 'cuda:0', '__len__': lambda self: 3})()
        result = fn(x)
        self.assertEqual(buffers, [((16, 384), 'bf16', 'cuda:0')])
        self.assertEqual(result, slice(None, 3))
        self.assertEqual(packs, ['fp4'])
        self.assertEqual(writes, [('q', 3, (slice(None), slice(0, 256))),
                                  ('z', 3, (slice(None), slice(256, 384)))])



class CandidateReadout(unittest.TestCase):
    def test_counts_coverage_and_requires_provider_support(self):
        runtime = make_runtime()
        runtime.expected_routes = {'lm_head': 'w4a16'}
        runtime.metadata = {'lm_head': {'shape': (256, 128)}}
        runtime.config = SimpleNamespace(provider=SimpleNamespace(packed_routes={'w4a16'}))
        seen = []
        runtime.provider.candidate_readout = lambda route, weight, packed: (
            lambda x, ids: seen.append((route, ids)) or Matrix())
        fn = runtime.candidate_readout('lm_head', object())
        self.assertEqual(fn(Matrix(), [5, 9]), (1, 3, 256))
        self.assertEqual(seen, [('w4a16', [5, 9])])
        self.assertEqual(runtime.calls, {'lm_head': 1})
        bare = make_runtime(); bare.expected_routes = {'lm_head': 'w4a16'}
        with self.assertRaises(ValueError):
            bare.candidate_readout('lm_head', object())

if __name__ == '__main__':
    unittest.main()
