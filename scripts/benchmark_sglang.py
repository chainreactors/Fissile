"""Paired BF16/Fissile or mixed/mixed benchmark on external tokenized inputs.

Both engines stay resident; calls alternate sequentially, never concurrently.
Each engine warms every input once before measured rounds. No training occurs.
"""

import argparse
from contextlib import contextmanager, redirect_stdout
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import select
import signal
import socket
import statistics
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write(chr(10))


def percentile(values, q):
    data = sorted(values)
    position = (len(data) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return data[lo] + (data[hi] - data[lo]) * (position - lo)


def decode_choice(labels, logprobs):
    """Four-decimal probability readout with the evaluation's lexical tie break."""
    values = [math.exp(v - max(logprobs)) for v in logprobs]
    probabilities = [v / sum(values) for v in values]
    scaled = [max(v, 0.0) * 10000 for v in probabilities]
    total = sum(scaled) or 1.0
    scaled = [v * 10000 / total for v in scaled]
    integers = [math.floor(v) for v in scaled]
    order = sorted(range(len(scaled)), key=lambda i:scaled[i]-integers[i], reverse=True)
    for i in order[:10000-sum(integers)]:
        integers[i] += 1
    distribution = dict(zip(labels, integers, strict=True))
    return max(sorted(distribution), key=distribution.get)


def baseline_settings(model):
    return dict(model_path=str(Path(model).resolve()), dtype='bfloat16',
        # Same as fissile.backends.sglang.engine: SGLang 0.5.21 reserves
        # positions, so 16384 + 7 admits full 16384-token inputs.
        context_length=16384 + 7, disable_radix_cache=True,
        cuda_graph_backend_decode='disabled', cuda_graph_backend_prefill='disabled',
        chunked_prefill_size=-1, mem_fraction_static=0.35, max_running_requests=1,
        max_total_tokens=32768, max_prefill_tokens=16384, attention_backend='triton',
        linear_attn_backend='triton', linear_attn_prefill_backend='triton',
        mamba_backend='triton', mamba_ssm_dtype='float32', log_level='warning')


@contextmanager
def baseline_session(model):
    import sglang
    from fissile.backends.contracts import CheckedSession, Score
    from fissile.models import get_profile

    class Session:
        def __init__(self, engine):
            self.engine = engine

        def score(self, ids, candidates):
            result = self.engine.score(query=[], items=[list(ids)],
                label_token_ids=list(candidates), apply_softmax=True,
                return_token_logprobs=True, temperature=1.0)
            return Score(tuple(result.token_logprobs[0]), result.prompt_tokens)

    engine = sglang.Engine(**baseline_settings(model))
    try:
        yield CheckedSession(Session(engine), get_profile('qwen3.5-dense-9b'))
    finally:
        engine.shutdown()


def worker(args):
    source = args.baseline_source if args.worker == 'bf16' else args.candidate_source
    if source:
        source = str(Path(source).resolve())
        if not (Path(source) / 'fissile/__init__.py').is_file():
            raise ValueError('Source root must contain fissile/__init__.py')
        sys.path.insert(0, source)
        os.environ['PYTHONPATH'] = source + os.pathsep + os.environ.get('PYTHONPATH', '')
    wire = sys.stdout
    with redirect_stdout(sys.stderr):
        import torch
        import fissile
        from fissile.backends.registry import open_backend
        from sglang.kernels.ops.kvcache.kvcache import can_use_store_cache
        torch.set_num_threads(2)
        if not can_use_store_cache(2048, 2048):
            raise RuntimeError('Required JIT KV-cache kernel unavailable')
        if args.worker == 'bf16' and not args.baseline_config:
            context = baseline_session(args.model)
        else:
            config = args.baseline_config if args.worker == 'bf16' else args.config
            backend = args.candidate_backend if args.worker == 'mixed' and args.candidate_backend else 'sglang'
            context = open_backend(backend, config, args.model)
        started = time.perf_counter()
        with context as session:
            engine = session.session.engine
            actual = engine.server_args.resolved_dict()
            expected = baseline_settings(args.model)
            for key, value in expected.items():
                if actual[key] != value:
                    raise RuntimeError('Engine setting differs: ' + key)
            ready = dict(ready=True, arm=args.worker, startup_seconds=time.perf_counter()-started,
                settings=expected, versions={n: importlib.metadata.version(n) for n in
                ['torch', 'sglang', 'transformers', 'triton', 'nvidia-cutlass-dsl']},
                jit_available=True,
                implementation='sglang-bf16' if args.worker == 'bf16' and not args.baseline_config else 'sglang-fissile',
                source_root=source,
                source_hashes={str(p.relative_to(Path(fissile.__file__).parent)):sha(p)
                    for p in sorted(Path(fissile.__file__).parent.rglob('*.py'))})
            wire.write('__FISSILE_BENCH__' + json.dumps(ready) + chr(10)); wire.flush()
            for line in sys.stdin:
                request = json.loads(line)
                if request.get('stop'):
                    break
                started = time.perf_counter()
                result = session.score(request['input_ids'], request['candidate_token_ids'])
                elapsed = (time.perf_counter() - started) * 1000
                record = dict(logprobs=result.token_logprobs, prompt_tokens=result.prompt_tokens,
                    elapsed_ms=elapsed, generated_tokens=0)
                wire.write('__FISSILE_BENCH__' + json.dumps(record, allow_nan=False) + chr(10)); wire.flush()


class Worker:
    def __init__(self, args, arm, output):
        self.log = (output / (arm + '.log')).open('x')
        bootstrap = ('import json,os,runpy,sys; '
            'sys.path[:]=json.loads(os.environ["FISSILE_BENCH_IMPORT_PATHS"]); '
            'sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")')
        env = dict(os.environ, FISSILE_BENCH_IMPORT_PATHS=json.dumps(sys.path))
        env.pop('SGLANG_EXTERNAL_MODEL_PACKAGE', None)
        env.pop('FISSILE_CONFIG', None)
        command = [sys.executable, '-S', '-u', '-c', bootstrap, str(Path(__file__).resolve()),
            '--worker', arm, '--model', args.model, '--config', args.config]
        for name in ['baseline_config', 'baseline_source', 'candidate_source', 'candidate_backend']:
            value = getattr(args, name, None)
            if value:
                command += ['--' + name.replace('_', '-'), value]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1, env=env, start_new_session=True)

    def receive(self, timeout=1200):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError('Benchmark worker timed out; inspect its log')
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('Benchmark worker exited; inspect its log')
            if line.startswith('__FISSILE_BENCH__'):
                return json.loads(line[len('__FISSILE_BENCH__'):])
            self.log.write(line); self.log.flush()

    def score(self, row):
        self.process.stdin.write(json.dumps({k:row[k] for k in
            ['input_ids','candidate_token_ids']}) + chr(10))
        self.process.stdin.flush()
        return self.receive(180)

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({'stop':True}) + chr(10))
                self.process.stdin.flush()
                self.process.wait(timeout=60)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                # Only this benchmark's process group is eligible for cleanup.
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        self.log.close()


def summarize(records, rows, rounds):
    labels = sorted({r['expected'] for r in rows})
    output = {}
    for arm in ['bf16','mixed']:
        selected = [r for r in records if r['arm']==arm]
        assert len(selected)==len(rows)*rounds
        latencies = [r['elapsed_ms'] for r in selected]
        seconds = math.fsum(latencies)/1000
        per_round = []
        predictions = {}
        for round_id in range(rounds):
            rr = [r for r in selected if r['round']==round_id]
            classes = {label:{'count':0,'correct':0} for label in labels}
            for r in rr:
                c=classes[r['expected']];c['count']+=1;c['correct']+=r['predicted']==r['expected']
                predictions.setdefault(r['id'],set()).add(r['predicted'])
            per_round.append(dict(round=round_id, correct=sum(v['correct'] for v in classes.values()),
                accuracy=sum(v['correct'] for v in classes.values())/len(rows),
                balanced_accuracy=statistics.mean(v['correct']/v['count'] for v in classes.values()),
                classes=classes, sum_seconds=math.fsum(r['elapsed_ms'] for r in rr)/1000))
        output[arm]=dict(requests=len(selected), unique_requests=len(rows), rounds=rounds,
            sum_seconds=seconds, mean_ms=statistics.mean(latencies),
            p50_ms=percentile(latencies,.5),p95_ms=percentile(latencies,.95),
            requests_per_second=len(selected)/seconds,
            input_tokens_per_second=sum(r['prompt_tokens'] for r in selected)/seconds,
            per_round=per_round, inconsistent_labels_across_rounds=sum(len(p)>1 for p in predictions.values()))
    return output


def validate_pairs(records, expected_pairs, *, require_exact=False):
    pairs = {}
    for record in records:
        key = (record['round'], record['id'])
        pair = pairs.setdefault(key, {})
        if record['arm'] not in {'bf16', 'mixed'} or record['arm'] in pair:
            raise ValueError('Duplicate or unknown benchmark arm')
        pair[record['arm']] = record
    if len(pairs) != expected_pairs or any(set(p) != {'bf16', 'mixed'} for p in pairs.values()):
        raise ValueError('Incomplete paired measurements')
    exact = all(p['bf16']['logprobs'] == p['mixed']['logprobs'] for p in pairs.values())
    if require_exact and not exact:
        raise ValueError('Raw candidate log probabilities differ between arms')
    return exact


def benchmark(args):
    required = ['inputs','output','host_policy','gpu_lock']
    if any(not getattr(args,name) for name in required):
        raise ValueError('inputs, output, host-policy and gpu-lock are required')
    if args.rounds < 1:
        raise ValueError('rounds must be positive')
    if args.require_exact and not args.baseline_config:
        raise ValueError('Exact mixed/mixed comparison requires --baseline-config')
    policy=json.loads(Path(args.host_policy).read_text());host=socket.gethostname()
    if host not in policy['mainline_compute_hosts'] or host in policy.get('disabled_host_aliases',[]):
        raise RuntimeError('Host is not allowed for benchmarks')
    for name in ['FISSILE_SGLANG_MEM_FRACTION_STATIC','FISSILE_SGLANG_NCCL_PORT','SGLANG_EXTERNAL_MODEL_PACKAGE']:
        if os.environ.get(name):
            raise RuntimeError('Unset benchmark override: '+name)
    rows=[json.loads(line) for line in Path(args.inputs).open()]
    if not rows or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Inputs require nonempty unique IDs')
    for r in rows:
        assert len(r['labels'])==len(r['candidate_token_ids'])
        assert r['expected'] in r['labels'] and len(set(r['labels']))==len(r['labels'])
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    lease=Path(args.gpu_lock).open('a+')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    util=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True)
    if active or any(int(v)>5 for v in util.splitlines()):
        raise RuntimeError('GPU occupied; no launch or interruption')
    memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
    if memory['MemAvailable'] < 80*1024*1024:
        raise RuntimeError('Paired resident engines require 80 GiB available host memory')
    protocol=dict(format='fissile-sglang-paired-v1',requests=len(rows),rounds=args.rounds,
        warmup_passes_per_arm=1,input_tokens=sum(len(r['input_ids']) for r in rows),
        inputs_sha256=sha(args.inputs),runner_sha256=sha(__file__),host_policy_sha256=sha(args.host_policy),
        random_seed=20261031,arm_order='Alternating per input; parity reversed each round',
        resident_engines=2,concurrent_gpu_calls=1,batch_size=1,generated_tokens=0,
        arms={'bf16': 'sglang-fissile-baseline' if args.baseline_config else 'sglang-bf16',
              'mixed': 'sglang-fissile-candidate'},
        require_exact=args.require_exact,
        timing='CheckedSession.score only; includes SGLang IPC, excludes external worker IPC, tokenization, decoding, startup and warmup',
        complete=False)
    write_json(output/'protocol.json',protocol)
    workers={};records=[]
    try:
        for arm in ['bf16','mixed']:
            print('STARTING',arm,flush=True)
            workers[arm]=Worker(args,arm,output)
            ready=workers[arm].receive();assert ready['ready']
            write_json(output/(arm+'-ready.json'),ready)
            print('READY',arm,round(ready['startup_seconds'],2),flush=True)
        for i,row in enumerate(rows):
            pair = {}
            for arm in ['bf16','mixed']:
                r=workers[arm].score(row);assert r['prompt_tokens']==len(row['input_ids'])
                pair[arm] = r
            if args.require_exact and pair['bf16']['logprobs'] != pair['mixed']['logprobs']:
                write_json(output/'warmup-mismatch.json',dict(id=row['id'],results=pair))
                raise ValueError('Raw candidate scores differ during warmup')
            if (i+1)%100==0: print('WARMUP',i+1,len(rows),flush=True)
        with (output/'records.jsonl').open('x') as stream:
            for round_id in range(args.rounds):
                order=list(range(len(rows)));random.Random(20261031+round_id).shuffle(order)
                for position,index in enumerate(order):
                    row=rows[index]
                    arms=['bf16','mixed'] if (position+round_id)%2==0 else ['mixed','bf16']
                    for arm in arms:
                        r=workers[arm].score(row)
                        assert r['prompt_tokens']==len(row['input_ids']) and r['elapsed_ms']>0
                        assert len(r['logprobs'])==len(row['labels']) and all(math.isfinite(v) for v in r['logprobs'])
                        predicted=decode_choice(row['labels'],r['logprobs'])
                        record=dict(r,id=row['id'],index=index,round=round_id,arm=arm,
                            expected=row['expected'],predicted=predicted)
                        records.append(record);stream.write(json.dumps(record,allow_nan=False)+chr(10))
                    if (position+1)%100==0:
                        stream.flush();print('MEASURED',round_id+1,position+1,len(rows),flush=True)
                stream.flush()
        exact = validate_pairs(records, len(rows)*args.rounds, require_exact=args.require_exact)
        metrics=summarize(records,rows,args.rounds)
        result=dict(complete=True,metrics=metrics,raw_logprobs_exact=exact,arms=protocol['arms'],
            speedup=metrics['bf16']['sum_seconds']/metrics['mixed']['sum_seconds'],
            inputs_sha256=sha(args.inputs),records_sha256=sha(output/'records.jsonl'),
            protocol_sha256=sha(output/'protocol.json'))
        write_json(output/'summary.json',result)
        print('COMPLETE',json.dumps(result,allow_nan=False),flush=True)
    except BaseException as exc:
        write_json(output/'failed.json',dict(type=type(exc).__name__,message=str(exc),complete=False))
        raise
    finally:
        for w in reversed(list(workers.values())): w.close()
        lease.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=['bf16','mixed'],help=argparse.SUPPRESS)
    for name in ['model','config']:
        parser.add_argument('--'+name,required=True)
    for name in ['inputs','output','host-policy','gpu-lock']:
        parser.add_argument('--'+name)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--baseline-config', help='Compare against a second Fissile config instead of BF16')
    parser.add_argument('--baseline-source', help='Trusted source root containing the baseline fissile package')
    parser.add_argument('--candidate-source', help='Trusted source root containing the candidate fissile package')
    parser.add_argument('--require-exact', action='store_true', help='Fail on any raw-score difference in mixed/mixed comparisons')
    parser.add_argument('--candidate-backend', help='Registered backend for the candidate arm (default sglang)')
    args=parser.parse_args()
    worker(args) if args.worker else benchmark(args)


if __name__=='__main__':
    main()
