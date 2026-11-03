"""Count input-prefix repetition without publishing tokens or executing a model.

This is only an unbounded-retention token opportunity, not cache feasibility,
model throughput or an accuracy result.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def analyze(sequences, lengths=(32,64,128,256,384)):
    if not sequences or any(not x or any(type(t) is not int or t<0 for t in x) for x in sequences):
        raise ValueError('Require nonempty sequences of nonnegative integer token IDs')
    if not lengths or any(type(n) is not int or n<1 for n in lengths):
        raise ValueError('Require positive integer prefix lengths')
    common=0
    for tokens in zip(*sequences):
        if len(set(tokens))!=1:break
        common+=1
    total=sum(map(len,sequences));results=[]
    for n in lengths:
        counts=Counter(tuple(x[:n]) for x in sequences if len(x)>=n)
        reused=sum((count-1)*n for count in counts.values())
        results.append(dict(prefix_tokens=n,eligible_requests=sum(counts.values()),unique_prefixes=len(counts),largest_group=max(counts.values(),default=0),hypothetical_avoided_tokens=reused,fraction_of_input_tokens=reused/total))
    return dict(requests=len(sequences),input_tokens=total,common_prefix_tokens=common,prefixes=results,assumptions=['Retain each distinct prefix after its first computation with no eviction.','Counts token positions only; ignores numerical partition invariance and all cache overhead.','No prompts, token IDs, semantic labels or model outputs are emitted.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    data=Path(a.inputs).read_bytes();rows=[json.loads(x) for x in data.splitlines()]
    result=analyze([x['input_ids'] for x in rows]);result['inputs_sha256']=hashlib.sha256(data).hexdigest()
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write(chr(10))
    print(json.dumps(result,indent=2))
