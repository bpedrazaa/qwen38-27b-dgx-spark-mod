#!/usr/bin/env python3
"""Five sequential repetitions of each official Qwen 3.8 thinking mode."""
import json, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from bench_modes import MODES, PROMPT, one_request

BASE="http://127.0.0.1:8000"
MODEL="unsloth/Qwen3.8-27B-NVFP4"
MAX_TOKENS=4096
REPETITIONS=5
# Rotate order each round so warm/cache/time drift does not consistently favor one mode.
base_modes=["off","low","medium","xhigh"]
out={"model":MODEL,"base_url":BASE,"prompt":PROMPT,"max_tokens":MAX_TOKENS,
     "repetitions_per_mode":REPETITIONS,"execution":"sequential, rotating mode order",
     "started_unix":time.time(),"modes":{m:[] for m in base_modes}}
path=Path(__file__).with_name("bench_results_modes_5x.json")

def save():
    out["updated_unix"]=time.time()
    path.write_text(json.dumps(out,indent=2)+"\n")

for rep in range(REPETITIONS):
    order=base_modes[rep:]+base_modes[:rep]
    for mode in order:
        print(f"rep={rep+1}/{REPETITIONS} mode={mode}",flush=True)
        r=one_request(base_url=BASE,model=MODEL,mode=mode,max_tokens=MAX_TOKENS,
                      timeout=1800,concurrency=1,worker_id=rep,prompt=PROMPT)
        d=r.__dict__.copy()
        d["repetition"]=rep+1
        d["decode_tok_per_s_excl_ttft"]=(r.completion_tokens/(r.total_s-r.ttft_s)
            if r.ok and r.completion_tokens and r.total_s and r.ttft_s is not None and r.total_s>r.ttft_s else None)
        out["modes"][mode].append(d)
        save()
        print(f"  ok={r.ok} tokens={r.completion_tokens} total={r.total_s:.2f}s ttft={r.ttft_s:.3f}s e2e={r.tok_per_s:.2f} decode={d['decode_tok_per_s_excl_ttft']:.2f} finish={r.finish_reason}",flush=True)

summaries={}
for mode,runs in out["modes"].items():
    ok=[r for r in runs if r["ok"]]
    def stats(key):
        xs=[r[key] for r in ok if r.get(key) is not None]
        return {"mean":statistics.mean(xs),"median":statistics.median(xs),"min":min(xs),"max":max(xs),"stdev":statistics.stdev(xs) if len(xs)>1 else 0}
    summaries[mode]={"succeeded":len(ok),"requested":len(runs),
        "e2e_tok_per_s":stats("tok_per_s"),
        "decode_tok_per_s_excl_ttft":stats("decode_tok_per_s_excl_ttft"),
        "completion_tokens":stats("completion_tokens"),
        "ttft_s":stats("ttft_s"),
        "finish_reasons":[r["finish_reason"] for r in ok],
        "sampler":MODES[mode]["sampler"]}
out["summaries"]=summaries
mode_means=[summaries[m]["decode_tok_per_s_excl_ttft"]["mean"] for m in base_modes]
out["headline_average_decode_tok_per_s"]={"method":"unweighted arithmetic mean of the four mode mean decode rates (completion tokens / (total time - TTFT))","value":statistics.mean(mode_means)}
out["finished_unix"]=time.time(); save()
print(json.dumps({"summaries":summaries,"headline":out["headline_average_decode_tok_per_s"]},indent=2),flush=True)
