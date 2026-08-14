#!/usr/bin/env python3
"""Concurrent chat completion benchmark for vLLM OpenAI API (sparkrun-ship).

Measures TTFT + completion tok/s for concurrency levels (default 1,4,10).
Uses stream=true so TTFT is real first-token latency (content or reasoning).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

import urllib.request


PROMPT = (
    "Write a concise technical explanation of what NVFP4 quantization is "
    "and why it helps dense Transformer inference on NVIDIA GB10. Use plain language. "
    "Keep the final answer under 120 words."
)


@dataclass
class RunResult:
    concurrency: int
    worker_id: int
    ok: bool
    status: int | None
    ttft_s: float | None
    total_s: float | None
    completion_tokens: int | None
    prompt_tokens: int | None
    tok_per_s: float | None
    content_chars: int
    reasoning_chars: int
    error: str | None


def one_request(
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    concurrency: int,
    worker_id: int,
) -> RunResult:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "reasoning_effort": "xhigh",
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    ttft = None
    content_chars = 0
    reasoning_chars = 0
    completion_tokens = None
    prompt_tokens = None
    status = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "usage" in chunk and chunk["usage"]:
                    usage = chunk["usage"]
                    completion_tokens = usage.get("completion_tokens")
                    prompt_tokens = usage.get("prompt_tokens")
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                c = delta.get("content") or ""
                r = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if (c or r) and ttft is None:
                    ttft = time.perf_counter() - t0
                content_chars += len(c)
                reasoning_chars += len(r)
        total = time.perf_counter() - t0
        if completion_tokens is None:
            est = max(1, (content_chars + reasoning_chars) // 4)
            completion_tokens = est
        tok_s = (completion_tokens / total) if total > 0 else None
        return RunResult(
            concurrency=concurrency,
            worker_id=worker_id,
            ok=True,
            status=status,
            ttft_s=ttft,
            total_s=total,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            tok_per_s=tok_s,
            content_chars=content_chars,
            reasoning_chars=reasoning_chars,
            error=None,
        )
    except Exception as e:  # noqa: BLE001
        total = time.perf_counter() - t0
        return RunResult(
            concurrency=concurrency,
            worker_id=worker_id,
            ok=False,
            status=status,
            ttft_s=ttft,
            total_s=total,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            tok_per_s=None,
            content_chars=content_chars,
            reasoning_chars=reasoning_chars,
            error=f"{type(e).__name__}: {e}",
        )


def summarize(level: int, runs: list[RunResult]) -> dict[str, Any]:
    ok = [r for r in runs if r.ok]
    fails = [r for r in runs if not r.ok]

    def avg(xs: list[float]) -> float | None:
        return statistics.mean(xs) if xs else None

    def p50(xs: list[float]) -> float | None:
        return statistics.median(xs) if xs else None

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    totals = [r.total_s for r in ok if r.total_s is not None]
    tps = [r.tok_per_s for r in ok if r.tok_per_s is not None]
    comps = [r.completion_tokens for r in ok if r.completion_tokens is not None]
    wall = max((r.total_s or 0.0) for r in runs) if runs else 0.0
    aggregate_tps = None
    if ok and wall > 0 and all(r.completion_tokens is not None for r in ok):
        aggregate_tps = sum(r.completion_tokens or 0 for r in ok) / wall

    return {
        "concurrency": level,
        "requested": len(runs),
        "succeeded": len(ok),
        "failed": len(fails),
        "avg_ttft_s": avg(ttfts),
        "p50_ttft_s": p50(ttfts),
        "avg_total_s": avg(totals),
        "p50_total_s": p50(totals),
        "avg_tok_per_s_per_stream": avg(tps),
        "aggregate_tok_per_s": aggregate_tps,
        "avg_completion_tokens": avg([float(x) for x in comps]) if comps else None,
        "errors": [r.error for r in fails],
        "runs": [asdict(r) for r in runs],
    }


def run_level(
    *,
    base_url: str,
    model: str,
    level: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    print(f"\n=== concurrency={level} ===", flush=True)
    t_wall0 = time.perf_counter()
    results: list[RunResult] = []
    with ThreadPoolExecutor(max_workers=level) as ex:
        futs = [
            ex.submit(
                one_request,
                base_url=base_url,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                concurrency=level,
                worker_id=i,
            )
            for i in range(level)
        ]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            status = "OK" if r.ok else "FAIL"
            ttft_s = f"{r.ttft_s:.3f}" if r.ttft_s is not None else "n/a"
            total_s = f"{r.total_s:.2f}" if r.total_s is not None else "n/a"
            tps = f"{r.tok_per_s:.1f}" if r.tok_per_s is not None else "n/a"
            print(
                f"  worker {r.worker_id}: {status} "
                f"ttft={ttft_s}s total={total_s}s tok/s={tps} "
                f"comp_tok={r.completion_tokens} "
                f"reason_chars={r.reasoning_chars} content_chars={r.content_chars}"
                + (f" err={r.error}" if r.error else ""),
                flush=True,
            )
    results.sort(key=lambda r: r.worker_id)
    summary = summarize(level, results)
    summary["wall_s"] = time.perf_counter() - t_wall0
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="unsloth/Qwen3.8-27B-NVFP4")
    p.add_argument("--levels", default="1,4,10")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", default="bench_results.json")
    args = p.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    models_url = args.base_url.rstrip("/") + "/v1/models"
    with urllib.request.urlopen(models_url, timeout=10) as resp:
        models = json.loads(resp.read().decode())
    print("models:", json.dumps(models)[:300], flush=True)

    if args.warmup > 0:
        print(f"warmup x{args.warmup}", flush=True)
        for i in range(args.warmup):
            r = one_request(
                base_url=args.base_url,
                model=args.model,
                max_tokens=min(64, args.max_tokens),
                temperature=args.temperature,
                timeout=args.timeout,
                concurrency=1,
                worker_id=i,
            )
            print(
                f"  warmup {i}: ok={r.ok} ttft={r.ttft_s} total={r.total_s} err={r.error}",
                flush=True,
            )

    report: dict[str, Any] = {
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "levels": levels,
        "prompt": PROMPT,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [],
    }
    for level in levels:
        report["results"].append(
            run_level(
                base_url=args.base_url,
                model=args.model,
                level=level,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.out}", flush=True)
    for s in report["results"]:
        print(
            json.dumps(
                {
                    "concurrency": s["concurrency"],
                    "succeeded": s["succeeded"],
                    "failed": s["failed"],
                    "avg_ttft_s": s["avg_ttft_s"],
                    "aggregate_tok_per_s": s["aggregate_tok_per_s"],
                    "avg_tok_per_s_per_stream": s["avg_tok_per_s_per_stream"],
                    "wall_s": s["wall_s"],
                    "errors": s["errors"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
