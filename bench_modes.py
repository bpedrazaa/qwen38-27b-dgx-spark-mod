#!/usr/bin/env python3
"""Four-mode official-sampling bench for Qwen3.8-27B on a live vLLM server.

Modes from the model card:
  thinking off  + instruct sampler
  thinking xhigh / medium / low + thinking sampler

Official sampling:
  Thinking:  temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
             presence_penalty=0.0, repetition_penalty=1.0
  Instruct:  temperature=0.7, top_p=0.80, top_k=20, min_p=0.0,
             presence_penalty=1.5, repetition_penalty=1.0
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
    "and why it helps dense Transformer inference on NVIDIA GB10. "
    "Use plain language. Keep the final answer under 120 words."
)

THINKING_SAMPLER = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}

INSTRUCT_SAMPLER = {
    "temperature": 0.7,
    "top_p": 0.80,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}

MODES = {
    "off": {
        "enable_thinking": False,
        "reasoning_effort": None,
        "sampler": INSTRUCT_SAMPLER,
    },
    "low": {
        "enable_thinking": True,
        "reasoning_effort": "low",
        "sampler": THINKING_SAMPLER,
    },
    "medium": {
        "enable_thinking": True,
        "reasoning_effort": "medium",
        "sampler": THINKING_SAMPLER,
    },
    "xhigh": {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
        "sampler": THINKING_SAMPLER,
    },
}


@dataclass
class RunResult:
    mode: str
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
    finish_reason: str | None
    error: str | None


def one_request(
    *,
    base_url: str,
    model: str,
    mode: str,
    max_tokens: int,
    timeout: float,
    concurrency: int,
    worker_id: int,
    prompt: str,
) -> RunResult:
    cfg = MODES[mode]
    sampler = cfg["sampler"]
    url = base_url.rstrip("/") + "/v1/chat/completions"
    chat_template_kwargs: dict[str, Any] = {
        "enable_thinking": bool(cfg["enable_thinking"]),
        "preserve_thinking": True,
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": sampler["temperature"],
        "top_p": sampler["top_p"],
        "presence_penalty": sampler["presence_penalty"],
        "top_k": sampler["top_k"],
        "min_p": sampler["min_p"],
        "repetition_penalty": sampler["repetition_penalty"],
        "chat_template_kwargs": chat_template_kwargs,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if cfg["reasoning_effort"]:
        body["reasoning_effort"] = cfg["reasoning_effort"]

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
    finish_reason = None
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
                if chunk.get("usage"):
                    usage = chunk["usage"]
                    completion_tokens = usage.get("completion_tokens")
                    prompt_tokens = usage.get("prompt_tokens")
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
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
            mode=mode,
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
            finish_reason=finish_reason,
            error=None,
        )
    except Exception as e:  # noqa: BLE001
        total = time.perf_counter() - t0
        return RunResult(
            mode=mode,
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
            finish_reason=finish_reason,
            error=f"{type(e).__name__}: {e}",
        )


def summarize(mode: str, level: int, runs: list[RunResult]) -> dict[str, Any]:
    ok = [r for r in runs if r.ok]
    fails = [r for r in runs if not r.ok]

    def avg(xs: list[float]) -> float | None:
        return statistics.mean(xs) if xs else None

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    totals = [r.total_s for r in ok if r.total_s is not None]
    tps = [r.tok_per_s for r in ok if r.tok_per_s is not None]
    comps = [r.completion_tokens for r in ok if r.completion_tokens is not None]
    wall = max((r.total_s or 0.0) for r in runs) if runs else 0.0
    aggregate_tps = None
    if ok and wall > 0 and all(r.completion_tokens is not None for r in ok):
        aggregate_tps = sum(r.completion_tokens or 0 for r in ok) / wall
    return {
        "mode": mode,
        "concurrency": level,
        "requested": len(runs),
        "succeeded": len(ok),
        "failed": len(fails),
        "avg_ttft_s": avg(ttfts),
        "avg_total_s": avg(totals),
        "avg_tok_per_s_per_stream": avg(tps),
        "aggregate_tok_per_s": aggregate_tps,
        "avg_completion_tokens": avg([float(x) for x in comps]) if comps else None,
        "avg_content_chars": avg([float(r.content_chars) for r in ok]) if ok else None,
        "avg_reasoning_chars": avg([float(r.reasoning_chars) for r in ok]) if ok else None,
        "finish_reasons": [r.finish_reason for r in ok],
        "sampler": MODES[mode]["sampler"],
        "enable_thinking": MODES[mode]["enable_thinking"],
        "reasoning_effort": MODES[mode]["reasoning_effort"],
        "errors": [r.error for r in fails],
        "runs": [asdict(r) for r in runs],
    }


def run_level(
    *,
    base_url: str,
    model: str,
    mode: str,
    level: int,
    max_tokens: int,
    timeout: float,
    prompt: str,
) -> dict[str, Any]:
    print(f"\n=== mode={mode} concurrency={level} ===", flush=True)
    t_wall0 = time.perf_counter()
    results: list[RunResult] = []
    with ThreadPoolExecutor(max_workers=level) as ex:
        futs = [
            ex.submit(
                one_request,
                base_url=base_url,
                model=model,
                mode=mode,
                max_tokens=max_tokens,
                timeout=timeout,
                concurrency=level,
                worker_id=i,
                prompt=prompt,
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
                f"comp_tok={r.completion_tokens} finish={r.finish_reason} "
                f"reason_chars={r.reasoning_chars} content_chars={r.content_chars}"
                + (f" err={r.error}" if r.error else ""),
                flush=True,
            )
    results.sort(key=lambda r: r.worker_id)
    summary = summarize(mode, level, results)
    summary["wall_s"] = time.perf_counter() - t_wall0
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="unsloth/Qwen3.8-27B-NVFP4")
    p.add_argument("--modes", default="off,low,medium,xhigh")
    p.add_argument("--levels", default="1")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", default="bench_results_modes.json")
    p.add_argument("--prompt", default=PROMPT)
    args = p.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m}; choose from {list(MODES)}")

    if args.warmup:
        print(f"warmup x{args.warmup} (thinking off)", flush=True)
        for i in range(args.warmup):
            r = one_request(
                base_url=args.base_url,
                model=args.model,
                mode="off",
                max_tokens=64,
                timeout=min(args.timeout, 180.0),
                concurrency=1,
                worker_id=i,
                prompt="Reply with the single word ready.",
            )
            print(
                f"  warmup {i}: {'OK' if r.ok else 'FAIL'} "
                f"comp={r.completion_tokens} content={r.content_chars} err={r.error}",
                flush=True,
            )

    payload: dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "started_unix": time.time(),
        "modes": {},
    }
    for mode in modes:
        payload["modes"][mode] = []
        for level in levels:
            payload["modes"][mode].append(
                run_level(
                    base_url=args.base_url,
                    model=args.model,
                    mode=mode,
                    level=level,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    prompt=args.prompt,
                )
            )
    payload["finished_unix"] = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
