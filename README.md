# unsloth/Qwen3.8-27B-NVFP4 on DGX Spark (GB10)

**Deployment package only** — not new weights.
**HF:** [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) (Unsloth Dynamic V3 NVFP4 of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B))
**Host tested:** ASUS Ascent GX10 / NVIDIA GB10 (aarch64), 2026-08-14

This is a **model-specific** recipe for the dense 27B NVFP4 checkpoint.
Do **not** reuse the 35B-A3B MoE package flags (`--moe-backend marlin`, etc.).

Qwen3.8-27B is a native vision-language dense model (64 layers, Gated DeltaNet + gated attention, 262K native context, MTP-trained). The BF16 card is ~52 GiB; this package uses the Unsloth NVFP4 quant (~22.5 GiB weights + 0.81 GiB MTP) so a single GB10 can actually serve it.

## Hardware / runtime

| Item | Value |
|------|--------|
| GPU | NVIDIA GB10 (SM121) |
| Image | `eugr/spark-vllm:latest` (package default) |
| Measured runtime | local vLLM **0.26.0** (`Qwen3_5MTP`) |
| Quant | `--quantization compressed-tensors` (card) |
| Attention | FlashInfer (auto) |
| Context (package) | `--max-model-len 32768` (card allows 262144) |
| Concurrent | `--max-num-seqs 10` (**stable 10/10**) |
| MTP | **on** — `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` |
| Port | 8000 |

Release-day note: official BF16 hung on HF Xet; this package uses the Unsloth NVFP4 checkpoint that actually fits a 121 GiB GB10.

Disable MTP only if you want the safer first-boot path: `ENABLE_MTP=0 ./start.sh`.

## Quick start

```bash
# optional: pre-download
hf download unsloth/Qwen3.8-27B-NVFP4 --local-dir ~/llm/qwen38-27b-nvfp4

./start.sh
# overrides: MAX_NUM_SEQS=1 MAX_MODEL_LEN=16384 ENABLE_MTP=0 ./start.sh
./stop.sh
```

On this host the measured serve was local vLLM 0.26.0 via `./start-local.sh` (same flags).

Smoke:

```bash
curl -s http://127.0.0.1:8000/v1/models | head
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.8-27B-NVFP4",
    "messages": [{"role":"user","content":"Say hi in one sentence."}],
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Reasoning field: vLLM `qwen3` parser. Clients can set `chat_template_kwargs.enable_thinking`.

## Measured benchmarks (this host)

Streaming chat, `max_tokens=256`, levels 1/4/10, thinking tokens counted via usage when present.
Host: ASUS Ascent GX10 / NVIDIA GB10. Runtime: local vLLM `0.26.0`, NVFP4 via `compressed-tensors`, thinking on (default). Date: 2026-08-14.

### MTP on (package default, k=3)

| Conc | OK | Avg TTFT | Aggregate tok/s | Per-stream tok/s |
|-----:|---:|---------:|----------------:|-----------------:|
| 1 | 1/1 | 0.27s | **19.6** | 19.6 |
| 4 | 4/4 | 1.48s | **67.5** | 17.3 |
| 10 | 10/10 | 1.05s | **131.9** | 14.8 |

- **Warm single-stream headline:** ~**20 tok/s**
- **10 concurrent:** **10/10 stable**, ~**132 aggregate tok/s**
- **MTP accept:** ~**54%** of draft tokens (`2418/4461` after subtracting the smoke request; vLLM end counters 4464 draft / 2419 accepted). Per-position accepts: p0=1114, p1=796, p2=509.
- Bench prompt was technical; streams spent budget in `reasoning` (content empty at short cap) — tok/s still from `completion_tokens`
- Coherence sample (thinking off): `Hello!`

### MTP off (same host, earlier same day)

| Conc | OK | Avg TTFT | Aggregate tok/s | Per-stream tok/s |
|-----:|---:|---------:|----------------:|-----------------:|
| 1 | 1/1 | 0.13s | 10.8 | 10.8 |
| 4 | 4/4 | 0.28s | 40.3 | 10.1 |
| 10 | 10/10 | 0.51s | 92.0 | 9.2 |

MTP is the speed lever: **+82%** single-stream (10.8 → 19.6), **+43%** at 10-wide aggregate (92.0 → 131.9). TTFT is a bit worse with speculation on.

Raw: [`bench_results.json`](./bench_results.json) (MTP on). MTP-off snapshot: [`bench_results_mtp_off.json`](./bench_results_mtp_off.json).

## Exact serve command (package default)

```bash
vllm serve /models/qwen38-27b-nvfp4 \
  --served-model-name unsloth/Qwen3.8-27B-NVFP4 \
  --host 0.0.0.0 --port 8000 \
  --language-model-only \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --max-model-len 32768 \
  --max-num-seqs 10 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.70 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

vLLM 0.26.0 resolves the draft as `Qwen3_5MTP` from `text_config.mtp_num_hidden_layers=1` and `model_mtp.safetensors`. `qwen3_5_mtp` is accepted as an alias but is deprecated in favor of `method=mtp`.

## Pitfalls (GB10)

1. **FlashInfer first-boot autotune can wedge the host** (SSH TCP open, banner hangs, `:8000` refused). Prefer minimal first boot (`MAX_NUM_SEQS=1`, modest ctx), poll API from another machine, escalate console/reboot if hung >~10 min.
2. **Dense 27B ≠ 35B-A3B MoE** — do not copy Marlin MoE flags from the sibling package.
3. **Official BF16 is ~52 GiB** and will not leave enough unified-memory headroom on a 121 GiB GB10. Use this NVFP4 checkpoint.
4. **Qwen3.8 is a VLM.** `--language-model-only` is the default to maximize KV-cache headroom for text. Set `LANGUAGE_ONLY=0` only after you confirm vision tensors fit.
5. **Local vLLM 0.26.0 needs `ninja` on PATH** for FlashInfer sampling JIT (`apt install ninja-build`). Without it the engine dies after compile during warmup.
6. **MTP k>1 reuses the single trained MTP layer.** vLLM warns this can lower acceptance vs k=1. Measured k=3 still beat MTP-off by a wide margin (~54% accept).
7. Need mostly free GB10 memory before launch. This box had ~89 GiB free after stopping the previous serve.

## Files

| File | Role |
|------|------|
| `start.sh` / `stop.sh` | download + docker serve + health poll (MTP on by default) |
| `start-local.sh` / `stop-local.sh` | host vLLM 0.26.0 path used for the measured numbers |
| `bench_concurrent.py` | streaming 1/4/10 bench |
| `bench_results.json` | measured MTP-on numbers from tux |
| `bench_results_mtp_off.json` | same-day MTP-off baseline |

## License

Upstream model: Apache-2.0 (see HF card). This repo is scripts/docs only.
