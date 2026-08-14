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
| Image | `eugr/spark-vllm:latest` |
| Quant | `--quantization compressed-tensors` (card) |
| Attention | FlashInfer (auto) |
| Context (package) | `--max-model-len 32768` (card allows 262144) |
| Concurrent | `--max-num-seqs 10` |
| MTP | **off** in package default (safer first boot; checkpoint has `model_mtp.safetensors`) |
| Port | 8000 |

## Quick start

```bash
# optional: pre-download
hf download unsloth/Qwen3.8-27B-NVFP4 --local-dir ~/llm/qwen38-27b-nvfp4

./start.sh
# overrides: MAX_NUM_SEQS=1 MAX_MODEL_LEN=16384 ./start.sh
./stop.sh
```

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

| Conc | OK | Avg TTFT | Aggregate tok/s | Per-stream tok/s |
|-----:|---:|---------:|----------------:|-----------------:|
| 1 | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD |
| 10 | TBD | TBD | TBD | TBD |

Raw: [`bench_results.json`](./bench_results.json) (filled after the release-day run)

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
  --enable-auto-tool-choice
```

## Pitfalls (GB10)

1. **FlashInfer first-boot autotune can wedge the host** (SSH TCP open, banner hangs, `:8000` refused). Prefer minimal first boot (`MAX_NUM_SEQS=1`, modest ctx), poll API from another machine, escalate console/reboot if hung >~10 min.
2. **Dense 27B ≠ 35B-A3B MoE** — do not copy Marlin MoE flags from the sibling package.
3. **Official BF16 is ~52 GiB** and will not leave enough unified-memory headroom on a 121 GiB GB10. Use this NVFP4 checkpoint.
4. **Qwen3.8 is a VLM.** `--language-model-only` is the default to maximize KV-cache headroom for text. Set `LANGUAGE_ONLY=0` only after you confirm vision tensors fit.
5. Optional later: enable MTP only after a stable non-MTP serve is proven on your box.

## Files

| File | Role |
|------|------|
| `start.sh` / `stop.sh` | download + docker serve + health poll |
| `bench_concurrent.py` | streaming 1/4/10 bench |
| `bench_results.json` | measured numbers from tux |

## License

Upstream model: Apache-2.0 (see HF card). This repo is scripts/docs only.
