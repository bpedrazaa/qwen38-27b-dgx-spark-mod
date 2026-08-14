# Qwen 3.8 27B on DGX Spark

Run [Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B) on a single NVIDIA GB10 using the [Unsloth NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) checkpoint and vLLM.

This repo is a deploy package: start/stop scripts, a measured serve command, and GB10 benchmarks. It does not republish model weights.

Qwen 3.8 27B is a dense 27B hybrid model (Gated DeltaNet + full attention, 262K native context, trained with MTP). The official BF16 checkpoint is about 52 GiB and is a poor fit for 121 GiB of unified memory. The Unsloth NVFP4 build is about 22.5 GiB plus 0.81 GiB of MTP weights, which leaves room for context, batching, and speculation.

## Defaults

| Item | Value |
|------|--------|
| GPU | NVIDIA GB10 |
| Image | `eugr/spark-vllm:latest` |
| Measured runtime | vLLM 0.26.0 |
| Quantization | `compressed-tensors` (NVFP4) |
| Context | `--max-model-len 32768` |
| Concurrency | `--max-num-seqs 10` |
| Speculative decoding | MTP, `k=3` |
| Port | 8000 |

To start without MTP: `ENABLE_MTP=0 ./start.sh`.

## Quick start

```bash
hf download unsloth/Qwen3.8-27B-NVFP4 --local-dir ~/llm/qwen38-27b-nvfp4

./start.sh
# optional overrides:
# MAX_NUM_SEQS=1 MAX_MODEL_LEN=16384 ENABLE_MTP=0 ./start.sh
./stop.sh
```

`./start.sh` pulls `eugr/spark-vllm:latest` and serves the model in Docker. If you already have vLLM 0.26.0 on the host, `./start-local.sh` uses the same flags.

Smoke test:

```bash
curl -s http://127.0.0.1:8000/v1/models

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.8-27B-NVFP4",
    "messages": [{"role":"user","content":"Say hi in one sentence."}],
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Thinking is on by default. Disable it with `chat_template_kwargs.enable_thinking = false`. The serve command enables the vLLM `qwen3` reasoning parser.

## Benchmarks

Measured on an ASUS Ascent GX10 / NVIDIA GB10, 2026-08-14.

Streaming chat completions, `max_tokens=256`, thinking on. Throughput uses `completion_tokens`. Runtime: vLLM 0.26.0, NVFP4 via `compressed-tensors`.

### MTP on (default, k=3)

| Concurrent | Success | Avg TTFT | Aggregate tok/s | Per-stream tok/s |
|-----------:|--------:|---------:|----------------:|-----------------:|
| 1 | 1/1 | 0.27s | **19.6** | 19.6 |
| 4 | 4/4 | 1.48s | **67.5** | 17.3 |
| 10 | 10/10 | 1.05s | **131.9** | 14.8 |

- Single stream: **~20 tok/s**
- 10 concurrent: **10/10 stable**, **~132 tok/s** aggregate
- MTP acceptance: **~54%** (2418 / 4461 draft tokens)

### MTP off

| Concurrent | Success | Avg TTFT | Aggregate tok/s | Per-stream tok/s |
|-----------:|--------:|---------:|----------------:|-----------------:|
| 1 | 1/1 | 0.13s | 10.8 | 10.8 |
| 4 | 4/4 | 0.28s | 40.3 | 10.1 |
| 10 | 10/10 | 0.51s | 92.0 | 9.2 |

MTP is the speed lever: **+82%** single-stream (10.8 → 19.6 tok/s) and **+43%** at 10-wide (92.0 → 131.9 tok/s). Time to first token is a bit higher with speculation enabled.

Raw results: [`bench_results.json`](./bench_results.json) (MTP on), [`bench_results_mtp_off.json`](./bench_results_mtp_off.json).

## Serve command

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

vLLM loads the native MTP head from `model_mtp.safetensors`.

## Notes

- **Use the NVFP4 checkpoint.** Official BF16 is ~52 GiB and does not leave enough unified-memory headroom on a 121 GiB GB10.
- **First boot can take several minutes.** FlashInfer / Triton compile and autotune on the first launch. If the API is not up after ~10 minutes, check `docker logs` (or the local vLLM log) before assuming it hung.
- **Host vLLM needs `ninja`.** FlashInfer's sampling JIT fails during warmup unless `ninja-build` is on `PATH` (`sudo apt install ninja-build`).
- **This is a VLM served as text-only.** `--language-model-only` is the default so more memory stays available for KV cache. Set `LANGUAGE_ONLY=0` only if you need vision and have confirmed the extra tensors fit.
- **MTP `k=3` reuses one trained draft layer.** vLLM may warn that acceptance can be lower than `k=1`. Measured `k=3` still beat MTP-off by a wide margin.
- The GPU should be mostly free before launch.

## Files

| File | Role |
|------|------|
| `start.sh` / `stop.sh` | Download + Docker serve + health poll (MTP on by default) |
| `start-local.sh` / `stop-local.sh` | Host vLLM 0.26.0 path used for the measured numbers |
| `bench_concurrent.py` | Streaming 1 / 4 / 10 concurrent bench |
| `bench_results.json` | MTP-on results |
| `bench_results_mtp_off.json` | Same-day MTP-off baseline |

## License

Upstream model: Apache-2.0 (see the Hugging Face cards). This repo is scripts and docs only.
