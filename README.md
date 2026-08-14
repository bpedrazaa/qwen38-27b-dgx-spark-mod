# Qwen 3.8 27B on DGX Spark

Run [Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B) on a single NVIDIA GB10 using the [Unsloth NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) checkpoint and vLLM.

This repo is a deploy package: start/stop scripts, a measured serve command, and GB10 benchmarks. It does not republish model weights.

Qwen 3.8 27B is a native vision-language model: dense 27B hybrid (Gated DeltaNet + full attention), 262,144-token context, trained MTP, and per-request thinking control (`enable_thinking` plus `reasoning_effort` of `xhigh` / `medium` / `low`). Official BF16 is about 52 GiB. The Unsloth NVFP4 build is about 22.5 GiB plus 0.81 GiB of MTP weights, which fits a 121 GiB GB10 with room for native context, vision, and speculation.

## Defaults

| Item | Value |
|------|--------|
| GPU | NVIDIA GB10 |
| Image | `eugr/spark-vllm:latest` |
| Measured runtime | vLLM 0.26.0 |
| Quantization | `compressed-tensors` (NVFP4) |
| Context | `--max-model-len 262144` |
| Vision | on (`image` + `video`) |
| Concurrency | `--max-num-seqs 4` |
| Speculative decoding | MTP, `k=3` |
| Port | 8000 |

Thinking is on by default (`reasoning_effort=xhigh`). Disable it per request with `chat_template_kwargs.enable_thinking = false`. Start without MTP with `ENABLE_MTP=0 ./start.sh`. Start text-only with `LANGUAGE_ONLY=1 ./start.sh`.

## Quick start

```bash
hf download unsloth/Qwen3.8-27B-NVFP4 --local-dir ~/llm/qwen38-27b-nvfp4

./start.sh
# optional overrides:
# MAX_NUM_SEQS=1 MAX_MODEL_LEN=32768 ENABLE_MTP=0 LANGUAGE_ONLY=1 ./start.sh
./stop.sh
```

`./start.sh` pulls `eugr/spark-vllm:latest` and serves the model in Docker. If you already have vLLM 0.26.0 on the host, `./start-local.sh` uses the same flags.

Text smoke:

```bash
curl -s http://127.0.0.1:8000/v1/models

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.8-27B-NVFP4",
    "messages": [{"role":"user","content":"Say hi in one sentence."}],
    "max_tokens": 64,
    "temperature": 0.7,
    "top_p": 0.8,
    "presence_penalty": 1.5,
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Vision smoke:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.8-27B-NVFP4",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image in one sentence."},
        {"type": "image_url", "image_url": {"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"}}
      ]
    }],
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## Official sampling

From the [Qwen 3.8 27B model card](https://huggingface.co/Qwen/Qwen3.8-27B):

| Mode | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|------|------------:|------:|------:|------:|-----------------:|-------------------:|
| Thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Instruct / thinking off | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

Thinking depth is `reasoning_effort`: `xhigh` (default), `medium`, or `low`.

## Benchmarks

Measured on an ASUS Ascent GX10 / NVIDIA GB10, 2026-08-14.

Live serve: vLLM 0.26.0, Unsloth NVFP4, MTP `k=3`, `--max-model-len 262144`, vision enabled, `--max-num-seqs 4`. Streaming chat, official sampling per mode, `max_tokens=4096`. Every run finished with `stop` (not the length cap). Throughput uses `completion_tokens`.

Same prompt for all four modes: a short technical explanation of NVFP4 on GB10, final answer under 120 words.

| Mode | Sampler | Success | TTFT | tok/s | Completion tokens | Reasoning chars | Content chars | Finish |
|------|---------|--------:|-----:|------:|------------------:|----------------:|--------------:|--------|
| Thinking off | instruct | 1/1 | 0.29s | **16.9** | 154 | 0 | 819 | `stop` |
| Thinking `low` | thinking | 1/1 | 0.29s | **19.3** | 394 | 1,149 | 673 | `stop` |
| Thinking `medium` | thinking | 1/1 | 0.35s | **23.5** | 1,644 | 5,034 | 759 | `stop` |
| Thinking `xhigh` | thinking | 1/1 | 0.28s | **22.0** | 804 | 2,391 | 599 | `stop` |

Decode speed stays in the same band across modes. What changes is how long the model thinks before it answers. `medium` produced the longest reasoning trace on this prompt.

Vision was verified on the same serve: a Statue of Liberty photo returned a correct one-sentence description (`52` completion tokens, `1,675` prompt tokens including image features).

This serve advertised `max_model_len=262144`. vLLM reported **81.1 GiB** KV cache, **2,287,535** GPU KV tokens, and **8.73x** concurrency at 262,144 tokens per request.

Raw four-mode results: [`bench_results_modes.json`](./bench_results_modes.json). Re-run with `./bench_modes.py`.

## Serve command

```bash
vllm serve /models/qwen38-27b-nvfp4 \
  --served-model-name unsloth/Qwen3.8-27B-NVFP4 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image":4,"video":1}' \
  --media-io-kwargs '{"video":{"num_frames":-1}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

vLLM loads the native MTP head from `model_mtp.safetensors` and the vision encoder from the same NVFP4 checkpoint.

## Notes

- **Use the NVFP4 checkpoint.** Official BF16 is ~52 GiB and does not leave enough unified-memory headroom on a 121 GiB GB10.
- **First boot can take several minutes.** FlashInfer / Triton compile and autotune on the first launch. If the API is not up after ~10 minutes, check `docker logs` (or the local vLLM log) before assuming it hung.
- **Host vLLM needs `ninja`.** FlashInfer's sampling JIT fails during warmup unless `ninja-build` is on `PATH` (`sudo apt install ninja-build`).
- **Vision is on by default.** The NVFP4 file includes the vision tensors. `LANGUAGE_ONLY=1` drops multimodal inputs if you want a text-only process.
- **Native context is 262,144.** YaRN to 1M is documented on the model card; do not enable it unless you actually need longer than 256K.
- **MTP `k=3` reuses one trained draft layer.** vLLM may warn that acceptance can be lower than `k=1`.
- The GPU should be mostly free before launch.

## Files

| File | Role |
|------|------|
| `start.sh` / `stop.sh` | Download + Docker serve + health poll (256K, vision, MTP on) |
| `start-local.sh` / `stop-local.sh` | Host vLLM 0.26.0 path used for the measured numbers |
| `bench_modes.py` | Official-sampler sweep: thinking off / low / medium / xhigh |
| `bench_results_modes.json` | Four-mode results |
| `bench_concurrent.py` | Streaming concurrency bench |
| `bench_results.json` | Earlier MTP-on concurrency capture |
| `bench_results_mtp_off.json` | Earlier MTP-off concurrency capture |

## License

Upstream model: Apache-2.0 (see the Hugging Face cards). This repo is scripts and docs only.
