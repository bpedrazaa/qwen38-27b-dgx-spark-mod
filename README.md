# Qwen 3.8 27B + DFlash 2 on DGX Spark

Run Qwen 3.8 27B on one NVIDIA GB10 with [Inco DFlash 2](https://inco.ai/blog/dflash2/) speculative decoding, 262K context, vision, reasoning controls, and tool calling.

This is a public deployment package, not a weight mirror. It contains the pinned vLLM compatibility patch, Docker build, start/stop scripts, a reproducible speculation benchmark, and measured DGX Spark results.

## Result

DFlash 2 is the best default we measured for ordinary, unpredictable generation. On the same standard Qwen3.8-27B NVFP4 target, prompt, sampling, and `k=7` configuration:

| Workload | DSpark `k=7` | DFlash 2 `k=7` | Difference |
|---|---:|---:|---:|
| Fresh code generation | 31.95 tok/s | **44.46 tok/s** | **+39.2%** |
| Draft acceptance | 41.1% | **64.4%** | +23.3 points |
| Tokens per target pass | 3.87 | **5.51** | **+42.4%** |
| Highly copyable edit | 60.91 tok/s | **60.92 tok/s** | tied |
| Edit acceptance | 98.5% | **99.3%** | +0.8 points |

The improvement is real but workload-dependent. Our deeper DSpark `k=14` configuration remains faster when output is almost entirely copied from the prompt:

| Standard target | DSpark `k=14` | DFlash 2 `k=7` |
|---|---:|---:|
| Fresh generation | 29.72 tok/s | **44.46 tok/s (+49.6%)** |
| Highly copyable edit, warm | **75.46 tok/s** | 60.67 tok/s |

### Uncensored target

DFlash 2 also works with [`joshebbs/qwen3.8-27b-uncensored-nvfp4-modelopt`](https://huggingface.co/joshebbs/qwen3.8-27b-uncensored-nvfp4-modelopt):

| Workload | Result |
|---|---:|
| Fresh code generation | **41.38 tok/s** |
| Fresh draft acceptance | 53.8% |
| Fresh tokens per target pass | 4.76 |
| Highly copyable edit | **64.18 / 64.09 tok/s** |
| Edit acceptance | 99.6% |
| Edit tokens per target pass | 7.97 |

Against our prior uncensored DSpark `k=14` serve, DFlash 2 is **52.7% faster** on fresh generation (41.38 vs 27.10 tok/s), while DSpark remains faster on the copy-heavy edit (84.32 vs 64.09 tok/s).

Raw machine-readable data: [`bench_results_dflash2.json`](./bench_results_dflash2.json).

## Measured configuration

| Item | Value |
|---|---|
| Hardware | NVIDIA DGX Spark / GB10 |
| Runtime | vLLM 0.27.1 + upstream DFlash 2 PR `#52816` |
| Target quantization | NVFP4 / compressed-tensors |
| DFlash 2 drafter | `incoai/Qwen3.8-27B-DFlash2` |
| Speculative block | `k=7`, probabilistic sampling |
| Served model name | `Qwen3.8-27B` |
| Native context | 262,144 tokens |
| Max sequences | 10 |
| Max batched tokens | 16,384 |
| GPU memory utilization | 0.85 |
| Vision | enabled, up to 2 images per prompt |
| Reasoning parser | `qwen3` |
| Tool parser | `qwen3_xml` |
| Prefix cache | enabled for the deployment; disabled for fair drafter benchmarks |
| Port | 8000 |

## Quick start

Requirements: an aarch64 DGX Spark/GB10 host with NVIDIA Container Toolkit, Docker, and enough free unified memory for the target and drafter.

### 1. Build the DFlash 2 vLLM image

```bash
git clone https://github.com/gitcommit90/qwen38-27b-dgx-spark
cd qwen38-27b-dgx-spark

docker build \
  -f Dockerfile.dflash2 \
  -t qwen38-dflash2:v0.27.1-aarch64 .
```

The Dockerfile starts from `vllm/vllm-openai:v0.27.1-aarch64` and applies the runtime portion of upstream vLLM PR `#52816`, pinned to commit `19c9351904df4c63042671bc67a866ca48dc7d6f`.

It also removes one overly broad type guard in the PR. The measured Qwen NVFP4 checkpoint is quantized overall, but the DFlash-facing `lm_head` weight is BF16/unquantized; candidate TopK works correctly after removing that false rejection. The patch is explicit and inspectable in [`Dockerfile.dflash2`](./Dockerfile.dflash2).

### 2. Start the uncensored NVFP4 target

```bash
export HF_TOKEN=hf_your_token
./start-dflash2.sh
```

Defaults:

- target: `joshebbs/qwen3.8-27b-uncensored-nvfp4-modelopt`
- drafter: `incoai/Qwen3.8-27B-DFlash2`
- API model name: `Qwen3.8-27B`
- context: 262,144
- `k=7`
- binds `0.0.0.0:8000`
- restart policy: `unless-stopped`

The script can use Hugging Face model IDs through its mounted cache, or already-downloaded directories:

```bash
TARGET_DIR=$HOME/llm/qwen38-27b-uncensored-nvfp4 \
DRAFT_DIR=$HOME/llm/qwen38-dflash2 \
./start-dflash2.sh
```

Use a different compatible target with `TARGET_MODEL` or `TARGET_DIR`:

```bash
TARGET_MODEL=your-org/your-qwen38-27b-nvfp4 ./start-dflash2.sh
```

Stop it with:

```bash
./stop-dflash2.sh
```

### Useful overrides

```bash
MAX_MODEL_LEN=32768 \
MAX_NUM_SEQS=4 \
GPU_MEM_UTIL=0.85 \
NUM_SPECULATIVE_TOKENS=7 \
PORT=8000 \
./start-dflash2.sh
```

For a custom chat template:

```bash
CHAT_TEMPLATE=$PWD/chat_template.jinja ./start-dflash2.sh
```

## API checks

Text:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B",
    "messages": [{"role":"user","content":"Reply with exactly: DFLASH2 OK"}],
    "max_tokens": 32,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Reasoning depth is controlled per request with `chat_template_kwargs.enable_thinking` and `reasoning_effort` (`low`, `medium`, or `xhigh`).

Tool calling:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B",
    "messages": [{"role":"user","content":"What is the weather in Seattle?"}],
    "tools": [{
      "type":"function",
      "function": {
        "name":"get_weather",
        "description":"Get current weather",
        "parameters": {
          "type":"object",
          "properties":{"city":{"type":"string"}},
          "required":["city"]
        }
      }
    }],
    "max_tokens": 128
  }'
```

Vision uses standard OpenAI-compatible `image_url` message content. The measured serve correctly handled image input with `--limit-mm-per-prompt.image 2`.

## Reproduce the DFlash 2 benchmark

The fair DFlash 2 vs DSpark comparison used:

- same standard NVFP4 target
- same prompts and output lengths
- `k=7` for both drafters
- 32K context
- prefix caching disabled
- temperature 0
- one fresh-generation workload
- one deliberately copy-heavy edit workload, repeated twice

Run:

```bash
python3 bench_speculation.py http://127.0.0.1:8000/v1 Qwen3.8-27B
```

The script reads vLLM's `/metrics` counters and reports:

- output tok/s
- accepted draft-token percentage
- mean emitted tokens per target-model pass

Do not generalize the edit-heavy number to all generation. It is intentionally favorable to speculative decoding because nearly the entire answer already appears in the prompt. Fresh generation is the more representative result for ordinary coding and agent work.

## DFlash 2 vs MTP and DSpark

- **DFlash 2 `k=7`** is the recommended default for general generation on this setup.
- **DSpark `k=14`** remains useful for highly predictable rewrites, boilerplate, and copy-heavy edits.
- **Native MTP `k=3`** needs no external drafter and remains the simplest fallback.

The older MTP deployment scripts remain available as `start.sh` / `stop.sh`. Their historical GB10 benchmark files are retained in this repository for comparison.

## Previous native-MTP measurements

Before DFlash 2, Qwen3.8-27B NVFP4 was measured with vLLM 0.26.0, native MTP `k=3`, 262K context, vision, and `--max-num-seqs 4`:

- four-mode average single-stream decode: **21.4 tok/s**
- default-thinking aggregate at 4 concurrent requests: **66.5 tok/s**
- all 20 four-mode requests finished with `stop`
- native 262K context and vision were verified

Raw files:

- [`bench_results_modes_5x.json`](./bench_results_modes_5x.json)
- [`bench_results_concurrent_4096.json`](./bench_results_concurrent_4096.json)
- [`bench_results_mtp_off.json`](./bench_results_mtp_off.json)

## Files

| File | Role |
|---|---|
| `Dockerfile.dflash2` | Reproducible aarch64 vLLM 0.27.1 + DFlash 2 image |
| `dflash2-vllm.patch` | Pinned runtime patch from upstream vLLM PR `#52816` |
| `start-dflash2.sh` / `stop-dflash2.sh` | DFlash 2 deployment with health polling and persistence |
| `bench_speculation.py` | Fresh-generation and copy-heavy speculation benchmark |
| `bench_results_dflash2.json` | Measured DFlash 2, DSpark, and uncensored-target results |
| `start.sh` / `stop.sh` | Legacy Docker native-MTP deployment |
| `start-local.sh` / `stop-local.sh` | Legacy host-vLLM native-MTP deployment |
| `bench_modes_5x.py` | Historical official-sampler reasoning-mode sweep |
| `bench_concurrent.py` | Historical 1/4/10 concurrency benchmark |

## Notes

- DFlash 2 requires a compatible target tokenizer and architecture. A fine-tune can work, as the uncensored target did here, but test acceptance and output correctness before treating an arbitrary checkpoint as compatible.
- First launch downloads both models and compiles kernels. It can take several minutes.
- The full 262K setting reserves substantial KV cache. Lower `MAX_MODEL_LEN` if you need more memory headroom or are comparing drafters under a smaller fixed context.
- Keep `k` fixed when comparing draft methods. Comparing DFlash 2 `k=7` directly to DSpark `k=14` answers a deployment question, not a drafter-quality question.
- The target model remains authoritative. Lossless speculative decoding accelerates output without replacing target-model token probabilities.

## License

The deployment code and documentation in this repository do not redistribute weights. Follow the licenses on the target and drafter Hugging Face repositories. vLLM is Apache-2.0.
