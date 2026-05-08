# Running `compare.py` on Colab (L4)

The L4 has 22.5 GB VRAM — enough to hold one multimodal model at a time, not
two. We run **vLLM first**, kill it, then **llama.cpp**, then compare.

vLLM serves it natively in FP16. For llama.cpp you have to pick a GGUF
precision, and the choice changes the question you're answering — see
[§ Choosing the GGUF precision](#choosing-the-gguf-precision) before
downloading.

## Caveats specific to Colab

- **CPU energy is estimated, not measured.** Colab is a VM, so RAPL is not
  exposed. codecarbon falls back to a TDP-based estimate for CPU. GPU energy
  is read from NVML and is real.
- **Grid carbon intensity is geolocated to the Colab VM's IP**, which is not
  where you'd actually deploy. The *ranking* between vLLM and llama.cpp is
  still fair (both pay the same intensity); the absolute kg-CO₂ numbers are
  ballpark.
- **Colab disconnects.** If your runtime dies between the two halves, just
  re-run the cells for whichever half you lost.

## Notebook (one cell per block)

### 1. Confirm L4 + install benchmarking deps

```python
!nvidia-smi | head -20
!pip install -q codecarbon requests huggingface_hub datasets nvidia-ml-py
```

### 2. Drop `compare.py` into the runtime

Either `!git clone` the repo, or direct upload.

---

## Half A — vLLM

### 3. Install vLLM (~3–5 min)

```python
!pip install -q vllm
```

### 4. Start vLLM in the background

```python
import subprocess, time, requests

vllm = subprocess.Popen(
    ["python", "-m", "vllm.entrypoints.openai.api_server",
     "--model", "google/gemma-4-E4B-it",
     "--port", "8000",
     "--max-model-len", "4096",
     "--gpu-memory-utilization", "0.85"],
    stdout=open("vllm.log", "w"), stderr=subprocess.STDOUT,
)

for _ in range(300):  # up to 10 min for first-time weight download
    try:
        if requests.get("http://localhost:8000/v1/models", timeout=2).ok:
            print("vllm ready"); break
    except Exception:
        pass
    time.sleep(2)
else:
    raise RuntimeError("vllm did not come up — check vllm.log")
```

### 5. Benchmark vLLM only

```python
!python compare.py --only vllm \
    --vllm-model "google/gemma-4-E4B-it" \
    --max-tokens 64 --rounds 2
```

Note the printed table — copy it somewhere.

### 6. Stop vLLM and free the GPU

```python
vllm.terminate(); vllm.wait()
!nvidia-smi --query-gpu=memory.used --format=csv  # should be ~0 MiB
```

---

## Half B — llama.cpp

### 7. Build `llama-server` from source with CUDA (~2–3 min)

```python
!apt-get install -y -q cmake ninja-build
!git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp
!cmake -S /content/llama.cpp -B /content/llama.cpp/build \
        -DGGML_CUDA=ON -DLLAMA_CURL=OFF
!cmake --build /content/llama.cpp/build --config Release -j --target llama-server
```

### Choosing the GGUF precision

Two valid framings, each answering a different question. Pick one — or run
both in sequence, swapping just the GGUF file between runs.

| GGUF | Framing | Question answered |
|---|---|---|
| `…-bf16.gguf` | matched precision | Which **engine** is more efficient on identical FP16 math? |
| `…-q4_k_m.gguf` | each tool as deployed | Which **deployment** is cheaper to run in production? |

**The mmproj stays BF16 in both cases.** The vision projector is small, runs
once per image, and quantizing it risks vision quality for negligible
savings — this matches what HF repos publish.

On batch=1 sequential workloads (what this script measures), expect Q4 to
beat F16 substantially: inference is **memory-bandwidth-bound** at low
concurrency, and Q4 reduces HBM reads ~3.6× per token. Dequant gets fused
into the matmul kernel in ggml-cuda, so it doesn't claw the savings back.
Under high-concurrency batched serving the gap narrows because compute
starts to dominate — but that is not what this script benchmarks. See the
[Findings](#findings-this-run) section below for what we actually
measured.

### 8. Download GGUF weights + multimodal projector

```python
from huggingface_hub import hf_hub_download

repo = "Qwen/Qwen2-VL-2B-Instruct-GGUF"
# LLM weights — pick precision based on the framing above:
#   "...-bf16.gguf"     → matched-precision (engine comparison)
#   "...-q4_k_m.gguf"  → as-deployed (deployment comparison)
# Verify the exact filename on the repo before running.
gguf   = hf_hub_download(repo, "qwen2-vl-2b-instruct-q4_k_m.gguf")
mmproj = hf_hub_download(repo, "mmproj-Qwen2-VL-2B-Instruct-f16.gguf")  # always F16
print(gguf); print(mmproj)
```

### 9. Start `llama-server`

```python
import subprocess, time, requests

ll = subprocess.Popen(
    ["/content/llama.cpp/build/bin/llama-server",
     "-m", gguf, "--mmproj", mmproj,
     "-ngl", "99",            # all layers on GPU
     "-c", "4096",
     "--host", "127.0.0.1", "--port", "9090",
     "--alias", "qwen2-vl"],
    stdout=open("llamacpp.log", "w"), stderr=subprocess.STDOUT,
)

for _ in range(120):
    try:
        if requests.get("http://localhost:9090/v1/models", timeout=2).ok:
            print("llama-server ready"); break
    except Exception:
        pass
    time.sleep(2)
else:
    raise RuntimeError("llama-server did not come up — check llamacpp.log")
```

### 10. Benchmark llama.cpp only

```python
!python compare.py --only llamacpp \
    --llamacpp-model qwen2-vl \
    --max-tokens 64 --rounds 2
```

(`qwen2-vl` matches the `--alias` we passed to `llama-server`.)

### 11. Stop the server

```python
ll.terminate(); ll.wait()
```

---

## Compare

Read the tables. **GPU kWh** in the breakdown is the only column with a real
measurement on Colab — CPU/RAM are duration × constant estimates because
RAPL isn't exposed in a VM, so they collapse to identical values whenever
durations match. For per-token economics, also look at **`kWh/1k tok`** and
**`gCO2/1k tok`** in the main table — they normalize for workload.

If you ran both F16 and Q4 GGUFs on llama.cpp, you have two answers:

- **F16 vs vLLM-FP16** → engine-vs-engine efficiency
- **Q4 vs vLLM-FP16** → deployment-vs-deployment economics

If a comparison is under your run-to-run noise floor (≤2% on a single round
is plausible), repeat with `--rounds 5` before drawing conclusions.

## Findings (this run)

30-prompt workload (20 text + 10 image), `--max-tokens 128`, single round,
batch=1 sequential, model `Qwen/Qwen2-VL-2B-Instruct`, hardware Colab L4.

| Server    | Precision | Duration | GPU kWh   | Δ vs vLLM-FP16 |
|-----------|-----------|----------|-----------|----------------|
| vLLM      | FP16      | 22.96 s  | 0.000452  | —              |
| llama.cpp | F16       | 22.93 s  | 0.000455  | +0.7%          |
| llama.cpp | Q4_K_M    | 10.88 s  | 0.000209  | **−54%**       |

**Engine comparison (F16 vs F16):** vLLM and llama.cpp run identical-
precision math in essentially identical time and energy (within 1%, which
is inside single-round measurement noise). At the engine level on
identical math, they are equivalent. Choose between them on operational
grounds (memory footprint, ops complexity, model availability), not
energy.

**Deployment comparison (vLLM-FP16 vs llama.cpp-Q4_K_M):** llama.cpp
serving Q4_K_M is ~2× more energy-efficient per output token than vLLM
serving FP16 on this workload. Wall-clock dropped 53% and GPU energy
dropped 54% — they track within a percentage point, the textbook
signature of a memory-bandwidth-bound workload. GPU *power* stayed
roughly constant; the kernel simply finished faster because fewer bytes
had to be read from HBM (Q4 weights are ~3.6× smaller than F16 weights).

**Conclusion for this regime:** **llama.cpp serving Q4_K_M is the most
economical**, conditional on Q4_K_M quality being acceptable for the
target task. Q4_K_M typically retains >95% of F16 quality on standard
benchmarks, but multimodal tasks (especially OCR) can be more sensitive
than text-only tasks — spot-check your own prompts before committing.

### Limits of this finding — read before generalizing

- **Single-stream only.** This is batch=1, one prompt at a time. Under
  high-concurrency batched serving, vLLM's continuous batching converts
  the workload from memory-bandwidth-bound to compute-bound, and Q4's
  advantage shrinks (sometimes substantially). If your production
  workload is concurrent (an API serving many users), re-run with
  realistic concurrency before deciding.
- **Short outputs.** `max_tokens=128` with mostly factual prompts means
  most prompts stop well before the cap. A workload dominated by long
  generations (essay writing, long code) would put more weight on decode
  and could change the ratio.
- **L4 specifically.** On a beefier compute-rich GPU (H100), the
  bandwidth/compute ratio shifts and Q4's advantage typically narrows.
  On a weaker bandwidth-limited GPU (T4, consumer cards), it can widen.
- **Colab measurement caveats apply.** Only the GPU column is real; CPU
  and RAM are TDP-times-duration estimates because RAPL isn't exposed
  in the VM. The conclusions above use only the GPU column.
