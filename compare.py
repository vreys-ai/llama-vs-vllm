#!/usr/bin/env python3
"""
Compare inference energy & CO2 between a vLLM server and a llama.cpp server.

Workload: a sample of POPE (lmms-lab/POPE), an object-hallucination
multimodal benchmark — yes/no questions about whether an object exists
in a given image. Streamed via the `datasets` library so only the rows
we use are downloaded.

Both servers must already be running and expose an OpenAI-compatible
/v1/chat/completions endpoint. The script sends the same prompts to each
and uses codecarbon to measure energy consumption and CO2 emissions.

Usage:
    python compare.py \\
        --vllm-url http://localhost:8000 \\
        --vllm-model google/gemma-4-E4B-it \\
        --llamacpp-url http://localhost:9090 \\
        --llamacpp-model gemma4-e4b

Caveats:
    * codecarbon measures whole-machine power, not just the server process.
      Only run one server at a time during the benchmark, or the idle one
      inflates the other's reading.
    * vLLM typically serves FP16/BF16 weights; llama.cpp typically serves
      quantized GGUF. The comparison is "deployment-as-configured", not
      identical weights.
"""

import argparse
import base64
import io
import time

import requests
from codecarbon import EmissionsTracker
from datasets import load_dataset


POPE_DATASET = "lmms-lab/POPE"
POPE_CONFIG = "Full"
POPE_SPLIT = "random"


SAMPLING_TEMP = 1.0
SAMPLING_TOP_P = 0.95
SAMPLING_TOP_K = 64


def load_pope_samples(n):
    """Stream N samples from POPE; pre-encode each image to a JPEG data URL.

    Returns list of {"question": str, "image_url": data-URL}. Streaming
    via `datasets` pulls only the bytes needed for N rows, not the full
    split. Pre-encoding once avoids repeating the JPEG/base64 work on
    every round.
    """
    print(f"Streaming {n} samples from {POPE_DATASET} "
          f"({POPE_CONFIG}/{POPE_SPLIT})...")
    ds = load_dataset(POPE_DATASET, POPE_CONFIG,
                      split=POPE_SPLIT, streaming=True)
    samples = []
    for row in ds.take(n):
        img = row["image"].convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        data_url = ("data:image/jpeg;base64,"
                    + base64.b64encode(buf.getvalue()).decode())
        samples.append({"question": row["question"], "image_url": data_url})
    print(f"Loaded {len(samples)} samples.")
    return samples


def build_messages(sample):
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": sample["question"]},
            {"type": "image_url", "image_url": {"url": sample["image_url"]}},
        ],
    }]


def send(url, model, messages, max_tokens):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": SAMPLING_TEMP,
        "top_p": SAMPLING_TOP_P,
        "top_k": SAMPLING_TOP_K,
        # Belt-and-suspenders stop string. Some GGUFs (notably Gemma 4 from
        # Unsloth) don't expose the turn-end token id correctly in their EOG
        # metadata, so llama.cpp doesn't stop on it natively. Matching on
        # the decoded text catches it. No-op for models that don't emit
        # this literal string (e.g. Qwen).
        #"stop": ["<end_of_turn>"],
        "stop": ["<end_of_turn>", "<eos>", "<turn|>"],
    }
    r = requests.post(
        url.rstrip("/") + "/v1/chat/completions",
        json=payload, timeout=180,
    )
    if not r.ok:
        # Surface the server's error body — vLLM and llama.cpp both put
        # the real reason here, and raise_for_status() hides it.
        raise RuntimeError(
            f"{r.status_code} from {url}\n"
            f"request: {payload}\n"
            f"response: {r.text[:1000]}"
        )
    return r.json()


def benchmark(name, url, model, max_tokens, rounds, samples):
    # Warm-up (not measured) so cold-start latency doesn't skew the reading.
    print(f"[{name}] warming up...")
    send(url, model, [{"role": "user", "content": "hi"}], 4)

    print(f"[{name}] measuring ({rounds} round(s) x {len(samples)} prompts)...")
    tracker = EmissionsTracker(
        project_name=name,
        measure_power_secs=1,
        save_to_file=False,
        log_level="warning",
    )
    tracker.start()
    t0 = time.perf_counter()
    total_completion = 0
    for _ in range(rounds):
        for s in samples:
            resp = send(url, model, build_messages(s), max_tokens)
            total_completion += resp.get("usage", {}).get("completion_tokens", 0)
    duration = time.perf_counter() - t0
    co2_kg = tracker.stop()
    data = tracker.final_emissions_data

    return {
        "name": name,
        "duration_s": duration,
        "tokens": total_completion,
        "energy_kwh": data.energy_consumed,
        "cpu_kwh": data.cpu_energy,
        "gpu_kwh": data.gpu_energy,
        "ram_kwh": data.ram_energy,
        "co2_kg": co2_kg,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--vllm-model")
    ap.add_argument("--llamacpp-url", default="http://localhost:9090")
    ap.add_argument("--llamacpp-model")
    ap.add_argument("--max-tokens", type=int, default=192,
                    help="Cap on output tokens. 192 leaves headroom for stacks "
                         "where the model emits a long tail of non-rendered "
                         "tokens between the visible answer and the actual "
                         "end-of-turn marker (observed: Gemma 4 multimodal on "
                         "llama.cpp emits ~120 such tokens). Models/stacks "
                         "that stop cleanly will finish well before the cap.")
    ap.add_argument("--rounds", type=int, default=1,
                    help="Repeat the prompt set N times for more stable totals.")
    ap.add_argument("--n-samples", type=int, default=100,
                    help="Number of POPE samples to stream (default 100). "
                         "Lower than the original 200 to keep wall-clock under "
                         "the 5-min/server budget when stacks emit long token "
                         "tails (Gemma 4 + llama.cpp).")
    ap.add_argument("--only", choices=["vllm", "llamacpp", "both"], default="both",
                    help="Benchmark only one server. Recommended on a single GPU "
                         "where you can't keep both servers loaded at once.")
    args = ap.parse_args()

    if args.only in ("vllm", "both") and not args.vllm_model:
        ap.error("--vllm-model is required unless --only llamacpp")
    if args.only in ("llamacpp", "both") and not args.llamacpp_model:
        ap.error("--llamacpp-model is required unless --only vllm")

    samples = load_pope_samples(args.n_samples)

    results = []
    if args.only in ("vllm", "both"):
        results.append(benchmark("vllm", args.vllm_url, args.vllm_model,
                                 args.max_tokens, args.rounds, samples))
    if args.only in ("llamacpp", "both"):
        results.append(benchmark("llamacpp", args.llamacpp_url, args.llamacpp_model,
                                 args.max_tokens, args.rounds, samples))

    print("\n=== Results ===")
    header = f"{'server':<10} {'dur(s)':>8} {'tokens':>8} {'kWh':>12} {'kg CO2':>12} {'kWh/1k tok':>12} {'gCO2/1k tok':>13}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["tokens"]:
            kwh_per_k = r["energy_kwh"] / r["tokens"] * 1000
            gco2_per_k = r["co2_kg"] * 1_000_000 / r["tokens"]  # kg -> g, then per 1k
        else:
            kwh_per_k = gco2_per_k = float("nan")
        print(
            f"{r['name']:<10} {r['duration_s']:>8.2f} {r['tokens']:>8} "
            f"{r['energy_kwh']:>12.6f} {r['co2_kg']:>12.6f} "
            f"{kwh_per_k:>12.6f} {gco2_per_k:>13.4f}"
        )

    print("\n=== Breakdown (kWh) ===")
    print(f"{'server':<10} {'cpu':>12} {'gpu':>12} {'ram':>12}")
    for r in results:
        print(f"{r['name']:<10} {r['cpu_kwh']:>12.6f} "
              f"{r['gpu_kwh']:>12.6f} {r['ram_kwh']:>12.6f}")
    if any(r["gpu_kwh"] == 0 for r in results):
        print("WARNING: GPU energy reported as 0 — codecarbon likely can't read "
              "NVML. Try `pip install nvidia-ml-py` and re-run.")

    #if all(r["tokens"] > 0 for r in results):
    #    winner = min(results, key=lambda r: r["co2_kg"] / r["tokens"])
    #    print(f"\nMost economical (lowest CO2 per output token): {winner['name']}")


if __name__ == "__main__":
    main()
