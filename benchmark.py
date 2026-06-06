"""Efficiency benchmark for AnisotropicCRNN: parameters, FLOPs, and latency.

These are the numbers backing the "time-efficient / low-latency" claim. For a
sequence model, FLOPs and latency scale with the number of frames T, so we report
per-frame figures (the streaming-relevant cost) alongside totals at a reference
length, and the real-time factor (RTF = compute time / audio duration).

Usage:
    python benchmark.py                                  # default LayerNorm model
    python benchmark.py --config configs/crnn_ravdess_ln.yaml
    python benchmark.py --lengths 100 250 500 --trials 100
"""
import argparse
import time

import torch
import yaml

from train import build_model

HOP_MS = 10.0  # mel hop; 1 frame = 10 ms of audio


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    by_module = {}
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        if n:
            by_module[name] = n
    return total, by_module


def measure_flops(model, T, device, n_mels=128):
    """Forward FLOPs at length T (FLOPs = 2*MACs).

    `counter_total` (torch FlopCounterMode) is the authoritative, architecture-agnostic
    number: it counts whatever ops actually run (conv, matmul, attention, ...). The
    analytic per-component breakdown (conv via hooks; recurrent/head via closed forms)
    is for insight and is best-effort — if the model has no nn.LSTM / `classifier`
    (e.g. a Transformer/SSM backbone), those rows fall back to 0 and you should read
    `counter_total`. The two agree exactly for the current CRNN.
    """
    conv_macs_list = []

    def hook(m, inp, out):
        oc, of, ot = out.shape[1], out.shape[2], out.shape[3]
        ic = inp[0].shape[1]
        kh, kw = m.kernel_size
        conv_macs_list.append(oc * of * ot * ic * kh * kw)

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    x = torch.randn(1, 1, n_mels, T, device=device)
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    conv_macs = sum(conv_macs_list)

    rec_macs = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.LSTM, torch.nn.GRU)):
            g = 4 if isinstance(m, torch.nn.LSTM) else 3  # gate count
            in_sz, h, layers = m.input_size, m.hidden_size, m.num_layers
            mult = 2 if m.bidirectional else 1
            rec_macs += sum(g * ((in_sz if l == 0 else h * mult) * h + h * h) for l in range(layers)) * T * mult

    head = getattr(model, "classifier", None)
    head_macs = head.in_features * head.out_features * T if isinstance(head, torch.nn.Linear) else 0

    breakdown = {"conv": 2 * conv_macs, "rec": 2 * rec_macs, "head": 2 * head_macs}
    breakdown["analytic_total"] = sum(breakdown.values())

    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    breakdown["counter_total"] = fc.get_total_flops()
    return breakdown


@torch.no_grad()
def measure_latency(model, T, device, trials, warmup, n_mels=128):
    x = torch.randn(1, 1, n_mels, T, device=device)
    cuda = device.type == "cuda"
    for _ in range(warmup):
        model(x)
    if cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        model(x)
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
    times.sort()
    n = len(times)
    mean = sum(times) / n
    std = (sum((t - mean) ** 2 for t in times) / n) ** 0.5
    return {"mean": mean, "std": std, "median": times[n // 2], "p95": times[int(0.95 * n)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config (else default LayerNorm model)")
    ap.add_argument("--lengths", type=int, nargs="+", default=[100, 250, 500])
    ap.add_argument("--ref_length", type=int, default=250, help="reference length for latency/RTF")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"num_classes": 4, "n_mels": 128, "sample_rate": 16000, "win_length_ms": 25,
               "model": {"conv_channels": [1, 8, 16, 32, 64], "kernel_freq": 32,
                         "lstm_hidden": 128, "lstm_layers": 1, "dropout": 0.3, "norm": "layer"}}

    device = torch.device(args.device)
    model = build_model(cfg).to(device).eval()

    total, by_mod = count_params(model)
    print(f"\n=== Parameters ===")
    print(f"total: {total:,}")
    for name, n in by_mod.items():
        print(f"  {name:14s} {n:>10,}  ({100*n/total:4.1f}%)")

    n_mels = cfg.get("n_mels", 128)
    print(f"\n=== FLOPs (1 forward pass; FLOPs = 2*MACs) ===")
    print(f"{'T':>6} {'audio':>7} {'conv':>14} {'rec':>13} {'head':>9} {'total':>14} {'per-frame':>12} {'(counter)':>14}")
    for T in args.lengths:
        b = measure_flops(model, T, device, n_mels)
        print(f"{T:>6} {T*HOP_MS/1000:>5.2f}s {b['conv']:>14,} {b['rec']:>13,} {b['head']:>9,} "
              f"{b['analytic_total']:>14,} {b['analytic_total']//T:>12,} {b['counter_total']:>14,}")

    print(f"\n=== Latency (batch=1, {args.device}, {args.trials} trials, {args.warmup} warmup) ===")
    T = args.ref_length
    lat = measure_latency(model, T, device, args.trials, args.warmup, n_mels)
    audio_s = T * HOP_MS / 1000
    print(f"reference length T={T} ({audio_s:.2f}s of audio)")
    print(f"  per utterance: {lat['mean']:.3f} +/- {lat['std']:.3f} ms "
          f"(median {lat['median']:.3f}, p95 {lat['p95']:.3f})")
    print(f"  per frame:     {lat['mean']/T:.4f} ms")
    print(f"  real-time factor (RTF): {(lat['mean']/1000)/audio_s:.4f}  (<1 = faster than real-time)")


if __name__ == "__main__":
    main()
