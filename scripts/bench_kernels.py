#!/usr/bin/env python3
"""
Run every kernel + backend combination in configs/bench_shapes.yaml and collect
timings into benchmarks/results.{md,json,png}.

Each binary is invoked with `--bench` and sweep-specific flags (--n, --p,
--iters, --warmup). Binaries print exactly one JSON object per run on stdout,
which we parse and aggregate.

Example:

    python scripts/bench_kernels.py                # default config
    python scripts/bench_kernels.py --config configs/bench_shapes.yaml \
        --out-dir benchmarks
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REPO = HERE.parent


def _run_binary(binary: Path, sweep: dict) -> dict:
    cmd = [str(binary), "--bench"]
    for key in ("n", "p", "iters", "warmup"):
        if key in sweep:
            cmd += [f"--{key}", str(sweep[key])]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        return {"error": f"binary missing: {binary} (did you run `make`?)"}
    except subprocess.CalledProcessError as e:
        return {"error": f"non-zero exit ({e.returncode}): {e.stderr.strip()}"}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}

    # binary prints exactly one JSON line; tolerate trailing whitespace
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"error": f"non-JSON stdout: {proc.stdout!r}"}


def _run_config(config: dict) -> list[dict]:
    rows: list[dict] = []
    for kernel_name, spec in config.items():
        binaries = spec["binaries"]
        for backend in spec["backends"]:
            binary = REPO / binaries[backend]
            for sweep in spec["sweeps"]:
                print(f"[{kernel_name}:{backend}] sweep {sweep}")
                rec = _run_binary(binary, sweep)
                rec.setdefault("kernel", kernel_name)
                rec.setdefault("backend", backend)
                for k, v in sweep.items():
                    rec.setdefault(k, v)
                rows.append(rec)
                if "error" in rec:
                    print(f"  ! {rec['error']}")
                else:
                    ms = rec.get("ms_mean")
                    extra = rec.get("gflops") or rec.get("gbps")
                    unit = "GFLOP/s" if "gflops" in rec else "GB/s"
                    print(f"  {ms:.3f} ms  ({extra:.2f} {unit})")
    return rows


def _write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2))


def _fmt_num(rec: dict, key: str, digits: int = 3) -> str:
    if key not in rec or rec[key] is None:
        return "-"
    return f"{rec[key]:.{digits}f}"


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Kernel benchmark results",
        "",
        "Times are kernel-only (no H2D/D2H). Each row is mean +/- std over "
        "`iters` launches after `warmup` untimed launches.",
        "",
        "| Kernel | Backend | Shape | iters | ms mean | ms std | Throughput |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "error" in r:
            shape = _shape_str(r)
            lines.append(
                f"| `{r.get('kernel','?')}` | `{r.get('backend','?')}` | {shape} "
                f"| {r.get('iters','-')} | ERROR | ERROR | `{r['error']}` |"
            )
            continue
        thru = ""
        if "gflops" in r:
            thru = f"{r['gflops']:.2f} GFLOP/s"
        elif "gbps" in r:
            thru = f"{r['gbps']:.2f} GB/s"
        shape = _shape_str(r)
        lines.append(
            f"| `{r['kernel']}` | `{r['backend']}` | {shape} | {r['iters']} "
            f"| {_fmt_num(r, 'ms_mean')} | {_fmt_num(r, 'ms_std')} | {thru} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def _shape_str(r: dict) -> str:
    parts = [f"n={r['n']}"] if "n" in r else []
    if "p" in r:
        parts.append(f"p={r['p']}")
    return ", ".join(parts) or "-"


def _write_plot(path: Path, rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib unavailable, skipping {path}")
        return

    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("no successful runs, skipping plot")
        return

    kernels = sorted({r["kernel"] for r in ok})
    fig, axes = plt.subplots(len(kernels), 1, figsize=(9, 2.8 * len(kernels)), squeeze=False)
    for ax, kernel in zip(axes[:, 0], kernels, strict=True):
        kernel_rows = [r for r in ok if r["kernel"] == kernel]
        backends = sorted({r["backend"] for r in kernel_rows})
        shapes = sorted({r["n"] for r in kernel_rows})
        width = 0.8 / max(len(backends), 1)
        for i, backend in enumerate(backends):
            xs = []
            ys = []
            errs = []
            for j, n in enumerate(shapes):
                rec = next((r for r in kernel_rows if r["backend"] == backend and r["n"] == n), None)
                if rec is None:
                    continue
                xs.append(j + i * width)
                ys.append(rec["ms_mean"])
                errs.append(rec.get("ms_std", 0.0))
            ax.bar(xs, ys, width=width, yerr=errs, capsize=3, label=backend)
        ax.set_yscale("log")
        ax.set_xticks([j + (len(backends) - 1) * width / 2 for j in range(len(shapes))])
        ax.set_xticklabels([f"n={n}" for n in shapes])
        ax.set_ylabel("ms (log)")
        ax.set_title(kernel)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "bench_shapes.yaml")
    parser.add_argument("--out-dir", type=Path, default=REPO / "benchmarks")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = _run_config(config)

    _write_json(args.out_dir / "results.json", rows)
    _write_markdown(args.out_dir / "results.md", rows)
    _write_plot(args.out_dir / "results.png", rows)
    print(f"\nWrote results to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
