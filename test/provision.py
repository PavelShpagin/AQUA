#!/usr/bin/env python3
"""
Provisioned Throughput – Cost/Throughput Feasibility Explorer

Purpose
  Offline what‑if analysis comparing Provisioned Throughput (PTU/dedicated capacity)
  against pay‑as‑you‑go (standard) or Batch API pricing using observed token usage.

Inputs (CLI)
  --model                 Model name (for pay‑go pricing lookup)
  --results_csv           CSV with columns: input_tokens, output_tokens, cached_tokens (optional)
  --total_tokens          If no CSV, you can pass total tokens directly
  --ptu_cost_per_100      Monthly $ cost per 100 PTUs (default: 31200)
  --ptu_tps_per_100       Max tokens/sec per 100 PTUs (default: 18multigec_dev_de000)
  --ptu_rps_per_100       Max requests/sec per 100 PTUs (default: 80)
  --target_qps            Expected avg requests/sec (optional)
  --avg_tokens_per_req    Expected average tokens/request (used with target_qps)
  --batch_discount        Optional 0.50 to model 50% Batch discount for pay‑go (default: 1.0 = none)

Outputs
  Prints a compact report:
    - Total tokens analyzed
    - Pay‑go cost (with optional Batch discount)
    - PTU cost at a chosen PTU level and utilization
    - Break‑even utilization and suggested PTUs
    - Throughput feasibility (rps/tps headroom)

Notes
  - This does NOT provision anything. It’s an estimator using your observed data.
  - Pricing for pay‑go is sourced from utils/pricing.py (OpenAI/official tables).
"""

from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from utils.pricing import calculate_cost, get_model_pricing


@dataclass
class PTUConfig:
    cost_per_100_ptu_per_month: float = 31200.0
    tokens_per_sec_per_100_ptu: float = 18000.0
    requests_per_sec_per_100_ptu: float = 80.0


def read_tokens(results_csv: Optional[str], total_tokens: Optional[int]) -> Tuple[int, float]:
    """Return (total_tokens, avg_tokens_per_request).
    results_csv may contain columns input_tokens, output_tokens, cached_tokens.
    If missing, returns (total_tokens, 0.0) and relies on CLI avg.
    """
    if results_csv and os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        cols = [c for c in df.columns]
        # Robust column names
        inp = df.get('input_tokens') if 'input_tokens' in cols else pd.Series([0]*len(df))
        out = df.get('output_tokens') if 'output_tokens' in cols else pd.Series([0]*len(df))
        total = (inp.fillna(0).astype(int) + out.fillna(0).astype(int)).sum()
        avg = (total / max(1, len(df)))
        return int(total), float(avg)
    if total_tokens is not None:
        return int(total_tokens), 0.0
    return 0, 0.0


def paygo_cost_usd(model: str, total_tokens: int, batch_discount: float = 1.0) -> float:
    # Split input vs output approx: 90/10 if unknown
    if total_tokens <= 0:
        return 0.0
    inp = int(total_tokens * 0.9)
    out = total_tokens - inp
    cost = calculate_cost(model, input_tokens=inp, output_tokens=out).total_cost
    return float(cost) * float(batch_discount)


def ptu_capacity(ptu: float, cfg: PTUConfig) -> Tuple[float, float]:
    """Return (tokens/sec, requests/sec) for a given PTU amount."""
    scale = ptu / 100.0
    return cfg.tokens_per_sec_per_100_ptu * scale, cfg.requests_per_sec_per_100_ptu * scale


def ptu_monthly_cost(ptu: float, cfg: PTUConfig) -> float:
    return (ptu / 100.0) * cfg.cost_per_100_ptu_per_month


def estimate_required_ptu(target_tps: float, cfg: PTUConfig) -> float:
    if target_tps <= 0:
        return 0.0
    return 100.0 * (target_tps / cfg.tokens_per_sec_per_100_ptu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--results_csv')
    ap.add_argument('--total_tokens', type=int)
    ap.add_argument('--ptu_cost_per_100', type=float, default=31200.0)
    ap.add_argument('--ptu_tps_per_100', type=float, default=18000.0)
    ap.add_argument('--ptu_rps_per_100', type=float, default=80.0)
    ap.add_argument('--target_qps', type=float, default=0.0)
    ap.add_argument('--avg_tokens_per_req', type=float, default=0.0)
    ap.add_argument('--batch_discount', type=float, default=1.0, help='0.5 to model 50% Batch discount')
    args = ap.parse_args()

    pricing = get_model_pricing(args.model)
    if not pricing:
        print(f"Unknown model pricing: {args.model}")
        return

    total_tokens, avg_tokens_csv = read_tokens(args.results_csv, args.total_tokens)
    avg_tokens = args.avg_tokens_per_req or avg_tokens_csv

    cfg = PTUConfig(
        cost_per_100_ptu_per_month=args.ptu_cost_per_100,
        tokens_per_sec_per_100_ptu=args.ptu_tps_per_100,
        requests_per_sec_per_100_ptu=args.ptu_rps_per_100,
    )

    # Pay‑go cost (optionally reflect Batch 50%)
    paygo = paygo_cost_usd(args.model, total_tokens, batch_discount=args.batch_discount)

    # If we have traffic assumptions, estimate needed PTU
    est_ptu = 0.0
    ptu_cost = 0.0
    if args.target_qps > 0 and avg_tokens > 0:
        target_tps = args.target_qps * avg_tokens
        est_ptu = estimate_required_ptu(target_tps, cfg)
        # Round up to nearest 100 PTUs block
        est_ptu = math.ceil(est_ptu / 100.0) * 100.0
        ptu_cost = ptu_monthly_cost(est_ptu, cfg)

    print("=" * 60)
    print("Provisioned Throughput – Feasibility Report")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Observed total tokens: {total_tokens:,}" if total_tokens else "Observed total tokens: n/a")
    if avg_tokens:
        print(f"Avg tokens/request: {avg_tokens:.1f}")
    if args.target_qps:
        print(f"Target QPS: {args.target_qps:.2f}")
    print()
    print(f"Pay‑go cost (batch_discount={args.batch_discount:.2f}): ${paygo:.4f}")

    if est_ptu > 0:
        tps, rps = ptu_capacity(est_ptu, cfg)
        print(f"Suggested PTU: {est_ptu:.0f} (≈{tps:.0f} tps, {rps:.0f} rps)")
        print(f"Provisioned monthly cost: ${ptu_cost:,.2f}")
        # Break‑even tokens per month (pay‑go == PTU)
        # Using pay‑go cost per token from model table (input/output mix is workload‑dependent).
        # Here we approximate with input price for simplicity.
        paygo_input_per_million = pricing['input']
        if paygo_input_per_million > 0:
            break_even_mtokens = (ptu_cost / paygo_input_per_million)
            print(f"Break‑even ~{break_even_mtokens:.2f} million input‑equivalent tokens/month")
    else:
        print("No traffic assumptions given (target_qps/avg_tokens_per_req). Supply these to size PTU.")

    print("\nTips")
    print("- Keep prompts compact; smaller avg tokens/request increases effective rps.")
    print("- Aim for high utilization. Under‑utilized PTU wastes dollars.")
    print("- Run a load test to validate rps/tps and observe real queue latency.")
    print("- Mix caching and PTU where applicable for further savings.")


if __name__ == "__main__":
    main()




