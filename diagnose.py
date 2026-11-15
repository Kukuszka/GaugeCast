from __future__ import annotations

import argparse
import os

import torch

from gaugecast.config import load_config
from gaugecast.data.preprocessing import load_preprocessing, prepare_test_data
from gaugecast.models.gaugecast import GaugeHydroDecoderPINN
from gaugecast.utils.diagnostics import (
    collect_geometry_diagnostics,
    diagnose_geometry_one_window,
    get_issue_indices,
    print_geometry_summary,
)
from gaugecast.utils.plotting import (
    plot_geometry_heatmaps,
    plot_rain_vs_geometry,
    plot_single_window_geometry,
)


def load_model(path, meta, device):
    ckpt = torch.load(path, map_location=device)
    model = GaugeHydroDecoderPINN(
        hist_input_dim=len(meta["hist_input_idx"]),
        hist_input_idx=meta["hist_input_idx"],
        rain_idx=meta["rain_idx"],
        vwc_idx=meta["vwc_idx"],
        hidden_dim=meta["hidden_size"],
    ).to(device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="GaugeCast++ pullback-geometry diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--event-start", default=None)
    parser.add_argument("--event-end", default=None)
    parser.add_argument("--issue-every-steps", type=int, default=16)
    parser.add_argument("--max-windows", type=int, default=120)
    args = parser.parse_args()

    cfg = load_config(args.config)
    save_dir = cfg.get("save_dir", "results")
    experiment_name = str(cfg.get("experiment_name", "gaugecast"))
    meta_path = os.path.join(save_dir, f"preprocessing_{experiment_name}.pkl")
    model_path = os.path.join(save_dir, f"final_model_seed_{args.seed}_{experiment_name}.pt")

    meta = load_preprocessing(meta_path)
    prepared = prepare_test_data(cfg["test_csv"], meta)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, meta, device)

    issue_idx_list = get_issue_indices(
        times_all=prepared["times_all"],
        test_scaled=prepared["test_scaled"],
        seq_len=meta["seq_len"],
        pred_len=meta["pred_len"],
        event_start=args.event_start,
        event_end=args.event_end,
        issue_every_steps=args.issue_every_steps,
        max_windows=args.max_windows,
    )
    G = collect_geometry_diagnostics(
        model=model,
        issue_idx_list=issue_idx_list,
        test_scaled=prepared["test_scaled"],
        times_all=prepared["times_all"],
        seq_len=meta["seq_len"],
        pred_len=meta["pred_len"],
        future_rain_idx=meta["future_rain_idx"],
        device=device,
    )
    print_geometry_summary(G, label=f"| seed {args.seed}")

    mid_issue = issue_idx_list[len(issue_idx_list) // 2]
    d_mid = diagnose_geometry_one_window(
        model=model,
        issue_idx=mid_issue,
        test_scaled=prepared["test_scaled"],
        times_all=prepared["times_all"],
        seq_len=meta["seq_len"],
        pred_len=meta["pred_len"],
        future_rain_idx=meta["future_rain_idx"],
        device=device,
    )
    plot_single_window_geometry(d_mid, title_prefix=f"Seed {args.seed}")
    plot_geometry_heatmaps(G, label=f"| seed {args.seed}")
    plot_rain_vs_geometry(G, label=f"| seed {args.seed}")


if __name__ == "__main__":
    main()
