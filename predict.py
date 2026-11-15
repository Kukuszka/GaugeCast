from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from gaugecast.config import load_config
from gaugecast.data.preprocessing import load_preprocessing, prepare_test_data, invert_level_only
from gaugecast.models.gaugecast import GaugeHydroDecoderPINN
from gaugecast.physics.pullback import build_grid


def load_model(checkpoint_path: str, meta: dict, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = GaugeHydroDecoderPINN(
        hist_input_dim=len(meta["hist_input_idx"]),
        hist_input_idx=meta["hist_input_idx"],
        rain_idx=meta["rain_idx"],
        vwc_idx=meta["vwc_idx"],
        hidden_dim=meta["hidden_size"],
    ).to(device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_window(model, test_scaled, issue_idx, meta, device):
    seq_len = meta["seq_len"]
    pred_len = meta["pred_len"]
    future_rain_idx = meta["future_rain_idx"]

    if issue_idx < seq_len or issue_idx + pred_len > len(test_scaled):
        raise IndexError("issue_idx does not have enough history/future samples.")

    x_hist_arr = test_scaled[issue_idx - seq_len: issue_idx]
    fut_seq_arr = test_scaled[issue_idx: issue_idx + pred_len, future_rain_idx]

    x_hist = torch.tensor(x_hist_arr, dtype=torch.float32, device=device).unsqueeze(0)
    fut_seq = torch.tensor(fut_seq_arr, dtype=torch.float32, device=device).view(1, pred_len, 1)

    with torch.no_grad():
        tau, x = build_grid(1, pred_len, device, requires_grad=False)
        out = model(x_hist, fut_seq, tau, x)
        pred_scaled = out["h_pred"].squeeze(0).squeeze(-1).cpu().numpy()

    return invert_level_only(
        pred_scaled, meta["scaler"], meta["feature_cols"], meta["level_idx"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single GaugeCast++ forecast window.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--issue-index", type=int, default=None)
    parser.add_argument("--issue-time", default=None, help="Timestamp matching the prepared test series.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    save_dir = cfg.get("save_dir", "results")
    experiment_name = str(cfg.get("experiment_name", "gaugecast"))

    preproc_path = os.path.join(save_dir, f"preprocessing_{experiment_name}.pkl")
    checkpoint_path = os.path.join(
        save_dir, f"final_model_seed_{args.seed}_{experiment_name}.pt"
    )
    meta = load_preprocessing(preproc_path)
    prepared = prepare_test_data(cfg["test_csv"], meta)

    if args.issue_index is not None:
        issue_idx = args.issue_index
    elif args.issue_time is not None:
        target = pd.Timestamp(args.issue_time)
        if target.tzinfo is None:
            target = target.tz_localize("UTC")
        else:
            target = target.tz_convert("UTC")
        matches = np.flatnonzero(prepared["times_all"].to_numpy() == target.to_datetime64())
        if len(matches) == 0:
            raise ValueError(f"Issue time not found: {target}")
        issue_idx = int(matches[0])
    else:
        issue_idx = meta["seq_len"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, meta, device)
    pred = predict_window(model, prepared["test_scaled"], issue_idx, meta, device)

    target_times = prepared["times_all"].iloc[
        issue_idx: issue_idx + meta["pred_len"]
    ].reset_index(drop=True)
    out_df = pd.DataFrame({"time": target_times, "predicted_level": pred})
    print(out_df)

    if args.output:
        out_df.to_csv(args.output, index=False)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
