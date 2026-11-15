from __future__ import annotations

import argparse
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gaugecast.config import load_config
from gaugecast.data.preprocessing import load_preprocessing, prepare_test_data, invert_level_only
from gaugecast.models.gaugecast import GaugeHydroDecoderPINN
from gaugecast.physics.pullback import build_grid
from gaugecast.utils.metrics import mse_np, nse_np, peak_metrics_np


def load_saved_model_for_seed(seed, model_template, meta, device):
    model_path = model_template.format(seed=seed)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found for seed {seed}: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    model = GaugeHydroDecoderPINN(
        hist_input_dim=len(meta["hist_input_idx"]),
        hist_input_idx=meta["hist_input_idx"],
        rain_idx=meta["rain_idx"],
        vwc_idx=meta["vwc_idx"],
        hidden_dim=meta["hidden_size"],
    ).to(device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


def predict_one_window(model, x_hist_arr, fut_seq_arr, scaler, feature_cols, level_idx, device):
    x_hist = torch.tensor(x_hist_arr, dtype=torch.float32, device=device).unsqueeze(0)
    fut_seq = torch.tensor(fut_seq_arr, dtype=torch.float32, device=device).view(1, -1, 1)

    with torch.no_grad():
        tau, x = build_grid(x_hist.size(0), fut_seq.size(1), device, requires_grad=False)
        out = model(x_hist, fut_seq, tau, x)
        pred_scaled = out["h_pred"].squeeze(0).squeeze(-1).detach().cpu().numpy()

    return invert_level_only(pred_scaled, scaler, feature_cols, level_idx)


def rolling_multiseed_evaluation(
    label,
    models,
    seed_list,
    meta,
    test_scaled,
    times_all,
    obs_level_all,
    rain_phys_all,
    device,
    start_time=None,
    end_time=None,
    issue_every_steps=32,
    make_plot=True,
    save_fig=False,
    save_basename="gaugecast_event_forecast",
    figsize=(25, 14),
    show_peak_markers=True,
):
    seq_len = meta["seq_len"]
    pred_len = meta["pred_len"]
    future_rain_idx = meta["future_rain_idx"]
    scaler = meta["scaler"]
    feature_cols = meta["feature_cols"]
    level_idx = meta["level_idx"]

    if start_time is not None:
        start_time = pd.Timestamp(start_time)
        start_time = start_time.tz_localize("UTC") if start_time.tzinfo is None else start_time.tz_convert("UTC")
    if end_time is not None:
        end_time = pd.Timestamp(end_time)
        end_time = end_time.tz_localize("UTC") if end_time.tzinfo is None else end_time.tz_convert("UTC")

    all_issue_idx = np.arange(seq_len, len(test_scaled) - pred_len + 1)
    if start_time is not None and end_time is not None:
        issue_mask = (
            (times_all.iloc[all_issue_idx] >= start_time)
            & (times_all.iloc[all_issue_idx] <= end_time)
        )
        all_issue_idx = all_issue_idx[np.asarray(issue_mask)]

    issue_idx_list = list(all_issue_idx[::issue_every_steps])
    if not issue_idx_list:
        raise ValueError(f"No valid issue indices for {label}.")

    print("\n" + "=" * 100)
    print(f"ROLLING EVALUATION: {label}")
    print("=" * 100)
    print("Number of issue times:", len(issue_idx_list))

    all_point_records = []
    issue_lines = []

    for issue_idx in issue_idx_list:
        issue_time = times_all.iloc[issue_idx]
        x_hist_arr = test_scaled[issue_idx - seq_len: issue_idx]
        fut_seq_arr = test_scaled[issue_idx: issue_idx + pred_len, future_rain_idx]
        target_times = pd.to_datetime(
            times_all.iloc[issue_idx: issue_idx + pred_len]
        ).to_numpy()

        seed_preds_this_issue = []
        for seed in seed_list:
            pred_phys = predict_one_window(
                model=models[seed],
                x_hist_arr=x_hist_arr,
                fut_seq_arr=fut_seq_arr,
                scaler=scaler,
                feature_cols=feature_cols,
                level_idx=level_idx,
                device=device,
            )
            seed_preds_this_issue.append(pred_phys)

            for tt, yy in zip(target_times, pred_phys):
                tt = pd.Timestamp(tt)
                keep = (
                    start_time <= tt <= end_time
                    if start_time is not None and end_time is not None
                    else True
                )
                if keep:
                    all_point_records.append(
                        {
                            "issue_time": issue_time,
                            "target_time": tt,
                            "seed": seed,
                            "pred": float(yy),
                        }
                    )

        seed_preds_this_issue = np.asarray(seed_preds_this_issue)
        issue_mean = seed_preds_this_issue.mean(axis=0)
        mask_inside = (
            np.array([(start_time <= pd.Timestamp(tt) <= end_time) for tt in target_times])
            if start_time is not None and end_time is not None
            else np.ones(len(target_times), dtype=bool)
        )
        if mask_inside.any():
            issue_lines.append(
                {
                    "issue_time": issue_time,
                    "target_times": pd.to_datetime(target_times[mask_inside]),
                    "mean_pred": issue_mean[mask_inside],
                }
            )

    pred_df = pd.DataFrame(all_point_records)
    if pred_df.empty:
        raise ValueError(f"No forecast points were generated for {label}.")

    seed_target_df = (
        pred_df.groupby(["target_time", "seed"])["pred"].mean().reset_index()
    )
    agg_df = (
        seed_target_df.groupby("target_time")["pred"]
        .agg(pred_mean="mean", pred_std=lambda x: x.std(ddof=1), n_models="count")
        .reset_index()
        .sort_values("target_time")
    )
    agg_df["pred_std"] = agg_df["pred_std"].fillna(0.0)

    obs_df = pd.DataFrame(
        {"target_time": times_all, "obs_level": obs_level_all, "rain": rain_phys_all}
    )
    if start_time is not None and end_time is not None:
        obs_df = obs_df[
            (obs_df["target_time"] >= start_time)
            & (obs_df["target_time"] <= end_time)
        ]

    plot_df = (
        agg_df.merge(obs_df, on="target_time", how="left")
        .dropna()
        .sort_values("target_time")
        .reset_index(drop=True)
    )

    consensus_nse = nse_np(plot_df["obs_level"].values, plot_df["pred_mean"].values)
    consensus_mse = mse_np(plot_df["obs_level"].values, plot_df["pred_mean"].values)
    consensus_peak_delay, consensus_peak_height, i_true, i_pred = peak_metrics_np(
        plot_df["target_time"].values,
        plot_df["obs_level"].values,
        plot_df["pred_mean"].values,
    )

    seed_metric_records = []
    for seed in seed_list:
        seed_df = seed_target_df[seed_target_df["seed"] == seed].copy()
        seed_plot_df = (
            seed_df.merge(obs_df[["target_time", "obs_level"]], on="target_time", how="left")
            .dropna()
            .sort_values("target_time")
            .reset_index(drop=True)
        )
        if len(seed_plot_df) == 0:
            continue
        seed_nse = nse_np(seed_plot_df["obs_level"].values, seed_plot_df["pred"].values)
        seed_mse = mse_np(seed_plot_df["obs_level"].values, seed_plot_df["pred"].values)
        seed_peak_delay, seed_peak_height, _, _ = peak_metrics_np(
            seed_plot_df["target_time"].values,
            seed_plot_df["obs_level"].values,
            seed_plot_df["pred"].values,
        )
        seed_metric_records.append(
            {
                "seed": seed,
                "MSE": seed_mse,
                "NSE": seed_nse,
                "peak_delay_h": seed_peak_delay,
                "peak_height_diff": seed_peak_height,
            }
        )

    seed_metrics_df = pd.DataFrame(seed_metric_records)
    metric_summary = seed_metrics_df[
        ["MSE", "NSE", "peak_delay_h", "peak_height_diff"]
    ].agg(["mean", "std"])

    result = {
        "label": label,
        "n_issue_times": len(issue_idx_list),
        "consensus_mse": consensus_mse,
        "consensus_nse": consensus_nse,
        "consensus_peak_delay_h": consensus_peak_delay,
        "consensus_peak_height_diff": consensus_peak_height,
        "mse_mean": metric_summary.loc["mean", "MSE"],
        "mse_std": metric_summary.loc["std", "MSE"],
        "nse_mean": metric_summary.loc["mean", "NSE"],
        "nse_std": metric_summary.loc["std", "NSE"],
        "peak_delay_mean": metric_summary.loc["mean", "peak_delay_h"],
        "peak_delay_std": metric_summary.loc["std", "peak_delay_h"],
        "peak_height_mean": metric_summary.loc["mean", "peak_height_diff"],
        "peak_height_std": metric_summary.loc["std", "peak_height_diff"],
    }

    print("\nConsensus metrics from ensemble mean:")
    print(f"MSE: {consensus_mse:.6f}")
    print(f"NSE: {consensus_nse:.6f}")
    print(f"Peak delay: {consensus_peak_delay:.3f} h")
    print(f"Peak height error: {consensus_peak_height:.6f}")
    print("\nPer-seed metrics:")
    print(seed_metrics_df)
    print("\nMean ± std across seeds:")
    print(metric_summary)

    if make_plot:
        fig, ax = plt.subplots(figsize=figsize)
        ax2 = ax.twinx()

        event_times = pd.to_datetime(plot_df["target_time"])
        event_rain = plot_df["rain"].values
        if len(event_times) > 1:
            dt_seconds = np.median(
                np.diff(event_times.values).astype("timedelta64[s]").astype(np.int64)
            )
            bar_w_days = (dt_seconds / 86400.0) * 0.85
        else:
            bar_w_days = 0.02

        ax2.bar(event_times, event_rain, width=bar_w_days, alpha=0.25, label="Rainfall")
        ax2.set_yticks([])
        ax2.set_ylabel("")

        for item in issue_lines:
            ax.plot(
                item["target_times"], item["mean_pred"],
                alpha=0.22, linewidth=0.9, zorder=1
            )

        x_band = pd.to_datetime(plot_df["target_time"]).to_numpy()
        y_mean = plot_df["pred_mean"].to_numpy(dtype=float)
        y_std = plot_df["pred_std"].to_numpy(dtype=float)
        ax.fill_between(
            x_band, y_mean - y_std, y_mean + y_std,
            alpha=0.18, label="Model spread ±1σ", zorder=2
        )
        ax.plot(
            plot_df["target_time"], plot_df["pred_mean"],
            linestyle="--", linewidth=1.8, label="Consensus mean", zorder=3
        )
        ax.plot(
            plot_df["target_time"], plot_df["obs_level"],
            linewidth=2.0, label="Observed", zorder=4
        )

        if show_peak_markers and len(plot_df) > 0:
            ax.plot(
                plot_df["target_time"].iloc[i_true],
                plot_df["obs_level"].iloc[i_true],
                "o", ms=5, label="True peak", zorder=5
            )
            ax.plot(
                plot_df["target_time"].iloc[i_pred],
                plot_df["pred_mean"].iloc[i_pred],
                "o", ms=5, label="Pred peak", zorder=5
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Water level (m)")
        ax.grid(alpha=0.28)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
        ax.set_title(f"{label}: rolling multi-seed forecasts")

        metrics_text = (
            f"NSE = {result['nse_mean']:.3f} ± {result['nse_std']:.3f}\n"
            f"MSE = {result['mse_mean']:.4f} ± {result['mse_std']:.4f}\n"
            f"Δt_peak = {result['peak_delay_mean']:.2f} ± {result['peak_delay_std']:.2f} h\n"
            f"Δh_peak = {result['peak_height_mean']:.3f} ± {result['peak_height_std']:.3f} m"
        )
        ax.text(
            0.015, 0.97, metrics_text, transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
        )
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False)
        fig.tight_layout()

        if save_fig:
            fig.savefig(f"{save_basename}.pdf", bbox_inches="tight")
            fig.savefig(f"{save_basename}.png", dpi=300, bbox_inches="tight")
        plt.show()

    return result, seed_metrics_df, plot_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GaugeCast++ checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--event-start", default=None)
    parser.add_argument("--event-end", default=None)
    parser.add_argument("--whole-test", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    save_dir = cfg.get("save_dir", "results")
    experiment_name = str(cfg.get("experiment_name", "gaugecast"))
    test_csv = cfg["test_csv"]
    seed_list = list(cfg.get("seeds", [42, 1, 0]))
    issue_every_steps = int(cfg.get("issue_every_steps_test", 32))

    preproc_path = os.path.join(save_dir, f"preprocessing_{experiment_name}.pkl")
    model_template = os.path.join(
        save_dir, f"final_model_seed_{{seed}}_{experiment_name}.pt"
    )
    meta = load_preprocessing(preproc_path)
    prepared = prepare_test_data(test_csv, meta)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        seed: load_saved_model_for_seed(seed, model_template, meta, device)
        for seed in seed_list
    }

    start = None if args.whole_test else (args.event_start or cfg.get("event_start"))
    end = None if args.whole_test else (args.event_end or cfg.get("event_end"))

    rolling_multiseed_evaluation(
        label=f"{experiment_name} | {'Whole test' if args.whole_test else 'Event'}",
        models=models,
        seed_list=seed_list,
        meta=meta,
        test_scaled=prepared["test_scaled"],
        times_all=prepared["times_all"],
        obs_level_all=prepared["obs_level_all"],
        rain_phys_all=prepared["rain_phys_all"],
        device=device,
        start_time=start,
        end_time=end,
        issue_every_steps=issue_every_steps,
        make_plot=not args.no_plot,
        save_fig=bool(cfg.get("save_fig", False)),
        save_basename=os.path.join(save_dir, f"{experiment_name}_event_forecast"),
    )


if __name__ == "__main__":
    main()
