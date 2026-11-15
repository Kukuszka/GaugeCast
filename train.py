from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from gaugecast.config import load_config
from gaugecast.data.dataset import SequenceDataset
from gaugecast.data.preprocessing import prepare_training_data
from gaugecast.losses import (
    compute_total_loss,
    LAMBDA_GAUGE, LAMBDA_PDE, LAMBDA_DIFF, LAMBDA_SHAPE, LAMBDA_SMOOTH,
    LAMBDA_EVENT_REG, LAMBDA_SM_SUP, LAMBDA_SM_SMOOTH, LAMBDA_LATE_MSE,
    LAMBDA_END, LAMBDA_TAIL_RISE, LAMBDA_SG_PREFIX, LAMBDA_SG_TAIL,
    LAMBDA_PH_PREFIX, LAMBDA_PH_TAIL,
)
from gaugecast.models.gaugecast import GaugeHydroDecoderPINN
from gaugecast.utils.metrics import compute_nse, evaluate_epoch_metrics
from gaugecast.utils.seed import set_all_seeds


def train(cfg: dict) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_csv = cfg["train_csv"]
    test_csv = cfg["test_csv"]
    save_dir = cfg.get("save_dir", "results")
    experiment_name = str(cfg.get("experiment_name", "gaugecast"))

    seq_len = int(cfg.get("seq_len", 32))
    pred_len = int(cfg.get("pred_len", 32))
    batch_size = int(cfg.get("batch_size", 64))
    epochs = int(cfg.get("epochs", 20))
    lr = float(cfg.get("lr", 1e-4))
    hidden_size = int(cfg.get("hidden_size", 64))
    use_temp_vwc = bool(cfg.get("use_temp_vwc", True))
    seed_list = list(cfg.get("seeds", [42, 1, 0]))
    sg_start_epoch = int(cfg.get("sg_start_epoch", 5))
    sg_warmup_epochs = int(cfg.get("sg_warmup_epochs", 5))

    if not use_temp_vwc:
        raise ValueError(
            "The uploaded GaugeCast notebook uses vwc05 supervision. "
            "Set use_temp_vwc: true for this release."
        )

    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, f"multiseed_results_{experiment_name}.pkl")

    data = prepare_training_data(
        train_csv=train_csv,
        test_csv=test_csv,
        save_dir=save_dir,
        experiment_name=experiment_name,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=hidden_size,
        use_temp_vwc=use_temp_vwc,
    )

    print("Feature columns:", data["feature_cols"])
    print("level_idx:", data["level_idx"])
    print("rain_idx:", data["rain_idx"])
    print("future_rain_idx:", data["future_rain_idx"])
    print("vwc_idx:", data["vwc_idx"])
    print("hist_input_idx:", data["hist_input_idx"])
    print("Saved preprocessing to:", data["preproc_path"])

    dataset_kwargs = dict(
        seq_len=seq_len,
        pred_len=pred_len,
        level_idx=data["level_idx"],
        future_rain_idx=data["future_rain_idx"],
        vwc_idx=data["vwc_idx"],
    )
    train_ds = SequenceDataset(data["train_data"], **dataset_kwargs)
    val_ds = SequenceDataset(data["val_data"], **dataset_kwargs)
    test_ds = SequenceDataset(data["test_scaled"], **dataset_kwargs)

    all_seed_results = []

    for seed in seed_list:
        print("\n" + "=" * 120)
        print(f"TRAINING SEED {seed}")
        print("=" * 120)

        set_all_seeds(int(seed))
        g = torch.Generator()
        g.manual_seed(int(seed))

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, generator=g
        )
        train_loader_eval = DataLoader(
            train_ds, batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = GaugeHydroDecoderPINN(
            hist_input_dim=len(data["hist_input_idx"]),
            hist_input_idx=data["hist_input_idx"],
            rain_idx=data["rain_idx"],
            vwc_idx=data["vwc_idx"],
            hidden_dim=hidden_size,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = {
            "train_total": [],
            "train_nse": [],
            "val_mse": [],
            "val_nse": [],
            "test_mse": [],
            "test_nse": [],
        }

        for epoch in range(epochs):
            model.train()

            if epoch + 1 < sg_start_epoch:
                sg_scale = 0.0
            else:
                sg_scale = min(
                    1.0,
                    (epoch + 1 - sg_start_epoch + 1) / max(sg_warmup_epochs, 1),
                )

            sums = {
                "total_loss": 0.0, "pred_loss": 0.0, "mse_loss": 0.0,
                "late_mse": 0.0, "end_loss": 0.0, "tail_rise_loss": 0.0,
                "sm_sup_loss": 0.0, "sm_smooth_loss": 0.0,
                "shape_loss": 0.0, "smooth_loss": 0.0,
                "event_reg_loss": 0.0, "gauge_loss": 0.0,
                "pde_loss": 0.0, "diff_loss": 0.0,
                "kernel_ent_loss": 0.0, "sg_prefix": 0.0,
                "sg_tail": 0.0, "ph_prefix": 0.0, "ph_tail": 0.0,
            }
            n_batches = 0
            train_preds, train_targets = [], []

            for x_hist, fut_seq, y, vwc_future in train_loader:
                x_hist = x_hist.to(device)
                fut_seq = fut_seq.to(device)
                y = y.to(device)
                vwc_future = vwc_future.to(device)

                losses = compute_total_loss(
                    model=model,
                    x_hist=x_hist,
                    fut_seq=fut_seq,
                    y_true=y,
                    vwc_true=vwc_future,
                    sg_scale=sg_scale,
                )

                optimizer.zero_grad()
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                pred = losses["pred"].squeeze(-1).detach()
                for key in sums:
                    sums[key] += float(losses[key].item())
                n_batches += 1

                train_preds.append(pred.cpu())
                train_targets.append(y.detach().cpu())

            train_preds = torch.cat(train_preds, dim=0)
            train_targets = torch.cat(train_targets, dim=0)
            train_nse_batch = compute_nse(train_targets, train_preds).item()
            means = {k: v / max(n_batches, 1) for k, v in sums.items()}

            train_eval_mse, train_eval_nse = evaluate_epoch_metrics(
                model, train_loader_eval, device
            )
            val_mse, val_nse = evaluate_epoch_metrics(model, val_loader, device)
            test_mse, test_nse = evaluate_epoch_metrics(model, test_loader, device)

            history["train_total"].append(means["total_loss"])
            history["train_nse"].append(train_eval_nse)
            history["val_mse"].append(val_mse)
            history["val_nse"].append(val_nse)
            history["test_mse"].append(test_mse)
            history["test_nse"].append(test_nse)

            print(
                f"Seed {seed} | Epoch {epoch+1:02d}/{epochs} | "
                f"TrainTotal: {means['total_loss']:.7f} | "
                f"TrainNSE(batch): {train_nse_batch:.7f} | SGScale: {sg_scale:.3f}"
            )
            print(
                f"  PredLoss: {means['pred_loss']:.7f} | MSE: {means['mse_loss']:.7f} | "
                f"LateMSE: {means['late_mse']:.7f} | EndLoss: {means['end_loss']:.7f} | "
                f"TailRise: {means['tail_rise_loss']:.7f}"
            )
            print(
                f"  SM-Sup: {means['sm_sup_loss']:.7f} | "
                f"SM-Smooth: {means['sm_smooth_loss']:.7f} | "
                f"Shape: {means['shape_loss']:.7f} | "
                f"Smooth: {means['smooth_loss']:.7f} | "
                f"EventReg: {means['event_reg_loss']:.7f}"
            )
            print(
                f"  Gauge: {means['gauge_loss']:.7f} | PDE: {means['pde_loss']:.7f} | "
                f"Diff: {means['diff_loss']:.7f} | "
                f"KernelEnt: {means['kernel_ent_loss']:.7f}"
            )
            print(
                f"  SG-Prefix: {means['sg_prefix']:.7f} | SG-Tail: {means['sg_tail']:.7f} | "
                f"PH-Prefix: {means['ph_prefix']:.7f} | PH-Tail: {means['ph_tail']:.7f}"
            )
            print(f"[TRAIN-EVAL] MSE: {train_eval_mse:.6f} | NSE: {train_eval_nse:.6f}")
            print(f"[VAL       ] MSE: {val_mse:.6f} | NSE: {val_nse:.6f}")
            print(f"[TEST      ] MSE: {test_mse:.6f} | NSE: {test_nse:.6f}")
            print("-" * 95)

        model_path = os.path.join(
            save_dir, f"final_model_seed_{seed}_{experiment_name}.pt"
        )
        torch.save(
            {
                "seed": seed,
                "experiment_name": experiment_name,
                "epochs": epochs,
                "model_state_dict": model.state_dict(),
                "hidden_size": hidden_size,
                "seq_len": seq_len,
                "pred_len": pred_len,
                "feature_cols": data["feature_cols"],
                "level_idx": data["level_idx"],
                "rain_idx": data["rain_idx"],
                "future_rain_idx": data["future_rain_idx"],
                "vwc_idx": data["vwc_idx"],
                "hist_input_idx": data["hist_input_idx"],
                "lambda_gauge": LAMBDA_GAUGE,
                "lambda_pde": LAMBDA_PDE,
                "lambda_diff": LAMBDA_DIFF,
                "lambda_shape": LAMBDA_SHAPE,
                "lambda_smooth": LAMBDA_SMOOTH,
                "lambda_event_reg": LAMBDA_EVENT_REG,
                "lambda_sm_sup": LAMBDA_SM_SUP,
                "lambda_sm_smooth": LAMBDA_SM_SMOOTH,
                "lambda_late_mse": LAMBDA_LATE_MSE,
                "lambda_end": LAMBDA_END,
                "lambda_tail_rise": LAMBDA_TAIL_RISE,
                "lambda_sg_prefix": LAMBDA_SG_PREFIX,
                "lambda_sg_tail": LAMBDA_SG_TAIL,
                "lambda_ph_prefix": LAMBDA_PH_PREFIX,
                "lambda_ph_tail": LAMBDA_PH_TAIL,
                "history": history,
                "final_train_mse": train_eval_mse,
                "final_train_nse": train_eval_nse,
                "final_val_mse": val_mse,
                "final_val_nse": val_nse,
                "final_test_mse": test_mse,
                "final_test_nse": test_nse,
            },
            model_path,
        )
        print(f"Saved final model for seed {seed} to: {model_path}")

        all_seed_results.append(
            {
                "seed": seed,
                "model_path": model_path,
                "train_eval_mse": train_eval_mse,
                "train_eval_nse": train_eval_nse,
                "val_mse": val_mse,
                "val_nse": val_nse,
                "test_mse": test_mse,
                "test_nse": test_nse,
                "history": history,
            }
        )

    with open(results_path, "wb") as f:
        pickle.dump(all_seed_results, f)
    print(f"Saved multi-seed results to: {results_path}")

    if len(all_seed_results) > 1:
        train_mses = np.array([r["train_eval_mse"] for r in all_seed_results])
        train_nses = np.array([r["train_eval_nse"] for r in all_seed_results])
        val_mses = np.array([r["val_mse"] for r in all_seed_results])
        val_nses = np.array([r["val_nse"] for r in all_seed_results])
        test_mses = np.array([r["test_mse"] for r in all_seed_results])
        test_nses = np.array([r["test_nse"] for r in all_seed_results])
        print("\n" + "=" * 120)
        print("MULTI-SEED SUMMARY")
        print("=" * 120)
        print(f"Train MSE mean ± std: {train_mses.mean():.6f} ± {train_mses.std(ddof=1):.6f}")
        print(f"Train NSE mean ± std: {train_nses.mean():.6f} ± {train_nses.std(ddof=1):.6f}")
        print(f"Val MSE mean ± std:   {val_mses.mean():.6f} ± {val_mses.std(ddof=1):.6f}")
        print(f"Val NSE mean ± std:   {val_nses.mean():.6f} ± {val_nses.std(ddof=1):.6f}")
        print(f"Test MSE mean ± std:  {test_mses.mean():.6f} ± {test_mses.std(ddof=1):.6f}")
        print(f"Test NSE mean ± std:  {test_nses.mean():.6f} ± {test_nses.std(ddof=1):.6f}")

    return all_seed_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GaugeCast++.")
    parser.add_argument("--config", required=True, help="Path to a YAML site configuration.")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
