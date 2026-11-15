) -> Dict[str, Any]:
    """Reproduce the notebook preprocessing and chronological 80/20 train/validation split."""
    train_raw = canonicalize_columns(pd.read_csv(train_csv))
    test_raw = canonicalize_columns(pd.read_csv(test_csv))

    train_df, feature_cols = engineer_features(train_raw, use_temp_vwc=use_temp_vwc)
    test_df, _ = engineer_features(test_raw, use_temp_vwc=use_temp_vwc)

    train_size = int(0.8 * len(train_df))
    train_df_split = train_df.iloc[:train_size].copy()
    val_df_split = train_df.iloc[train_size:].copy()

    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_df_split[feature_cols])
    val_data = scaler.transform(val_df_split[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])

    level_idx = feature_cols.index("level")
    rain_idx = feature_cols.index("rain_edge_green")
    future_rain_idx = feature_cols.index("future_rain")
    vwc_idx = feature_cols.index("vwc05")
    hist_input_idx = [i for i, c in enumerate(feature_cols) if c != "future_rain"]

    os.makedirs(save_dir, exist_ok=True)
    preproc_path = os.path.join(save_dir, f"preprocessing_{experiment_name}.pkl")
    meta = {
        "scaler": scaler,
        "feature_cols": feature_cols,
        "level_idx": level_idx,
        "rain_idx": rain_idx,
        "future_rain_idx": future_rain_idx,
        "vwc_idx": vwc_idx,
        "hist_input_idx": hist_input_idx,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "hidden_size": hidden_size,
        "use_temp_vwc": use_temp_vwc,
        "experiment_name": experiment_name,
    }
    with open(preproc_path, "wb") as f:
        pickle.dump(meta, f)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "train_data": train_data,
        "val_data": val_data,
        "test_scaled": test_scaled,
        "preproc_path": preproc_path,
        **{k: meta[k] for k in (
            "level_idx", "rain_idx", "future_rain_idx", "vwc_idx",
            "hist_input_idx", "seq_len", "pred_len", "hidden_size"
        )},
    }


def load_preprocessing(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_test_data(test_csv: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare test data with the exact training feature order and scaler."""
    raw = canonicalize_columns(pd.read_csv(test_csv))

    # Evaluation notebook allowed a missing future-rain column by falling back to observed rain.
    if "future_rain" not in raw.columns and "rain_edge_green" in raw.columns:
        raw["future_rain"] = raw["rain_edge_green"]

    test_df, _ = engineer_features(raw, use_temp_vwc=meta.get("use_temp_vwc", True))
    feature_cols = meta["feature_cols"]
    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns in test data: {missing}")

    scaler = meta["scaler"]
    test_scaled = scaler.transform(test_df[feature_cols])
    times_all = pd.to_datetime(test_df["dates"], utc=True).reset_index(drop=True)
    obs_level_all = invert_level_only(
        test_scaled[:, meta["level_idx"]],
        scaler,
        feature_cols,
        meta["level_idx"],
    )
    rain_phys_all = np.clip(np.expm1(test_df["rain_edge_green"].to_numpy()), 0.0, None)

    return {
        "test_df": test_df,
        "test_scaled": test_scaled,
        "times_all": times_all,
        "obs_level_all": obs_level_all,
        "rain_phys_all": rain_phys_all,
    }
