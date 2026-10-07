from pathlib import Path
import tempfile
import streamlit as st
st.set_page_config(page_title="EEG Seizure Prediction", layout="wide")
st.title("Patient-independent EEG seizure prediction")
st.caption("Research prototype: CHB-MIT • multi-domain EEG • LOSO evaluation")

# Keep model libraries out of the initial page load so the interface can render first.
try:
    with st.spinner("Loading interface dependencies..."):
        import pandas as pd
        from seizure_prediction.config import load_config
except Exception as exc:
    st.error(f"Could not load the app dependencies: {exc}")
    st.stop()
st.info("Primary output: preictal vs non-preictal prediction for a configured horizon. "
        "Supporting output: interictal / preictal / ictal state classification. "
        "A prediction score is not a clinical warning system.")
st.subheader("Proposal-reported preliminary CNN-LSTM history")
preliminary_path = Path(__file__).parent / "docs" / "proposal_preliminary_cnn_lstm.csv"
preliminary = pd.read_csv(preliminary_path)
st.caption("Transcribed from proposal page 16 for context only. The final training/validation gap is about 29.1 percentage points. These values were not reproduced by this project and are not LOSO results.")
st.line_chart(preliminary.set_index("epoch")[["training_accuracy_percent", "validation_accuracy_percent"]])
project_root = Path(__file__).resolve().parent
config_path = Path(st.sidebar.text_input("Configuration file", "config.yaml"))
if not config_path.is_absolute():
    config_path = project_root / config_path
if not config_path.exists():
    st.error(f"Configuration not found: {config_path.resolve()}")
    st.stop()
cfg = load_config(config_path)
artifacts = Path(cfg["data"]["output"])
if not artifacts.is_absolute():
    artifacts = project_root / artifacts
st.subheader("Patient-wise LOSO results")
metrics_file = artifacts / "all_loso_patient_metrics.csv"
if metrics_file.exists():
    results = pd.read_csv(metrics_file)
    st.dataframe(results, use_container_width=True)
else:
    st.write("No LOSO results yet. Run the training command in the project README.")
external_files = sorted(artifacts.glob("horizon_*/complete/external_siena/siena_patient_metrics.csv"))
if external_files:
    selected_external = st.selectbox("Optional Siena report", external_files,
                                     format_func=lambda p: str(p.relative_to(artifacts)))
    st.dataframe(pd.read_csv(selected_external), use_container_width=True)

st.subheader("Run prediction on an EDF")
st.caption("Choose a trained patient-fold checkpoint and an EDF. When you use the downloaded CHB-MIT data, select the same patient as the checkpoint.")
checkpoints = sorted(artifacts.glob("horizon_*/complete/*.pt"))
checkpoint = None
if checkpoints:
    selected_checkpoint = st.selectbox(
        "Checkpoint (held-out patient fold)",
        checkpoints,
        format_func=lambda path: f"{path.stem} — {path.parent.parent.name}",
    )
    checkpoint = str(selected_checkpoint.resolve())
else:
    checkpoint_text = st.text_input("Checkpoint path (.pt)").strip()
    checkpoint = checkpoint_text or None
    st.info("No trained checkpoint is available yet. When training finishes a fold, its .pt file appears under artifacts.")

recording_mode = st.selectbox(
    "EEG recording source",
    ["Downloaded CHB-MIT data on E:", "Upload an EDF file"],
)
uploaded = None
selected_edf = None
if recording_mode == "Downloaded CHB-MIT data on E:":
    data_root = Path(cfg["data"]["root"])
    if not data_root.is_absolute():
        data_root = project_root / data_root
    data_root = data_root.resolve()
    local_edfs = sorted(data_root.glob("**/*.edf")) if data_root.exists() else []
    if local_edfs:
        checkpoint_patient = Path(checkpoint).stem.lower() if checkpoint else ""
        matching_edfs = [path for path in local_edfs if path.parent.name.lower() == checkpoint_patient]
        edf_choices = matching_edfs or local_edfs
        if checkpoint and not matching_edfs:
            st.warning("No EDFs were found in the matching patient folder. Choose an available EDF or switch to upload.")
        selected_edf = st.selectbox(
            "EEG recording (.edf)",
            edf_choices,
            format_func=lambda path: str(path.relative_to(project_root)),
        )
    else:
        st.info(f"No EDF files found under {data_root.resolve()}. Switch to upload, or put the downloaded CHB-MIT patient folders there.")
else:
    uploaded = st.file_uploader("EEG recording (.edf)", type=["edf"])

show_ig = st.checkbox("Explain the highest-scoring window with Integrated Gradients (slower)", value=False)
run_shap = st.checkbox("Also compute SHAP explanations (slower)", value=False)
has_recording = selected_edf is not None or uploaded is not None
if st.button("Predict windows", disabled=not (checkpoint and has_recording)):
    temp_path = None
    input_path = selected_edf
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as f:
            f.write(uploaded.getvalue())
            temp_path = Path(f.name)
        input_path = temp_path
    try:
        from seizure_prediction.inference import predict_edf, explain_edf_window
        from seizure_prediction.data import CHB_CHANNELS
        device = cfg["training"]["device"]
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        output = predict_edf(input_path, checkpoint, device)
        st.markdown("**Primary prediction: preictal vs non-preictal**")
        st.line_chart(output.set_index("start_time")[["preictal_probability"]])
        attention_columns = [c for c in output.columns if c.startswith("attention_")]
        if attention_columns:
            st.markdown("**Attention across active feature domains**")
            st.line_chart(output.set_index("start_time")[attention_columns])
        st.markdown("**Window scores and supporting three-state classification**")
        st.dataframe(output, use_container_width=True)
        st.download_button("Download window predictions", output.to_csv(index=False),
                           file_name="window_predictions.csv", mime="text/csv")
        if show_ig:
            best = int(output["preictal_probability"].to_numpy().argmax())
            with st.spinner("Calculating input attributions for the highest-scoring window..."):
                explanation = explain_edf_window(input_path, checkpoint, best, device, use_shap=run_shap)
            st.subheader(f"Explanation for window {best} (input influence only)")
            st.caption("These maps describe model input influence. They do not identify a clinical seizure focus.")
            import numpy as np
            import matplotlib.pyplot as plt
            def show_importance(title, imp):
                st.markdown(f"**{title}**")
                channel_names = CHB_CHANNELS[:len(imp["temporal_channel_importance"])]
                channel_frame = pd.DataFrame({
                    "temporal": imp["temporal_channel_importance"],
                    "spectral": imp["spectral_channel_importance"],
                    "time_frequency": imp["time_frequency_channel_importance"]}, index=channel_names)
                st.markdown("Channel importance by input domain")
                st.bar_chart(channel_frame)
                st.bar_chart(pd.DataFrame({"importance": imp["spectral_band_importance"]},
                                          index=["delta", "theta", "alpha", "beta", "gamma"]))
                time = np.arange(len(imp["temporal_time_importance"])) / cfg["data"]["sampling_rate"]
                st.line_chart(pd.DataFrame({"importance": imp["temporal_time_importance"]}, index=time))
                fig, ax = plt.subplots(figsize=(9, 3.5))
                ax.imshow(imp["time_frequency_region_importance"], origin="lower", aspect="auto",
                          extent=[0, cfg["data"]["window_seconds"], 0,
                                  cfg["data"]["high_hz"]], cmap="magma")
                ax.set(xlabel="Time within window (s)", ylabel="Frequency (Hz)", title="Time-frequency attribution")
                st.pyplot(fig); plt.close(fig)
            show_importance("Integrated Gradients", explanation["integrated_gradients"])
            if explanation["shap"] is not None:
                show_importance("SHAP", explanation["shap"])
    except Exception as exc:
        st.error(str(exc))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
