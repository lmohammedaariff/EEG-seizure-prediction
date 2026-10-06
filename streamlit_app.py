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
config_path = Path(st.sidebar.text_input("Configuration file", "config.yaml"))
if not config_path.exists():
    st.error(f"Configuration not found: {config_path.resolve()}")
    st.stop()
cfg = load_config(config_path)
artifacts = Path(cfg["data"]["output"])
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
st.caption("Select a checkpoint for a held-out patient fold. The model produces window scores and both output types.")
checkpoint = st.text_input("Checkpoint path (.pt)")
uploaded = st.file_uploader("EEG recording (.edf)", type=["edf"])
show_ig = st.checkbox("Explain the highest-scoring window with Integrated Gradients", value=True)
run_shap = st.checkbox("Also compute SHAP explanations (slower)", value=False)
if st.button("Predict windows", disabled=not (checkpoint and uploaded)):
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as f:
        f.write(uploaded.getvalue()); temp_path = Path(f.name)
    try:
        from seizure_prediction.inference import predict_edf, explain_edf_window
        from seizure_prediction.data import CHB_CHANNELS
        device = cfg["training"]["device"]
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        output = predict_edf(temp_path, checkpoint, device)
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
                explanation = explain_edf_window(temp_path, checkpoint, best, device, use_shap=run_shap)
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
        temp_path.unlink(missing_ok=True)
