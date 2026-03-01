"""Streamlit demo application for Audio-to-MIDI conversion."""

import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.device import get_device, get_device_info
from src.utils.audio import load_audio, extract_features
from src.models.baseline import BaselineAudioToMIDI, BaselineConfig
from src.models.deep_learning import create_model, DeepLearningConfig
from src.utils.midi import create_midi_file, NoteEvent
from src.metrics.evaluation import AudioToMIDIMetrics
from omegaconf import OmegaConf


# Page configuration
st.set_page_config(
    page_title="Audio-to-MIDI Conversion",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Privacy disclaimer
PRIVACY_DISCLAIMER = """
**PRIVACY DISCLAIMER**

This is a research and educational demonstration of Audio-to-MIDI conversion technology. 

**Important Notes:**
- This tool is for research and educational purposes only
- No audio data is stored or transmitted to external servers
- All processing is performed locally on your device
- This technology should not be used for biometric identification or voice cloning in production systems
- Misuse of this technology for creating misleading or harmful content is strictly prohibited

**Intended Use:**
- Music education and analysis
- Research in music information retrieval
- Creative applications in music production
- Academic research and development

By using this application, you agree to use it responsibly and in accordance with ethical guidelines.
"""


def load_config() -> Dict[str, Any]:
    """Load configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if config_path.exists():
        return OmegaConf.load(config_path)
    else:
        # Default configuration
        return {
            "model": {
                "type": "baseline",
                "baseline": {
                    "sr": 22050,
                    "hop_length": 512,
                    "n_fft": 2048,
                    "onset_threshold": 0.1,
                    "pitch_fmin": 80.0,
                    "pitch_fmax": 2000.0,
                    "min_note_duration": 0.05,
                    "quantization_resolution": 0.125,
                }
            },
            "demo": {
                "max_audio_length": 30.0,
                "supported_formats": ["wav", "mp3", "flac", "m4a"],
                "enable_preview": True,
                "enable_metrics": True,
            }
        }


def initialize_model(config: Dict[str, Any]) -> Any:
    """Initialize the Audio-to-MIDI model."""
    if config["model"]["type"] == "baseline":
        model_config = BaselineConfig(**config["model"]["baseline"])
        return BaselineAudioToMIDI(model_config)
    else:
        # For deep learning models, we'd need to load a trained model
        st.warning("Deep learning models require trained checkpoints. Using baseline model.")
        model_config = BaselineConfig(**config["model"]["baseline"])
        return BaselineAudioToMIDI(model_config)


def process_audio(audio_file, model, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process audio file and convert to MIDI."""
    try:
        # Load audio
        audio, sr = load_audio(audio_file, sr=config["model"]["baseline"]["sr"])
        
        # Check audio length
        duration = len(audio) / sr
        max_length = config["demo"]["max_audio_length"]
        
        if duration > max_length:
            st.warning(f"Audio is {duration:.1f}s long. Truncating to {max_length}s.")
            audio = audio[:int(max_length * sr)]
            duration = max_length
        
        # Convert to MIDI
        notes = model.predict(audio, sr)
        
        # Extract features for visualization
        features = extract_features(
            audio=audio,
            sr=sr,
            n_fft=config["model"]["baseline"]["n_fft"],
            hop_length=config["model"]["baseline"]["hop_length"],
            n_mels=128,
        )
        
        return {
            "success": True,
            "notes": notes,
            "audio": audio,
            "sr": sr,
            "duration": duration,
            "features": features,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def visualize_audio_features(features: Dict[str, np.ndarray], duration: float) -> None:
    """Visualize audio features."""
    st.subheader("Audio Analysis")
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("Mel Spectrogram", "Chroma Features", "Onset Strength", "Spectral Centroid", "RMS Energy", "Zero Crossing Rate"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Time axis
    hop_length = features["hop_length"]
    sr = features["sr"]
    times = np.linspace(0, duration, features["log_mel_spec"].shape[1])
    
    # Mel spectrogram
    fig.add_trace(
        go.Heatmap(
            z=features["log_mel_spec"],
            x=times,
            y=np.arange(features["log_mel_spec"].shape[0]),
            colorscale='Viridis',
            name="Mel Spectrogram"
        ),
        row=1, col=1
    )
    
    # Chroma features
    fig.add_trace(
        go.Heatmap(
            z=features["chroma"],
            x=times,
            y=np.arange(features["chroma"].shape[0]),
            colorscale='Blues',
            name="Chroma"
        ),
        row=1, col=2
    )
    
    # Onset strength
    fig.add_trace(
        go.Scatter(
            x=times,
            y=features["onset_strength"],
            mode='lines',
            name="Onset Strength"
        ),
        row=2, col=1
    )
    
    # Spectral centroid
    fig.add_trace(
        go.Scatter(
            x=times,
            y=features["spectral_centroid"][0],
            mode='lines',
            name="Spectral Centroid"
        ),
        row=2, col=2
    )
    
    # RMS energy
    fig.add_trace(
        go.Scatter(
            x=times,
            y=features["rms"][0],
            mode='lines',
            name="RMS Energy"
        ),
        row=3, col=1
    )
    
    # Zero crossing rate
    fig.add_trace(
        go.Scatter(
            x=times,
            y=features["zcr"][0],
            mode='lines',
            name="Zero Crossing Rate"
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Audio Feature Analysis",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def visualize_midi_notes(notes: list, duration: float) -> None:
    """Visualize MIDI notes."""
    st.subheader("MIDI Conversion Results")
    
    if not notes:
        st.warning("No notes detected in the audio.")
        return
    
    # Create piano roll visualization
    fig = go.Figure()
    
    # Add note rectangles
    for note in notes:
        fig.add_trace(go.Scatter(
            x=[note.start_time, note.end_time, note.end_time, note.start_time, note.start_time],
            y=[note.pitch, note.pitch, note.pitch + 1, note.pitch + 1, note.pitch],
            fill='toself',
            mode='lines',
            line=dict(width=0),
            fillcolor=f'rgba(31, 119, 180, {note.velocity/127})',
            name=f'Note {note.pitch}',
            showlegend=False,
            hovertemplate=f'Pitch: {note.pitch}<br>Velocity: {note.velocity}<br>Duration: {note.duration:.2f}s<br>Start: {note.start_time:.2f}s<br>End: {note.end_time:.2f}s'
        ))
    
    # Update layout
    fig.update_layout(
        title="MIDI Piano Roll",
        xaxis_title="Time (seconds)",
        yaxis_title="MIDI Note Number",
        height=400,
        yaxis=dict(
            range=[min([n.pitch for n in notes]) - 5, max([n.pitch for n in notes]) + 5],
            tickmode='linear',
            tick0=60,
            dtick=12,
            ticktext=[f"C{i}" for i in range(3, 7)],
            tickvals=list(range(48, 97, 12))
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display note statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Notes", len(notes))
    
    with col2:
        pitch_range = max([n.pitch for n in notes]) - min([n.pitch for n in notes])
        st.metric("Pitch Range", f"{pitch_range} semitones")
    
    with col3:
        avg_velocity = np.mean([n.velocity for n in notes])
        st.metric("Avg Velocity", f"{avg_velocity:.1f}")
    
    with col4:
        avg_duration = np.mean([n.duration for n in notes])
        st.metric("Avg Duration", f"{avg_duration:.2f}s")


def compute_metrics(notes: list, duration: float) -> Dict[str, float]:
    """Compute conversion metrics."""
    if not notes:
        return {}
    
    metrics = {}
    
    # Basic statistics
    metrics["total_notes"] = len(notes)
    metrics["notes_per_second"] = len(notes) / duration
    
    # Pitch statistics
    pitches = [n.pitch for n in notes]
    metrics["pitch_range"] = max(pitches) - min(pitches)
    metrics["avg_pitch"] = np.mean(pitches)
    metrics["pitch_std"] = np.std(pitches)
    
    # Velocity statistics
    velocities = [n.velocity for n in notes]
    metrics["avg_velocity"] = np.mean(velocities)
    metrics["velocity_std"] = np.std(velocities)
    
    # Duration statistics
    durations = [n.duration for n in notes]
    metrics["avg_duration"] = np.mean(durations)
    metrics["duration_std"] = np.std(durations)
    
    # Polyphony estimation
    time_points = np.linspace(0, duration, int(duration * 10))  # 10 Hz resolution
    polyphony = np.zeros(len(time_points))
    
    for note in notes:
        start_idx = int(note.start_time * 10)
        end_idx = int(note.end_time * 10)
        start_idx = max(0, min(start_idx, len(polyphony) - 1))
        end_idx = max(0, min(end_idx, len(polyphony) - 1))
        polyphony[start_idx:end_idx + 1] += 1
    
    metrics["max_polyphony"] = np.max(polyphony)
    metrics["avg_polyphony"] = np.mean(polyphony)
    
    return metrics


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🎵 Audio-to-MIDI Conversion</h1>', unsafe_allow_html=True)
    
    # Privacy disclaimer
    with st.expander("Privacy Disclaimer", expanded=True):
        st.markdown(PRIVACY_DISCLAIMER)
    
    # Load configuration
    config = load_config()
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["baseline", "transformer", "cnn", "lstm"],
        index=0,
        help="Select the model type for conversion"
    )
    
    # Model parameters
    if model_type == "baseline":
        st.sidebar.subheader("Baseline Model Parameters")
        
        onset_threshold = st.sidebar.slider(
            "Onset Threshold",
            min_value=0.01,
            max_value=0.5,
            value=config["model"]["baseline"]["onset_threshold"],
            step=0.01,
            help="Threshold for onset detection"
        )
        
        pitch_fmin = st.sidebar.number_input(
            "Min Frequency (Hz)",
            min_value=20.0,
            max_value=200.0,
            value=config["model"]["baseline"]["pitch_fmin"],
            step=10.0,
            help="Minimum frequency for pitch detection"
        )
        
        pitch_fmax = st.sidebar.number_input(
            "Max Frequency (Hz)",
            min_value=1000.0,
            max_value=8000.0,
            value=config["model"]["baseline"]["pitch_fmax"],
            step=100.0,
            help="Maximum frequency for pitch detection"
        )
        
        min_duration = st.sidebar.number_input(
            "Min Note Duration (s)",
            min_value=0.01,
            max_value=1.0,
            value=config["model"]["baseline"]["min_note_duration"],
            step=0.01,
            help="Minimum duration for detected notes"
        )
    
    # Initialize model
    if st.sidebar.button("Initialize Model"):
        with st.spinner("Initializing model..."):
            model = initialize_model(config)
            st.session_state.model = model
            st.sidebar.success("Model initialized successfully!")
    
    # Main content
    if "model" not in st.session_state:
        st.info("Please initialize the model using the sidebar.")
        return
    
    model = st.session_state.model
    
    # File upload
    st.header("Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=config["demo"]["supported_formats"],
        help="Supported formats: WAV, MP3, FLAC, M4A"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"File uploaded: {uploaded_file.name}")
        st.info(f"File size: {uploaded_file.size / 1024 / 1024:.2f} MB")
        
        # Process button
        if st.button("Convert to MIDI", type="primary"):
            with st.spinner("Processing audio..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Process audio
                    result = process_audio(tmp_file_path, model, config)
                    
                    if result["success"]:
                        st.success("Audio processed successfully!")
                        
                        # Display results
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Visualize audio features
                            if config["demo"]["enable_preview"]:
                                visualize_audio_features(result["features"], result["duration"])
                            
                            # Visualize MIDI notes
                            visualize_midi_notes(result["notes"], result["duration"])
                        
                        with col2:
                            # Display metrics
                            if config["demo"]["enable_metrics"]:
                                st.subheader("Conversion Metrics")
                                metrics = compute_metrics(result["notes"], result["duration"])
                                
                                for key, value in metrics.items():
                                    if isinstance(value, float):
                                        st.metric(key.replace("_", " ").title(), f"{value:.2f}")
                                    else:
                                        st.metric(key.replace("_", " ").title(), value)
                        
                        # Download MIDI file
                        if result["notes"]:
                            st.subheader("Download MIDI File")
                            
                            # Create MIDI file
                            midi_path = tempfile.mktemp(suffix=".mid")
                            create_midi_file(result["notes"], midi_path)
                            
                            # Read MIDI file for download
                            with open(midi_path, "rb") as f:
                                midi_data = f.read()
                            
                            st.download_button(
                                label="Download MIDI File",
                                data=midi_data,
                                file_name=f"{Path(uploaded_file.name).stem}.mid",
                                mime="audio/midi"
                            )
                            
                            # Clean up
                            os.unlink(midi_path)
                    
                    else:
                        st.error(f"Error processing audio: {result['error']}")
                
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
    
    # Device information
    st.sidebar.subheader("System Information")
    device_info = get_device_info()
    st.sidebar.text(f"Device: {device_info['device']}")
    
    if "cuda_version" in device_info:
        st.sidebar.text(f"CUDA: {device_info['cuda_version']}")
        st.sidebar.text(f"GPU: {device_info['gpu_name']}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Audio-to-MIDI Conversion Demo** | "
        "Built with Streamlit | "
        "For research and educational purposes only"
    )


if __name__ == "__main__":
    main()
