"""Audio processing utilities for Audio-to-MIDI conversion."""

import warnings
from typing import Tuple, Optional, List, Dict, Any

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy import signal


def load_audio(
    file_path: str,
    sr: Optional[int] = None,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with proper error handling.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (None to keep original)
        mono: Convert to mono if True
        normalize: Normalize audio to [-1, 1] if True
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        audio, sr = librosa.load(
            file_path, 
            sr=sr, 
            mono=mono,
            res_type='kaiser_fast'
        )
        
        if normalize:
            audio = librosa.util.normalize(audio)
            
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {e}")


def resample_audio(
    audio: np.ndarray, 
    orig_sr: int, 
    target_sr: int
) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    
    return librosa.resample(
        audio, 
        orig_sr=orig_sr, 
        target_sr=target_sr,
        res_type='kaiser_fast'
    )


def extract_features(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract comprehensive audio features for MIDI conversion.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bins
        fmin: Minimum frequency for mel scale
        fmax: Maximum frequency for mel scale
        
    Returns:
        Dictionary containing various audio features
    """
    if fmax is None:
        fmax = sr // 2
    
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    
    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=n_fft,
        hop_length=hop_length,
    )
    
    # RMS energy
    rms = librosa.feature.rms(
        y=audio,
        frame_length=n_fft,
        hop_length=hop_length,
    )
    
    # Onset strength
    onset_strength = librosa.onset.onset_strength(
        y=audio,
        sr=sr,
        hop_length=hop_length,
    )
    
    # Tempo and beats
    tempo, beats = librosa.beat.beat_track(
        y=audio,
        sr=sr,
        hop_length=hop_length,
    )
    
    return {
        "log_mel_spec": log_mel_spec,
        "chroma": chroma,
        "spectral_centroid": spectral_centroid,
        "zcr": zcr,
        "rms": rms,
        "onset_strength": onset_strength,
        "tempo": tempo,
        "beats": beats,
        "sr": sr,
        "hop_length": hop_length,
        "n_fft": n_fft,
    }


def detect_onsets(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    threshold: float = 0.1,
    pre_max: int = 3,
    post_max: int = 3,
    pre_avg: int = 3,
    post_avg: int = 5,
    delta: float = 0.2,
    wait: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect note onsets in audio.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        hop_length: Hop length for analysis
        threshold: Onset detection threshold
        pre_max: Pre-maximum window size
        post_max: Post-maximum window size
        pre_avg: Pre-average window size
        post_avg: Post-average window size
        delta: Delta threshold
        wait: Wait time between onsets
        
    Returns:
        Tuple of (onset_times, onset_frames)
    """
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        threshold=threshold,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=wait,
    )
    
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    
    return onset_times, onset_frames


def detect_pitch(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    fmin: float = 80.0,
    fmax: float = 2000.0,
    threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect fundamental frequency (pitch) in audio.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        hop_length: Hop length for analysis
        fmin: Minimum frequency
        fmax: Maximum frequency
        threshold: Pitch detection threshold
        
    Returns:
        Tuple of (pitches, times)
    """
    pitches, magnitudes = librosa.piptrack(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        threshold=threshold,
    )
    
    # Extract the most prominent pitch at each frame
    pitch_values = []
    for t in range(pitches.shape[1]):
        pitch = pitches[:, t]
        index = np.argmax(pitch)
        if magnitudes[index, t] > threshold:
            pitch_values.append(pitch[index])
        else:
            pitch_values.append(0.0)
    
    times = librosa.frames_to_time(np.arange(len(pitch_values)), sr=sr, hop_length=hop_length)
    
    return np.array(pitch_values), times


def apply_preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter to audio."""
    return signal.lfilter([1, -coeff], [1], audio)


def remove_preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Remove pre-emphasis filter from audio."""
    return signal.lfilter([1], [1, -coeff], audio)


def compute_cmvn(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Cepstral Mean and Variance Normalization.
    
    Args:
        features: Feature matrix (n_features, n_frames)
        
    Returns:
        Tuple of (mean, std)
    """
    mean = np.mean(features, axis=1, keepdims=True)
    std = np.std(features, axis=1, keepdims=True)
    
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    
    return mean, std


def apply_cmvn(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply CMVN to features."""
    return (features - mean) / std


def chunk_audio(
    audio: np.ndarray,
    chunk_length: float,
    sr: int,
    overlap: float = 0.0,
) -> List[np.ndarray]:
    """
    Split audio into overlapping chunks.
    
    Args:
        audio: Audio signal
        chunk_length: Length of each chunk in seconds
        sr: Sample rate
        overlap: Overlap ratio between chunks (0.0 to 1.0)
        
    Returns:
        List of audio chunks
    """
    chunk_samples = int(chunk_length * sr)
    hop_samples = int(chunk_samples * (1 - overlap))
    
    chunks = []
    for start in range(0, len(audio) - chunk_samples + 1, hop_samples):
        chunk = audio[start:start + chunk_samples]
        chunks.append(chunk)
    
    return chunks
