"""Unit tests for Audio-to-MIDI conversion."""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.baseline import BaselineAudioToMIDI, BaselineConfig
from src.data.dataset import SyntheticDataset, DataConfig
from src.utils.device import get_device, set_seed
from src.utils.audio import load_audio, extract_features
from src.utils.midi import NoteEvent, hz_to_midi, midi_to_hz


class TestBaselineModel:
    """Test baseline Audio-to-MIDI model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        config = BaselineConfig()
        model = BaselineAudioToMIDI(config)
        assert model is not None
        assert model.config.sr == 22050
    
    def test_model_prediction(self):
        """Test model prediction on synthetic data."""
        config = BaselineConfig()
        model = BaselineAudioToMIDI(config)
        
        # Create synthetic audio (sine wave)
        sr = 22050
        duration = 2.0
        frequency = 440.0  # A4
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Predict MIDI notes
        notes = model.predict(audio, sr)
        
        # Should detect at least one note
        assert len(notes) > 0
        
        # Check note properties
        note = notes[0]
        assert isinstance(note, NoteEvent)
        assert 0 <= note.pitch <= 127
        assert 0 <= note.velocity <= 127
        assert note.start_time >= 0
        assert note.end_time > note.start_time


class TestDataUtils:
    """Test data utility functions."""
    
    def test_hz_to_midi_conversion(self):
        """Test frequency to MIDI conversion."""
        # A4 = 440 Hz = MIDI note 69
        assert hz_to_midi(440.0) == 69.0
        
        # C4 = 261.63 Hz = MIDI note 60
        assert abs(hz_to_midi(261.63) - 60.0) < 0.1
        
        # Test edge cases
        assert hz_to_midi(0) == 0.0
        assert hz_to_midi(-100) == 0.0
    
    def test_midi_to_hz_conversion(self):
        """Test MIDI to frequency conversion."""
        # MIDI note 69 = A4 = 440 Hz
        assert midi_to_hz(69) == 440.0
        
        # MIDI note 60 = C4 = 261.63 Hz
        assert abs(midi_to_hz(60) - 261.63) < 0.1
        
        # Test edge cases
        assert midi_to_hz(0) > 0
        assert midi_to_hz(127) > 0
    
    def test_note_event_creation(self):
        """Test NoteEvent creation."""
        note = NoteEvent(
            pitch=60,
            velocity=64,
            start_time=0.0,
            end_time=1.0,
            duration=1.0
        )
        
        assert note.pitch == 60
        assert note.velocity == 64
        assert note.start_time == 0.0
        assert note.end_time == 1.0
        assert note.duration == 1.0


class TestSyntheticDataset:
    """Test synthetic dataset generation."""
    
    def test_dataset_creation(self):
        """Test synthetic dataset creation."""
        config = DataConfig()
        dataset = SyntheticDataset(config, num_samples=5)
        
        assert len(dataset) == 5
    
    def test_dataset_sample(self):
        """Test dataset sample structure."""
        config = DataConfig()
        dataset = SyntheticDataset(config, num_samples=1)
        
        sample = dataset[0]
        
        # Check sample structure
        assert "audio_features" in sample
        assert "midi_events" in sample
        assert "midi_pitches" in sample
        assert "midi_velocities" in sample
        assert "audio_path" in sample
        assert "midi_path" in sample
        
        # Check tensor shapes
        assert isinstance(sample["audio_features"], torch.Tensor)
        assert isinstance(sample["midi_events"], torch.Tensor)
        assert isinstance(sample["midi_pitches"], torch.Tensor)
        assert isinstance(sample["midi_velocities"], torch.Tensor)
        
        # Check dimensions
        assert len(sample["audio_features"].shape) == 2  # (features, time)
        assert len(sample["midi_events"].shape) == 1     # (time,)


class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device is not None
        assert isinstance(device, torch.device)
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Test numpy randomness
        np.random.seed(42)
        val1 = np.random.random()
        
        set_seed(42)
        val2 = np.random.random()
        
        assert val1 == val2
    
    def test_device_info(self):
        """Test device information."""
        from src.utils.device import get_device_info
        
        info = get_device_info()
        assert "device" in info
        assert isinstance(info["device"], str)


class TestAudioUtils:
    """Test audio utility functions."""
    
    def test_extract_features(self):
        """Test audio feature extraction."""
        # Create synthetic audio
        sr = 22050
        duration = 2.0
        frequency = 440.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Extract features
        features = extract_features(audio, sr)
        
        # Check feature structure
        expected_features = [
            "log_mel_spec", "chroma", "spectral_centroid", 
            "zcr", "rms", "onset_strength", "tempo", "beats"
        ]
        
        for feature_name in expected_features:
            assert feature_name in features
        
        # Check feature shapes
        assert features["log_mel_spec"].shape[0] == 128  # n_mels
        assert features["chroma"].shape[0] == 12        # chroma bins
        assert features["onset_strength"].shape[0] == 1  # single channel


if __name__ == "__main__":
    pytest.main([__file__])
