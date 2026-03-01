"""Models package for Audio-to-MIDI conversion."""

from .baseline import BaselineAudioToMIDI, ImprovedBaselineAudioToMIDI, BaselineConfig
from .deep_learning import (
    AudioToMIDITransformer,
    AudioToMIDICNN,
    AudioToMIDILSTM,
    create_model,
    DeepLearningConfig,
    CTCHead,
)

__all__ = [
    "BaselineAudioToMIDI",
    "ImprovedBaselineAudioToMIDI", 
    "BaselineConfig",
    "AudioToMIDITransformer",
    "AudioToMIDICNN",
    "AudioToMIDILSTM",
    "create_model",
    "DeepLearningConfig",
    "CTCHead",
]
