"""Data loading and preprocessing for Audio-to-MIDI conversion."""

import warnings
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf

from ..utils.audio import load_audio, extract_features, resample_audio
from ..utils.midi import load_midi_file, NoteEvent, MIDISequence


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    # Audio parameters
    sr: int = 22050
    hop_length: int = 512
    n_fft: int = 2048
    n_mels: int = 128
    fmin: float = 0.0
    fmax: Optional[float] = None
    
    # Data augmentation
    enable_augmentation: bool = True
    pitch_shift_range: Tuple[float, float] = (-2.0, 2.0)
    time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    noise_level: float = 0.01
    
    # Preprocessing
    normalize_audio: bool = True
    apply_preemphasis: bool = False
    preemphasis_coeff: float = 0.97
    
    # MIDI processing
    quantization_resolution: float = 0.125
    min_note_duration: float = 0.05
    max_polyphony: int = 10
    
    # Dataset parameters
    max_audio_length: float = 30.0  # seconds
    min_audio_length: float = 1.0   # seconds


class AudioMIDIDataset(Dataset):
    """
    Dataset for Audio-to-MIDI conversion.
    
    This dataset loads audio files and their corresponding MIDI files,
    extracts features, and prepares them for training.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        config: DataConfig,
        split: str = "train",
        metadata_file: Optional[str] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing audio and MIDI files
            config: Data configuration
            split: Dataset split ("train", "val", "test")
            metadata_file: Optional metadata CSV file
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        
        # Load metadata if provided
        if metadata_file:
            self.metadata = pd.read_csv(metadata_file)
            self.metadata = self.metadata[self.metadata['split'] == split]
        else:
            # Auto-discover files
            self.metadata = self._discover_files()
        
        # Filter by audio length
        self.metadata = self._filter_by_length()
        
    def _discover_files(self) -> pd.DataFrame:
        """Auto-discover audio and MIDI files."""
        audio_files = list(self.data_dir.glob("**/*.wav")) + \
                     list(self.data_dir.glob("**/*.mp3")) + \
                     list(self.data_dir.glob("**/*.flac"))
        
        metadata = []
        for audio_file in audio_files:
            # Look for corresponding MIDI file
            midi_file = audio_file.with_suffix('.mid')
            if not midi_file.exists():
                midi_file = audio_file.with_suffix('.midi')
            
            if midi_file.exists():
                metadata.append({
                    'audio_path': str(audio_file),
                    'midi_path': str(midi_file),
                    'split': self.split,
                })
        
        return pd.DataFrame(metadata)
    
    def _filter_by_length(self) -> pd.DataFrame:
        """Filter files by audio length."""
        valid_files = []
        
        for _, row in self.metadata.iterrows():
            try:
                # Load audio to check length
                audio, sr = load_audio(row['audio_path'], sr=None)
                duration = len(audio) / sr
                
                if self.config.min_audio_length <= duration <= self.config.max_audio_length:
                    valid_files.append(row)
            except Exception as e:
                print(f"Warning: Could not load {row['audio_path']}: {e}")
                continue
        
        return pd.DataFrame(valid_files)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing audio features and MIDI targets
        """
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio, sr = load_audio(
            row['audio_path'],
            sr=self.config.sr,
            normalize=self.config.normalize_audio,
        )
        
        # Load MIDI
        midi_sequence = load_midi_file(row['midi_path'])
        
        # Apply data augmentation for training
        if self.split == "train" and self.config.enable_augmentation:
            audio = self._augment_audio(audio, sr)
        
        # Extract audio features
        features = extract_features(
            audio=audio,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
        )
        
        # Convert MIDI to target format
        targets = self._midi_to_targets(midi_sequence, features)
        
        # Convert to tensors
        audio_features = torch.tensor(features["log_mel_spec"], dtype=torch.float32)
        
        return {
            "audio_features": audio_features,
            "midi_events": targets["events"],
            "midi_pitches": targets["pitches"],
            "midi_velocities": targets["velocities"],
            "audio_path": row['audio_path'],
            "midi_path": row['midi_path'],
        }
    
    def _augment_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply data augmentation to audio."""
        # Pitch shifting
        if self.config.pitch_shift_range[0] != self.config.pitch_shift_range[1]:
            pitch_shift = np.random.uniform(
                self.config.pitch_shift_range[0],
                self.config.pitch_shift_range[1]
            )
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
        
        # Time stretching
        if self.config.time_stretch_range[0] != self.config.time_stretch_range[1]:
            time_stretch = np.random.uniform(
                self.config.time_stretch_range[0],
                self.config.time_stretch_range[1]
            )
            audio = librosa.effects.time_stretch(audio, rate=time_stretch)
        
        # Add noise
        if self.config.noise_level > 0:
            noise = np.random.normal(0, self.config.noise_level, len(audio))
            audio = audio + noise
        
        return audio
    
    def _midi_to_targets(
        self, 
        midi_sequence: MIDISequence, 
        features: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert MIDI sequence to target tensors.
        
        Args:
            midi_sequence: MIDI sequence object
            features: Audio features dictionary
            
        Returns:
            Dictionary containing target tensors
        """
        seq_len = features["log_mel_spec"].shape[1]
        hop_length = features["hop_length"]
        sr = features["sr"]
        
        # Initialize target tensors
        events = torch.zeros(seq_len, dtype=torch.long)
        pitches = torch.zeros(seq_len, dtype=torch.long)
        velocities = torch.zeros(seq_len, dtype=torch.long)
        
        # Convert MIDI notes to frame-based targets
        for note in midi_sequence.notes:
            # Convert time to frame indices
            start_frame = int(note.start_time * sr / hop_length)
            end_frame = int(note.end_time * sr / hop_length)
            
            # Ensure indices are within bounds
            start_frame = max(0, min(start_frame, seq_len - 1))
            end_frame = max(0, min(end_frame, seq_len - 1))
            
            # Set note on event
            if start_frame < seq_len:
                events[start_frame] = 1  # note_on
                pitches[start_frame] = note.pitch
                velocities[start_frame] = note.velocity
            
            # Set note off event
            if end_frame < seq_len:
                events[end_frame] = 2  # note_off
                pitches[end_frame] = note.pitch
                velocities[end_frame] = note.velocity
        
        return {
            "events": events,
            "pitches": pitches,
            "velocities": velocities,
        }


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing and demonstration.
    
    This dataset generates synthetic audio-MIDI pairs for testing
    the Audio-to-MIDI conversion pipeline.
    """
    
    def __init__(
        self,
        config: DataConfig,
        num_samples: int = 100,
        duration_range: Tuple[float, float] = (2.0, 10.0),
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            config: Data configuration
            num_samples: Number of synthetic samples to generate
            duration_range: Range of audio durations in seconds
        """
        self.config = config
        self.num_samples = num_samples
        self.duration_range = duration_range
        
    def __len__(self) -> int:
        """Return dataset length."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a synthetic sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing synthetic audio features and MIDI targets
        """
        # Generate random duration
        duration = np.random.uniform(*self.duration_range)
        
        # Generate synthetic MIDI sequence
        midi_sequence = self._generate_synthetic_midi(duration)
        
        # Generate corresponding audio
        audio = self._generate_synthetic_audio(midi_sequence, duration)
        
        # Extract features
        features = extract_features(
            audio=audio,
            sr=self.config.sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
        )
        
        # Convert MIDI to targets
        targets = self._midi_to_targets(midi_sequence, features)
        
        # Convert to tensors
        audio_features = torch.tensor(features["log_mel_spec"], dtype=torch.float32)
        
        return {
            "audio_features": audio_features,
            "midi_events": targets["events"],
            "midi_pitches": targets["pitches"],
            "midi_velocities": targets["velocities"],
            "audio_path": f"synthetic_{idx}",
            "midi_path": f"synthetic_{idx}",
        }
    
    def _generate_synthetic_midi(self, duration: float) -> MIDISequence:
        """Generate a synthetic MIDI sequence."""
        notes = []
        
        # Generate random notes
        num_notes = np.random.randint(5, 20)
        
        for _ in range(num_notes):
            # Random pitch (C4 to C6)
            pitch = np.random.randint(60, 84)
            
            # Random velocity
            velocity = np.random.randint(40, 100)
            
            # Random timing
            start_time = np.random.uniform(0, duration - 0.5)
            duration_note = np.random.uniform(0.2, 2.0)
            end_time = min(start_time + duration_note, duration)
            
            note = NoteEvent(
                pitch=pitch,
                velocity=velocity,
                start_time=start_time,
                end_time=end_time,
                duration=duration_note,
            )
            notes.append(note)
        
        return MIDISequence(notes=notes, tempo=120.0)
    
    def _generate_synthetic_audio(self, midi_sequence: MIDISequence, duration: float) -> np.ndarray:
        """Generate synthetic audio from MIDI sequence."""
        sr = self.config.sr
        audio = np.zeros(int(duration * sr))
        
        for note in midi_sequence.notes:
            # Generate sine wave for each note
            frequency = 440.0 * (2 ** ((note.pitch - 69) / 12.0))
            
            start_sample = int(note.start_time * sr)
            end_sample = int(note.end_time * sr)
            
            if start_sample < len(audio) and end_sample <= len(audio):
                t = np.linspace(0, note.duration, end_sample - start_sample)
                note_audio = np.sin(2 * np.pi * frequency * t)
                
                # Apply envelope
                envelope = np.exp(-t * 2)  # Simple decay
                note_audio *= envelope
                
                # Scale by velocity
                note_audio *= note.velocity / 127.0
                
                audio[start_sample:end_sample] += note_audio
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def _midi_to_targets(
        self, 
        midi_sequence: MIDISequence, 
        features: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Convert MIDI sequence to target tensors (same as AudioMIDIDataset)."""
        seq_len = features["log_mel_spec"].shape[1]
        hop_length = features["hop_length"]
        sr = features["sr"]
        
        events = torch.zeros(seq_len, dtype=torch.long)
        pitches = torch.zeros(seq_len, dtype=torch.long)
        velocities = torch.zeros(seq_len, dtype=torch.long)
        
        for note in midi_sequence.notes:
            start_frame = int(note.start_time * sr / hop_length)
            end_frame = int(note.end_time * sr / hop_length)
            
            start_frame = max(0, min(start_frame, seq_len - 1))
            end_frame = max(0, min(end_frame, seq_len - 1))
            
            if start_frame < seq_len:
                events[start_frame] = 1
                pitches[start_frame] = note.pitch
                velocities[start_frame] = note.velocity
            
            if end_frame < seq_len:
                events[end_frame] = 2
                pitches[end_frame] = note.pitch
                velocities[end_frame] = note.velocity
        
        return {
            "events": events,
            "pitches": pitches,
            "velocities": velocities,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset object
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        DataLoader object
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching variable-length sequences.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched tensors
    """
    # Pad sequences to same length
    max_len = max(sample["audio_features"].shape[1] for sample in batch)
    
    batched_audio = []
    batched_events = []
    batched_pitches = []
    batched_velocities = []
    
    for sample in batch:
        audio_len = sample["audio_features"].shape[1]
        
        # Pad audio features
        if audio_len < max_len:
            pad_len = max_len - audio_len
            audio_padded = F.pad(
                sample["audio_features"].T, 
                (0, pad_len), 
                mode='constant', 
                value=0
            ).T
        else:
            audio_padded = sample["audio_features"]
        
        batched_audio.append(audio_padded)
        
        # Pad MIDI targets
        if audio_len < max_len:
            pad_len = max_len - audio_len
            events_padded = F.pad(sample["midi_events"], (0, pad_len), value=0)
            pitches_padded = F.pad(sample["midi_pitches"], (0, pad_len), value=0)
            velocities_padded = F.pad(sample["midi_velocities"], (0, pad_len), value=0)
        else:
            events_padded = sample["midi_events"]
            pitches_padded = sample["midi_pitches"]
            velocities_padded = sample["midi_velocities"]
        
        batched_events.append(events_padded)
        batched_pitches.append(pitches_padded)
        batched_velocities.append(velocities_padded)
    
    return {
        "audio_features": torch.stack(batched_audio),
        "midi_events": torch.stack(batched_events),
        "midi_pitches": torch.stack(batched_pitches),
        "midi_velocities": torch.stack(batched_velocities),
    }
