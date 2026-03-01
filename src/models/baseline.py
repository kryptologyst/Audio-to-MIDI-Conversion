"""Baseline Audio-to-MIDI conversion model using traditional signal processing."""

import warnings
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import librosa
from scipy import signal

from ..utils.audio import detect_onsets, detect_pitch, extract_features
from ..utils.midi import NoteEvent, hz_to_midi, quantize_time, merge_overlapping_notes


@dataclass
class BaselineConfig:
    """Configuration for baseline Audio-to-MIDI model."""
    # Audio processing parameters
    sr: int = 22050
    hop_length: int = 512
    n_fft: int = 2048
    
    # Onset detection parameters
    onset_threshold: float = 0.1
    onset_pre_max: int = 3
    onset_post_max: int = 3
    onset_pre_avg: int = 3
    onset_post_avg: int = 5
    onset_delta: float = 0.2
    onset_wait: int = 0
    
    # Pitch detection parameters
    pitch_fmin: float = 80.0
    pitch_fmax: float = 2000.0
    pitch_threshold: float = 0.1
    
    # MIDI conversion parameters
    min_note_duration: float = 0.05
    min_velocity: int = 20
    quantization_resolution: float = 0.125
    merge_tolerance: float = 0.05


class BaselineAudioToMIDI:
    """
    Baseline Audio-to-MIDI conversion using traditional signal processing.
    
    This model uses onset detection and pitch tracking to convert audio to MIDI.
    It serves as a simple baseline for comparison with more advanced models.
    """
    
    def __init__(self, config: Optional[BaselineConfig] = None):
        """Initialize the baseline model."""
        self.config = config or BaselineConfig()
        
    def predict(self, audio: np.ndarray, sr: int) -> List[NoteEvent]:
        """
        Convert audio to MIDI note events.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            List of NoteEvent objects
        """
        # Resample if necessary
        if sr != self.config.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sr)
            sr = self.config.sr
        
        # Detect onsets
        onset_times, onset_frames = detect_onsets(
            audio=audio,
            sr=sr,
            hop_length=self.config.hop_length,
            threshold=self.config.onset_threshold,
            pre_max=self.config.onset_pre_max,
            post_max=self.config.onset_post_max,
            pre_avg=self.config.onset_pre_avg,
            post_avg=self.config.onset_post_avg,
            delta=self.config.onset_delta,
            wait=self.config.onset_wait,
        )
        
        # Detect pitch
        pitches, times = detect_pitch(
            audio=audio,
            sr=sr,
            hop_length=self.config.hop_length,
            fmin=self.config.pitch_fmin,
            fmax=self.config.pitch_fmax,
            threshold=self.config.pitch_threshold,
        )
        
        # Convert to MIDI notes
        notes = self._convert_to_midi_notes(onset_times, onset_frames, pitches, times)
        
        # Post-process notes
        notes = self._post_process_notes(notes)
        
        return notes
    
    def _convert_to_midi_notes(
        self,
        onset_times: np.ndarray,
        onset_frames: np.ndarray,
        pitches: np.ndarray,
        times: np.ndarray,
    ) -> List[NoteEvent]:
        """Convert onset and pitch information to MIDI notes."""
        notes = []
        
        for onset_time, onset_frame in zip(onset_times, onset_frames):
            # Find the closest pitch frame
            pitch_idx = np.argmin(np.abs(times - onset_time))
            
            if pitch_idx < len(pitches) and pitches[pitch_idx] > 0:
                # Convert frequency to MIDI note
                midi_note = int(round(hz_to_midi(pitches[pitch_idx])))
                
                # Ensure MIDI note is in valid range
                if 0 <= midi_note <= 127:
                    # Estimate note duration (simplified)
                    duration = self._estimate_note_duration(onset_time, onset_times)
                    
                    note = NoteEvent(
                        pitch=midi_note,
                        velocity=64,  # Default velocity
                        start_time=onset_time,
                        end_time=onset_time + duration,
                        duration=duration,
                    )
                    notes.append(note)
        
        return notes
    
    def _estimate_note_duration(self, onset_time: float, all_onsets: np.ndarray) -> float:
        """Estimate note duration based on onset spacing."""
        # Find next onset
        next_onsets = all_onsets[all_onsets > onset_time]
        
        if len(next_onsets) > 0:
            next_onset = next_onsets[0]
            duration = min(next_onset - onset_time, 2.0)  # Cap at 2 seconds
        else:
            duration = 1.0  # Default duration
        
        return max(duration, self.config.min_note_duration)
    
    def _post_process_notes(self, notes: List[NoteEvent]) -> List[NoteEvent]:
        """Post-process notes with filtering and quantization."""
        # Filter by minimum duration
        notes = [note for note in notes if note.duration >= self.config.min_note_duration]
        
        # Filter by minimum velocity
        notes = [note for note in notes if note.velocity >= self.config.min_velocity]
        
        # Quantize timing
        notes = [NoteEvent(
            pitch=note.pitch,
            velocity=note.velocity,
            start_time=quantize_time(note.start_time, self.config.quantization_resolution),
            end_time=quantize_time(note.end_time, self.config.quantization_resolution),
            duration=quantize_time(note.duration, self.config.quantization_resolution),
        ) for note in notes]
        
        # Merge overlapping notes
        notes = merge_overlapping_notes(notes, self.config.merge_tolerance)
        
        return notes
    
    def get_config(self) -> BaselineConfig:
        """Get model configuration."""
        return self.config
    
    def set_config(self, config: BaselineConfig) -> None:
        """Set model configuration."""
        self.config = config


class ImprovedBaselineAudioToMIDI(BaselineAudioToMIDI):
    """
    Improved baseline model with better pitch tracking and note duration estimation.
    """
    
    def __init__(self, config: Optional[BaselineConfig] = None):
        """Initialize the improved baseline model."""
        super().__init__(config)
        
    def predict(self, audio: np.ndarray, sr: int) -> List[NoteEvent]:
        """Convert audio to MIDI with improved algorithms."""
        # Resample if necessary
        if sr != self.config.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sr)
            sr = self.config.sr
        
        # Extract comprehensive features
        features = extract_features(
            audio=audio,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )
        
        # Use onset strength for better onset detection
        onset_strength = features["onset_strength"]
        onset_times, onset_frames = detect_onsets(
            audio=audio,
            sr=sr,
            hop_length=self.config.hop_length,
            threshold=self.config.onset_threshold,
        )
        
        # Use chroma features for better pitch tracking
        chroma = features["chroma"]
        pitches, times = self._improved_pitch_detection(audio, sr, chroma)
        
        # Convert to MIDI notes with better duration estimation
        notes = self._convert_to_midi_notes_improved(
            onset_times, onset_frames, pitches, times, features
        )
        
        # Post-process notes
        notes = self._post_process_notes(notes)
        
        return notes
    
    def _improved_pitch_detection(
        self, 
        audio: np.ndarray, 
        sr: int, 
        chroma: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Improved pitch detection using chroma features."""
        # Use librosa's advanced pitch tracking
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=sr,
            hop_length=self.config.hop_length,
            fmin=self.config.pitch_fmin,
            fmax=self.config.pitch_fmax,
            threshold=self.config.pitch_threshold,
        )
        
        # Extract the most prominent pitch at each frame
        pitch_values = []
        for t in range(pitches.shape[1]):
            pitch = pitches[:, t]
            magnitude = magnitudes[:, t]
            
            # Find pitch with highest magnitude
            if np.max(magnitude) > self.config.pitch_threshold:
                index = np.argmax(magnitude)
                pitch_values.append(pitch[index])
            else:
                pitch_values.append(0.0)
        
        times = librosa.frames_to_time(
            np.arange(len(pitch_values)), 
            sr=sr, 
            hop_length=self.config.hop_length
        )
        
        return np.array(pitch_values), times
    
    def _convert_to_midi_notes_improved(
        self,
        onset_times: np.ndarray,
        onset_frames: np.ndarray,
        pitches: np.ndarray,
        times: np.ndarray,
        features: Dict[str, np.ndarray],
    ) -> List[NoteEvent]:
        """Convert to MIDI notes with improved duration estimation."""
        notes = []
        
        for onset_time, onset_frame in zip(onset_times, onset_frames):
            # Find the closest pitch frame
            pitch_idx = np.argmin(np.abs(times - onset_time))
            
            if pitch_idx < len(pitches) and pitches[pitch_idx] > 0:
                # Convert frequency to MIDI note
                midi_note = int(round(hz_to_midi(pitches[pitch_idx])))
                
                # Ensure MIDI note is in valid range
                if 0 <= midi_note <= 127:
                    # Improved duration estimation using RMS energy
                    duration = self._estimate_note_duration_improved(
                        onset_time, onset_times, features["rms"], times
                    )
                    
                    # Estimate velocity from RMS energy
                    velocity = self._estimate_velocity(features["rms"], pitch_idx)
                    
                    note = NoteEvent(
                        pitch=midi_note,
                        velocity=velocity,
                        start_time=onset_time,
                        end_time=onset_time + duration,
                        duration=duration,
                    )
                    notes.append(note)
        
        return notes
    
    def _estimate_note_duration_improved(
        self,
        onset_time: float,
        all_onsets: np.ndarray,
        rms: np.ndarray,
        times: np.ndarray,
    ) -> float:
        """Estimate note duration using RMS energy decay."""
        # Find next onset
        next_onsets = all_onsets[all_onsets > onset_time]
        
        if len(next_onsets) > 0:
            next_onset = next_onsets[0]
            max_duration = next_onset - onset_time
        else:
            max_duration = 2.0
        
        # Find RMS frame corresponding to onset
        onset_idx = np.argmin(np.abs(times - onset_time))
        
        # Find where RMS drops significantly
        onset_rms = rms[0, onset_idx]
        threshold_rms = onset_rms * 0.3  # 30% of onset RMS
        
        # Look for RMS drop within reasonable time window
        search_end = min(onset_idx + int(max_duration * len(times) / times[-1]), len(times))
        
        for i in range(onset_idx, search_end):
            if rms[0, i] < threshold_rms:
                duration = times[i] - onset_time
                return max(duration, self.config.min_note_duration)
        
        return max(max_duration, self.config.min_note_duration)
    
    def _estimate_velocity(self, rms: np.ndarray, frame_idx: int) -> int:
        """Estimate MIDI velocity from RMS energy."""
        if frame_idx < rms.shape[1]:
            energy = rms[0, frame_idx]
            # Normalize energy to velocity range (20-127)
            velocity = int(20 + (energy / np.max(rms)) * 107)
            return min(max(velocity, 20), 127)
        return 64
