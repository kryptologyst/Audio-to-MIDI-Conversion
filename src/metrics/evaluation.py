"""Evaluation metrics for Audio-to-MIDI conversion."""

import warnings
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from ..utils.midi import NoteEvent, MIDISequence


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    # Timing tolerance
    onset_tolerance: float = 0.05  # seconds
    offset_tolerance: float = 0.1   # seconds
    
    # Pitch tolerance
    pitch_tolerance: int = 1  # semitones
    
    # Velocity tolerance
    velocity_tolerance: int = 10  # MIDI velocity units
    
    # Polyphony evaluation
    max_polyphony: int = 10
    
    # Frame-based evaluation
    frame_tolerance: float = 0.05  # seconds


class AudioToMIDIMetrics:
    """
    Comprehensive evaluation metrics for Audio-to-MIDI conversion.
    
    This class provides various metrics to evaluate the quality of
    Audio-to-MIDI conversion including note-level, frame-level, and
    polyphony-aware metrics.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize metrics calculator."""
        self.config = config or EvaluationConfig()
        
    def evaluate(
        self,
        predicted_notes: List[NoteEvent],
        ground_truth_notes: List[NoteEvent],
        audio_duration: float,
    ) -> Dict[str, float]:
        """
        Evaluate predicted notes against ground truth.
        
        Args:
            predicted_notes: List of predicted NoteEvent objects
            ground_truth_notes: List of ground truth NoteEvent objects
            audio_duration: Duration of audio in seconds
            
        Returns:
            Dictionary containing evaluation metrics
        """
        metrics = {}
        
        # Note-level metrics
        note_metrics = self._compute_note_level_metrics(
            predicted_notes, ground_truth_notes
        )
        metrics.update(note_metrics)
        
        # Frame-level metrics
        frame_metrics = self._compute_frame_level_metrics(
            predicted_notes, ground_truth_notes, audio_duration
        )
        metrics.update(frame_metrics)
        
        # Polyphony metrics
        polyphony_metrics = self._compute_polyphony_metrics(
            predicted_notes, ground_truth_notes, audio_duration
        )
        metrics.update(polyphony_metrics)
        
        # Timing metrics
        timing_metrics = self._compute_timing_metrics(
            predicted_notes, ground_truth_notes
        )
        metrics.update(timing_metrics)
        
        return metrics
    
    def _compute_note_level_metrics(
        self,
        predicted_notes: List[NoteEvent],
        ground_truth_notes: List[NoteEvent],
    ) -> Dict[str, float]:
        """Compute note-level precision, recall, and F1."""
        if not predicted_notes and not ground_truth_notes:
            return {"note_precision": 1.0, "note_recall": 1.0, "note_f1": 1.0}
        
        if not predicted_notes:
            return {"note_precision": 0.0, "note_recall": 0.0, "note_f1": 0.0}
        
        if not ground_truth_notes:
            return {"note_precision": 0.0, "note_recall": 0.0, "note_f1": 0.0}
        
        # Create cost matrix for Hungarian algorithm
        cost_matrix = self._create_cost_matrix(predicted_notes, ground_truth_notes)
        
        # Solve assignment problem
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # Count matches
        matches = 0
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            if cost_matrix[pred_idx, gt_idx] < 1.0:  # Threshold for match
                matches += 1
        
        precision = matches / len(predicted_notes) if predicted_notes else 0.0
        recall = matches / len(ground_truth_notes) if ground_truth_notes else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "note_precision": precision,
            "note_recall": recall,
            "note_f1": f1,
            "note_matches": matches,
            "note_total_predicted": len(predicted_notes),
            "note_total_ground_truth": len(ground_truth_notes),
        }
    
    def _create_cost_matrix(
        self,
        predicted_notes: List[NoteEvent],
        ground_truth_notes: List[NoteEvent],
    ) -> np.ndarray:
        """Create cost matrix for Hungarian algorithm."""
        cost_matrix = np.ones((len(predicted_notes), len(ground_truth_notes)))
        
        for i, pred_note in enumerate(predicted_notes):
            for j, gt_note in enumerate(ground_truth_notes):
                # Check if notes match within tolerance
                if self._notes_match(pred_note, gt_note):
                    cost_matrix[i, j] = 0.0
        
        return cost_matrix
    
    def _notes_match(self, pred_note: NoteEvent, gt_note: NoteEvent) -> bool:
        """Check if two notes match within tolerance."""
        # Pitch match
        pitch_match = abs(pred_note.pitch - gt_note.pitch) <= self.config.pitch_tolerance
        
        # Onset match
        onset_match = abs(pred_note.start_time - gt_note.start_time) <= self.config.onset_tolerance
        
        # Offset match
        offset_match = abs(pred_note.end_time - gt_note.end_time) <= self.config.offset_tolerance
        
        return pitch_match and onset_match and offset_match
    
    def _compute_frame_level_metrics(
        self,
        predicted_notes: List[NoteEvent],
        ground_truth_notes: List[NoteEvent],
        audio_duration: float,
    ) -> Dict[str, float]:
        """Compute frame-level metrics."""
        # Create frame-based representations
        frame_rate = 1.0 / self.config.frame_tolerance
        num_frames = int(audio_duration * frame_rate)
        
        pred_frames = self._notes_to_frames(predicted_notes, num_frames, frame_rate)
        gt_frames = self._notes_to_frames(ground_truth_notes, num_frames, frame_rate)
        
        # Compute frame-level metrics
        frame_accuracy = accuracy_score(gt_frames, pred_frames)
        
        # Precision, recall, F1 for each pitch
        pitch_precisions = []
        pitch_recalls = []
        pitch_f1s = []
        
        for pitch in range(128):  # MIDI pitch range
            gt_pitch_frames = (gt_frames == pitch).astype(int)
            pred_pitch_frames = (pred_frames == pitch).astype(int)
            
            if np.sum(gt_pitch_frames) > 0 or np.sum(pred_pitch_frames) > 0:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    gt_pitch_frames, pred_pitch_frames, average='binary', zero_division=0
                )
                pitch_precisions.append(precision)
                pitch_recalls.append(recall)
                pitch_f1s.append(f1)
        
        avg_precision = np.mean(pitch_precisions) if pitch_precisions else 0.0
        avg_recall = np.mean(pitch_recalls) if pitch_recalls else 0.0
        avg_f1 = np.mean(pitch_f1s) if pitch_f1s else 0.0
        
        return {
            "frame_accuracy": frame_accuracy,
            "frame_precision": avg_precision,
            "frame_recall": avg_recall,
            "frame_f1": avg_f1,
        }
    
    def _notes_to_frames(
        self,
        notes: List[NoteEvent],
        num_frames: int,
        frame_rate: float,
    ) -> np.ndarray:
        """Convert notes to frame-based representation."""
        frames = np.zeros(num_frames, dtype=int)
        
        for note in notes:
            start_frame = int(note.start_time * frame_rate)
            end_frame = int(note.end_time * frame_rate)
            
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(0, min(end_frame, num_frames - 1))
            
            # Handle polyphony by using the highest velocity note
            for frame in range(start_frame, end_frame + 1):
                if frames[frame] == 0 or note.velocity > frames[frame]:
                    frames[frame] = note.pitch
        
        return frames
    
    def _compute_polyphony_metrics(
        self,
        predicted_notes: List[NoteEvent],
        ground_truth_notes: List[NoteEvent],
        audio_duration: float,
    ) -> Dict[str, float]:
        """Compute polyphony-aware metrics."""
        frame_rate = 1.0 / self.config.frame_tolerance
        num_frames = int(audio_duration * frame_rate)
        
        # Compute polyphony at each frame
        pred_polyphony = self._compute_polyphony_per_frame(
            predicted_notes, num_frames, frame_rate
        )
        gt_polyphony = self._compute_polyphony_per_frame(
            ground_truth_notes, num_frames, frame_rate
        )
        
        # Polyphony accuracy
        polyphony_accuracy = accuracy_score(gt_polyphony, pred_polyphony)
        
        # Polyphony MSE
        polyphony_mse = np.mean((pred_polyphony - gt_polyphony) ** 2)
        
        # Polyphony correlation
        polyphony_corr = np.corrcoef(pred_polyphony, gt_polyphony)[0, 1]
        if np.isnan(polyphony_corr):
            polyphony_corr = 0.0
        
        return {
            "polyphony_accuracy": polyphony_accuracy,
            "polyphony_mse": polyphony_mse,
            "polyphony_correlation": polyphony_corr,
        }
    
    def _compute_polyphony_per_frame(
        self,
        notes: List[NoteEvent],
        num_frames: int,
        frame_rate: float,
    ) -> np.ndarray:
        """Compute polyphony (number of simultaneous notes) per frame."""
        polyphony = np.zeros(num_frames, dtype=int)
        
        for note in notes:
            start_frame = int(note.start_time * frame_rate)
            end_frame = int(note.end_time * frame_rate)
            
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(0, min(end_frame, num_frames - 1))
            
            polyphony[start_frame:end_frame + 1] += 1
        
        return polyphony
    
    def _compute_timing_metrics(
        self,
        predicted_notes: List[NoteEvent],
        ground_truth_notes: List[NoteEvent],
    ) -> Dict[str, float]:
        """Compute timing accuracy metrics."""
        if not predicted_notes or not ground_truth_notes:
            return {
                "onset_error_mean": 0.0,
                "onset_error_std": 0.0,
                "offset_error_mean": 0.0,
                "offset_error_std": 0.0,
                "duration_error_mean": 0.0,
                "duration_error_std": 0.0,
            }
        
        # Find matching notes
        cost_matrix = self._create_cost_matrix(predicted_notes, ground_truth_notes)
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        onset_errors = []
        offset_errors = []
        duration_errors = []
        
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            if cost_matrix[pred_idx, gt_idx] < 1.0:
                pred_note = predicted_notes[pred_idx]
                gt_note = ground_truth_notes[gt_idx]
                
                onset_errors.append(abs(pred_note.start_time - gt_note.start_time))
                offset_errors.append(abs(pred_note.end_time - gt_note.end_time))
                duration_errors.append(abs(pred_note.duration - gt_note.duration))
        
        return {
            "onset_error_mean": np.mean(onset_errors) if onset_errors else 0.0,
            "onset_error_std": np.std(onset_errors) if onset_errors else 0.0,
            "offset_error_mean": np.mean(offset_errors) if offset_errors else 0.0,
            "offset_error_std": np.std(offset_errors) if offset_errors else 0.0,
            "duration_error_mean": np.mean(duration_errors) if duration_errors else 0.0,
            "duration_error_std": np.std(duration_errors) if duration_errors else 0.0,
        }


class ModelEvaluator:
    """
    Model evaluator for Audio-to-MIDI conversion.
    
    This class provides methods to evaluate models on datasets
    and compute comprehensive metrics.
    """
    
    def __init__(self, metrics: AudioToMIDIMetrics):
        """Initialize evaluator."""
        self.metrics = metrics
        
    def evaluate_model(
        self,
        model,
        dataloader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation
            device: Device to run evaluation on
            
        Returns:
            Dictionary containing average metrics
        """
        model.eval()
        all_metrics = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                audio_features = batch["audio_features"].to(device)
                
                # Get model predictions
                if hasattr(model, 'generate'):
                    predictions = model.generate(audio_features)
                else:
                    outputs = model(audio_features)
                    predictions = self._decode_outputs(outputs)
                
                # Convert predictions to NoteEvent objects
                predicted_notes = self._predictions_to_notes(predictions, batch)
                
                # Get ground truth notes
                ground_truth_notes = self._batch_to_notes(batch)
                
                # Compute metrics for this batch
                batch_metrics = []
                for pred_notes, gt_notes in zip(predicted_notes, ground_truth_notes):
                    # Estimate audio duration from features
                    audio_duration = audio_features.shape[1] * 512 / 22050  # Approximate
                    
                    metrics = self.metrics.evaluate(pred_notes, gt_notes, audio_duration)
                    batch_metrics.append(metrics)
                
                all_metrics.extend(batch_metrics)
        
        # Average metrics across all samples
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def _decode_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decode model outputs to predictions."""
        # This is a placeholder - implement based on your model's output format
        return {
            "events": torch.argmax(outputs["event_logits"], dim=-1),
            "pitches": torch.argmax(outputs["pitch_logits"], dim=-1),
            "velocities": torch.argmax(outputs["velocity_logits"], dim=-1),
        }
    
    def _predictions_to_notes(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> List[List[NoteEvent]]:
        """Convert model predictions to NoteEvent objects."""
        # This is a placeholder - implement based on your model's output format
        batch_notes = []
        
        for i in range(predictions["events"].shape[0]):
            notes = []
            # Implement conversion logic here
            batch_notes.append(notes)
        
        return batch_notes
    
    def _batch_to_notes(self, batch: Dict[str, torch.Tensor]) -> List[List[NoteEvent]]:
        """Convert batch targets to NoteEvent objects."""
        batch_notes = []
        
        for i in range(batch["midi_events"].shape[0]):
            notes = []
            # Implement conversion logic here
            batch_notes.append(notes)
        
        return batch_notes
