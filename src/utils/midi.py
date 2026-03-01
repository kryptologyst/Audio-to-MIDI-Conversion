"""MIDI processing utilities for Audio-to-MIDI conversion."""

import warnings
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
import mido
import pretty_midi
from mido import MidiFile, MidiTrack, Message


@dataclass
class NoteEvent:
    """Represents a MIDI note event."""
    pitch: int  # MIDI note number (0-127)
    velocity: int  # Note velocity (0-127)
    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    duration: float  # Duration in seconds


@dataclass
class MIDISequence:
    """Represents a sequence of MIDI events."""
    notes: List[NoteEvent]
    tempo: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)
    key_signature: str = "C"


def hz_to_midi(frequency: float) -> float:
    """Convert frequency in Hz to MIDI note number."""
    if frequency <= 0:
        return 0.0
    return 12 * np.log2(frequency / 440.0) + 69.0


def midi_to_hz(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))


def quantize_time(time: float, resolution: float = 0.125) -> float:
    """Quantize time to nearest resolution (default: 1/8 note)."""
    return round(time / resolution) * resolution


def create_midi_file(
    notes: List[NoteEvent],
    output_path: str,
    tempo: float = 120.0,
    time_signature: Tuple[int, int] = (4, 4),
) -> None:
    """
    Create a MIDI file from note events.
    
    Args:
        notes: List of NoteEvent objects
        output_path: Path to save MIDI file
        tempo: Tempo in BPM
        time_signature: Time signature as (numerator, denominator)
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Set tempo
    track.append(Message('set_tempo', tempo=mido.bpm2tempo(tempo)))
    
    # Set time signature
    track.append(Message('time_signature', 
                        numerator=time_signature[0], 
                        denominator=time_signature[1]))
    
    # Sort notes by start time
    sorted_notes = sorted(notes, key=lambda x: x.start_time)
    
    # Convert to MIDI ticks
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_second = ticks_per_beat * tempo / 60.0
    
    current_time = 0
    
    for note in sorted_notes:
        # Note on
        note_on_time = int(note.start_time * ticks_per_second)
        delta_time = note_on_time - current_time
        track.append(Message('note_on', 
                            note=note.pitch, 
                            velocity=note.velocity, 
                            time=delta_time))
        current_time = note_on_time
        
        # Note off
        note_off_time = int(note.end_time * ticks_per_second)
        delta_time = note_off_time - current_time
        track.append(Message('note_off', 
                            note=note.pitch, 
                            velocity=note.velocity, 
                            time=delta_time))
        current_time = note_off_time
    
    mid.save(output_path)


def load_midi_file(file_path: str) -> MIDISequence:
    """
    Load a MIDI file and extract note events.
    
    Args:
        file_path: Path to MIDI file
        
    Returns:
        MIDISequence object containing note events
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        notes = []
        
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                note_event = NoteEvent(
                    pitch=note.pitch,
                    velocity=note.velocity,
                    start_time=note.start,
                    end_time=note.end,
                    duration=note.end - note.start,
                )
                notes.append(note_event)
        
        # Get tempo (use first tempo change or default)
        tempo = 120.0
        if midi_data.get_tempo_changes()[1]:
            tempo = midi_data.get_tempo_changes()[1][0]
        
        return MIDISequence(
            notes=notes,
            tempo=tempo,
            time_signature=(4, 4),  # Default
            key_signature="C",  # Default
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to load MIDI file {file_path}: {e}")


def merge_overlapping_notes(notes: List[NoteEvent], tolerance: float = 0.05) -> List[NoteEvent]:
    """
    Merge overlapping notes of the same pitch.
    
    Args:
        notes: List of NoteEvent objects
        tolerance: Time tolerance for merging in seconds
        
    Returns:
        List of merged NoteEvent objects
    """
    if not notes:
        return []
    
    # Sort by pitch and start time
    sorted_notes = sorted(notes, key=lambda x: (x.pitch, x.start_time))
    merged_notes = []
    
    current_note = sorted_notes[0]
    
    for note in sorted_notes[1:]:
        if (note.pitch == current_note.pitch and 
            note.start_time <= current_note.end_time + tolerance):
            # Merge notes
            current_note.end_time = max(current_note.end_time, note.end_time)
            current_note.duration = current_note.end_time - current_note.start_time
        else:
            # Add current note and start new one
            merged_notes.append(current_note)
            current_note = note
    
    merged_notes.append(current_note)
    return merged_notes


def filter_notes_by_velocity(notes: List[NoteEvent], min_velocity: int = 20) -> List[NoteEvent]:
    """Filter notes by minimum velocity threshold."""
    return [note for note in notes if note.velocity >= min_velocity]


def filter_notes_by_duration(notes: List[NoteEvent], min_duration: float = 0.05) -> List[NoteEvent]:
    """Filter notes by minimum duration threshold."""
    return [note for note in notes if note.duration >= min_duration]


def quantize_notes(notes: List[NoteEvent], resolution: float = 0.125) -> List[NoteEvent]:
    """
    Quantize note timing to grid.
    
    Args:
        notes: List of NoteEvent objects
        resolution: Quantization resolution in seconds
        
    Returns:
        List of quantized NoteEvent objects
    """
    quantized_notes = []
    
    for note in notes:
        quantized_note = NoteEvent(
            pitch=note.pitch,
            velocity=note.velocity,
            start_time=quantize_time(note.start_time, resolution),
            end_time=quantize_time(note.end_time, resolution),
            duration=quantize_time(note.duration, resolution),
        )
        quantized_notes.append(quantized_note)
    
    return quantized_notes


def extract_chord_sequence(notes: List[NoteEvent], window_size: float = 0.5) -> List[List[int]]:
    """
    Extract chord sequence from note events.
    
    Args:
        notes: List of NoteEvent objects
        window_size: Window size for chord extraction in seconds
        
    Returns:
        List of chord lists (each chord is a list of MIDI note numbers)
    """
    if not notes:
        return []
    
    # Find time range
    start_time = min(note.start_time for note in notes)
    end_time = max(note.end_time for note in notes)
    
    chords = []
    current_time = start_time
    
    while current_time < end_time:
        window_end = current_time + window_size
        
        # Find notes active in this window
        active_notes = []
        for note in notes:
            if (note.start_time < window_end and 
                note.end_time > current_time):
                active_notes.append(note.pitch)
        
        # Remove duplicates and sort
        chord = sorted(list(set(active_notes)))
        chords.append(chord)
        
        current_time += window_size
    
    return chords


def compute_polyphony_stats(notes: List[NoteEvent]) -> Dict[str, float]:
    """
    Compute polyphony statistics.
    
    Args:
        notes: List of NoteEvent objects
        
    Returns:
        Dictionary containing polyphony statistics
    """
    if not notes:
        return {"max_polyphony": 0, "avg_polyphony": 0, "polyphony_ratio": 0}
    
    # Create time series of active notes
    events = []
    for note in notes:
        events.append((note.start_time, 1))  # Note on
        events.append((note.end_time, -1))   # Note off
    
    events.sort(key=lambda x: x[0])
    
    # Compute polyphony over time
    polyphony_values = []
    current_polyphony = 0
    
    for time, delta in events:
        current_polyphony += delta
        polyphony_values.append(current_polyphony)
    
    return {
        "max_polyphony": max(polyphony_values) if polyphony_values else 0,
        "avg_polyphony": np.mean(polyphony_values) if polyphony_values else 0,
        "polyphony_ratio": np.mean([p > 1 for p in polyphony_values]) if polyphony_values else 0,
    }


def validate_midi_notes(notes: List[NoteEvent]) -> List[NoteEvent]:
    """
    Validate and clean MIDI notes.
    
    Args:
        notes: List of NoteEvent objects
        
    Returns:
        List of validated NoteEvent objects
    """
    valid_notes = []
    
    for note in notes:
        # Check pitch range
        if not (0 <= note.pitch <= 127):
            continue
        
        # Check velocity range
        if not (0 <= note.velocity <= 127):
            continue
        
        # Check timing validity
        if note.start_time < 0 or note.end_time <= note.start_time:
            continue
        
        # Check duration
        if note.duration <= 0:
            continue
        
        valid_notes.append(note)
    
    return valid_notes
