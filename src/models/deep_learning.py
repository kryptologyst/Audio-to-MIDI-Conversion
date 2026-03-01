"""Deep learning models for Audio-to-MIDI conversion."""

import warnings
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer


@dataclass
class DeepLearningConfig:
    """Configuration for deep learning Audio-to-MIDI models."""
    # Model architecture
    model_type: str = "transformer"  # "transformer", "cnn", "lstm"
    input_dim: int = 128  # Mel spectrogram dimensions
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # Audio processing
    sr: int = 22050
    hop_length: int = 512
    n_fft: int = 2048
    n_mels: int = 128
    
    # MIDI vocabulary
    max_pitch: int = 127
    max_velocity: int = 127
    num_events: int = 4  # note_on, note_off, time_shift, velocity_change
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 16
    max_length: int = 1000
    
    # Decoding
    beam_size: int = 1
    temperature: float = 1.0


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(0), :]


class AudioToMIDITransformer(nn.Module):
    """
    Transformer-based Audio-to-MIDI conversion model.
    
    This model uses a transformer encoder-decoder architecture to convert
    audio features to MIDI event sequences.
    """
    
    def __init__(self, config: DeepLearningConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # Output projection for MIDI events
        self.output_projection = nn.Linear(
            config.hidden_dim, 
            config.max_pitch + config.num_events
        )
        
        # Event type prediction
        self.event_classifier = nn.Linear(config.hidden_dim, config.num_events)
        
        # Velocity prediction
        self.velocity_predictor = nn.Linear(config.hidden_dim, config.max_velocity + 1)
        
    def forward(
        self, 
        audio_features: torch.Tensor,
        target_events: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            audio_features: Input audio features (batch_size, seq_len, input_dim)
            target_events: Target MIDI events for training (batch_size, seq_len)
            
        Returns:
            Dictionary containing model outputs
        """
        batch_size, seq_len, _ = audio_features.shape
        
        # Project input features
        x = self.input_projection(audio_features)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        
        # Transformer encoding
        encoded = self.encoder(x)
        
        # Predict MIDI events
        pitch_logits = self.output_projection(encoded)
        event_logits = self.event_classifier(encoded)
        velocity_logits = self.velocity_predictor(encoded)
        
        return {
            "pitch_logits": pitch_logits,
            "event_logits": event_logits,
            "velocity_logits": velocity_logits,
            "encoded_features": encoded,
        }
    
    def generate(
        self, 
        audio_features: torch.Tensor,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate MIDI events from audio features.
        
        Args:
            audio_features: Input audio features
            max_length: Maximum sequence length for generation
            
        Returns:
            Dictionary containing generated events
        """
        self.eval()
        max_length = max_length or self.config.max_length
        
        with torch.no_grad():
            # Get encoder output
            outputs = self.forward(audio_features)
            encoded = outputs["encoded_features"]
            
            # Generate events autoregressively
            batch_size = encoded.shape[0]
            device = encoded.device
            
            # Initialize with start token
            generated_events = []
            generated_pitches = []
            generated_velocities = []
            
            for i in range(max_length):
                # Get current context
                if i == 0:
                    context = encoded[:, 0:1, :]  # First frame
                else:
                    # Use previous predictions as context
                    context = encoded[:, i:i+1, :]
                
                # Predict next event
                event_logits = self.event_classifier(context)
                pitch_logits = self.output_projection(context)
                velocity_logits = self.velocity_predictor(context)
                
                # Sample from distributions
                event_probs = F.softmax(event_logits, dim=-1)
                pitch_probs = F.softmax(pitch_logits, dim=-1)
                velocity_probs = F.softmax(velocity_logits, dim=-1)
                
                event = torch.multinomial(event_probs.squeeze(1), 1)
                pitch = torch.multinomial(pitch_probs.squeeze(1), 1)
                velocity = torch.multinomial(velocity_probs.squeeze(1), 1)
                
                generated_events.append(event)
                generated_pitches.append(pitch)
                generated_velocities.append(velocity)
                
                # Stop if end token is generated
                if event.item() == 3:  # Assuming 3 is end token
                    break
            
            return {
                "events": torch.cat(generated_events, dim=1),
                "pitches": torch.cat(generated_pitches, dim=1),
                "velocities": torch.cat(generated_velocities, dim=1),
            }


class AudioToMIDICNN(nn.Module):
    """
    CNN-based Audio-to-MIDI conversion model.
    
    This model uses convolutional neural networks to process audio features
    and predict MIDI events.
    """
    
    def __init__(self, config: DeepLearningConfig):
        super().__init__()
        self.config = config
        
        # CNN layers for feature extraction
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        ])
        
        # Calculate flattened size
        self.flattened_size = 256
        
        # Output layers
        self.pitch_classifier = nn.Linear(self.flattened_size, config.max_pitch + 1)
        self.event_classifier = nn.Linear(self.flattened_size, config.num_events)
        self.velocity_classifier = nn.Linear(self.flattened_size, config.max_velocity + 1)
        
    def forward(self, audio_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the CNN model.
        
        Args:
            audio_features: Input audio features (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary containing model outputs
        """
        batch_size, seq_len, input_dim = audio_features.shape
        
        # Reshape for CNN (batch_size, 1, seq_len, input_dim)
        x = audio_features.unsqueeze(1)
        
        # Apply CNN layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Flatten
        x = x.view(batch_size, seq_len, self.flattened_size)
        
        # Predict outputs
        pitch_logits = self.pitch_classifier(x)
        event_logits = self.event_classifier(x)
        velocity_logits = self.velocity_classifier(x)
        
        return {
            "pitch_logits": pitch_logits,
            "event_logits": event_logits,
            "velocity_logits": velocity_logits,
        }


class AudioToMIDILSTM(nn.Module):
    """
    LSTM-based Audio-to-MIDI conversion model.
    
    This model uses LSTM networks to process sequential audio features
    and predict MIDI events.
    """
    
    def __init__(self, config: DeepLearningConfig):
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )
        
        # Output layers
        self.pitch_classifier = nn.Linear(config.hidden_dim * 2, config.max_pitch + 1)
        self.event_classifier = nn.Linear(config.hidden_dim * 2, config.num_events)
        self.velocity_classifier = nn.Linear(config.hidden_dim * 2, config.max_velocity + 1)
        
    def forward(self, audio_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the LSTM model.
        
        Args:
            audio_features: Input audio features (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary containing model outputs
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(audio_features)
        
        # Predict outputs
        pitch_logits = self.pitch_classifier(lstm_out)
        event_logits = self.event_classifier(lstm_out)
        velocity_logits = self.velocity_classifier(lstm_out)
        
        return {
            "pitch_logits": pitch_logits,
            "event_logits": event_logits,
            "velocity_logits": velocity_logits,
            "hidden_states": lstm_out,
        }


def create_model(config: DeepLearningConfig) -> nn.Module:
    """
    Create a model based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized model
    """
    if config.model_type == "transformer":
        return AudioToMIDITransformer(config)
    elif config.model_type == "cnn":
        return AudioToMIDICNN(config)
    elif config.model_type == "lstm":
        return AudioToMIDILSTM(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


class CTCHead(nn.Module):
    """
    CTC (Connectionist Temporal Classification) head for framewise MIDI prediction.
    
    This module predicts MIDI events for each time frame, useful for
    polyphonic music transcription.
    """
    
    def __init__(self, input_dim: int, vocab_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        
        # CTC head
        self.ctc_head = nn.Linear(input_dim, vocab_size)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CTC head.
        
        Args:
            features: Input features (batch_size, seq_len, input_dim)
            
        Returns:
            CTC logits (batch_size, seq_len, vocab_size)
        """
        return self.ctc_head(features)
    
    def compute_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CTC loss.
        
        Args:
            logits: Model logits (batch_size, seq_len, vocab_size)
            targets: Target sequences (batch_size, max_target_len)
            input_lengths: Input sequence lengths
            target_lengths: Target sequence lengths
            
        Returns:
            CTC loss
        """
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # (seq_len, batch_size, vocab_size)
        
        loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,  # Assuming 0 is blank token
            reduction='mean',
        )
        
        return loss
