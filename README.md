# Audio-to-MIDI Conversion

Research-ready Audio-to-MIDI conversion system with both traditional signal processing and deep learning approaches. This project provides a comprehensive framework for converting audio recordings into MIDI format, suitable for music analysis, education, and research applications.

## PRIVACY DISCLAIMER

**This is a research and educational demonstration of Audio-to-MIDI conversion technology.**

### Important Notes:
- This tool is for research and educational purposes only
- No audio data is stored or transmitted to external servers
- All processing is performed locally on your device
- This technology should not be used for biometric identification or voice cloning in production systems
- Misuse of this technology for creating misleading or harmful content is strictly prohibited

### Intended Use:
- Music education and analysis
- Research in music information retrieval
- Creative applications in music production
- Academic research and development

By using this application, you agree to use it responsibly and in accordance with ethical guidelines.

## Features

### Models
- **Baseline Model**: Traditional signal processing approach using onset detection and pitch tracking
- **Improved Baseline**: Enhanced version with better pitch tracking and duration estimation
- **Deep Learning Models**: Transformer, CNN, and LSTM architectures for advanced conversion
- **CTC Head**: Connectionist Temporal Classification for framewise polyphonic prediction

### Audio Processing
- Comprehensive feature extraction (mel spectrograms, chroma, onset strength, etc.)
- Data augmentation (pitch shifting, time stretching, noise addition)
- Support for multiple audio formats (WAV, MP3, FLAC, M4A)
- Configurable preprocessing pipeline

### Evaluation Metrics
- Note-level precision, recall, and F1 scores
- Frame-level accuracy and polyphony metrics
- Timing accuracy (onset/offset errors)
- Comprehensive evaluation suite

### Interactive Demo
- Streamlit-based web interface
- Real-time audio visualization
- MIDI piano roll display
- Downloadable MIDI files
- Conversion metrics and statistics

## Installation

### Prerequisites
- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA support (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Audio-to-MIDI-Conversion.git
cd Audio-to-MIDI-Conversion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

### Development Setup

For development with additional tools:
```bash
pip install -e ".[dev]"
pre-commit install
```

## Quick Start

### Using the Demo Application

1. Start the Streamlit demo:
```bash
streamlit run demo/app.py
```

2. Open your browser to `http://localhost:8501`

3. Upload an audio file and click "Convert to MIDI"

### Using the Command Line

1. Train a model:
```bash
python -m src.train.trainer --config configs/default.yaml --data_dir data/ --output_dir outputs/
```

2. Evaluate a model:
```bash
python -m src.eval.evaluator --config configs/default.yaml --checkpoint outputs/best_model.pth --data_dir data/
```

### Using the Python API

```python
from src.models.baseline import BaselineAudioToMIDI, BaselineConfig
from src.utils.audio import load_audio

# Initialize model
config = BaselineConfig()
model = BaselineAudioToMIDI(config)

# Load audio
audio, sr = load_audio("path/to/audio.wav")

# Convert to MIDI
notes = model.predict(audio, sr)

# Save MIDI file
from src.utils.midi import create_midi_file
create_midi_file(notes, "output.mid")
```

## Dataset Schema

### Directory Structure
```
data/
├── raw/                    # Raw audio and MIDI files
│   ├── audio/             # Audio files (WAV, MP3, FLAC)
│   └── midi/              # Corresponding MIDI files
├── processed/              # Processed features and targets
└── meta.csv               # Metadata file (optional)
```

### Metadata Format
The optional `meta.csv` file should contain:
- `audio_path`: Path to audio file
- `midi_path`: Path to corresponding MIDI file
- `split`: Dataset split (train/val/test)
- `duration`: Audio duration in seconds
- `instrument`: Instrument type (optional)
- `genre`: Music genre (optional)

### Synthetic Dataset
If no real dataset is available, the system can generate synthetic audio-MIDI pairs for testing and demonstration purposes.

## Configuration

The system uses YAML configuration files. See `configs/default.yaml` for all available options:

### Model Configuration
- Model type selection (baseline, transformer, cnn, lstm)
- Architecture parameters
- Training hyperparameters

### Data Configuration
- Audio processing parameters
- Data augmentation settings
- MIDI processing options

### Training Configuration
- Learning rate, batch size, epochs
- Checkpointing and logging
- Early stopping

## Training

### Baseline Models
Baseline models don't require training and can be used immediately.

### Deep Learning Models
For deep learning models, you need to train them first:

```bash
python -m src.train.trainer \
    --config configs/default.yaml \
    --data_dir data/ \
    --output_dir outputs/ \
    --seed 42
```

### Training with Synthetic Data
For testing without real data:
```bash
python -m src.train.trainer \
    --config configs/synthetic.yaml \
    --data_dir data/ \
    --output_dir outputs/ \
    --use_synthetic
```

## Evaluation

### Metrics
The system provides comprehensive evaluation metrics:

- **Note-level**: Precision, recall, F1 score
- **Frame-level**: Accuracy, precision, recall
- **Polyphony**: Polyphony accuracy and correlation
- **Timing**: Onset/offset error statistics

### Running Evaluation
```bash
python -m src.eval.evaluator \
    --config configs/default.yaml \
    --checkpoint outputs/best_model.pth \
    --data_dir data/test/
```

## API Reference

### Core Classes

#### `BaselineAudioToMIDI`
Traditional signal processing approach for Audio-to-MIDI conversion.

```python
from src.models.baseline import BaselineAudioToMIDI, BaselineConfig

config = BaselineConfig()
model = BaselineAudioToMIDI(config)
notes = model.predict(audio, sr)
```

#### `AudioMIDIDataset`
Dataset class for loading audio-MIDI pairs.

```python
from src.data.dataset import AudioMIDIDataset, DataConfig

config = DataConfig()
dataset = AudioMIDIDataset("data/", config, split="train")
```

#### `AudioToMIDIMetrics`
Comprehensive evaluation metrics.

```python
from src.metrics.evaluation import AudioToMIDIMetrics

metrics = AudioToMIDIMetrics()
results = metrics.evaluate(predicted_notes, ground_truth_notes, duration)
```

## Project Structure

```
audio-to-midi/
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature extraction
│   ├── losses/            # Loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── decoding/          # Decoding algorithms
│   ├── train/             # Training scripts
│   ├── eval/              # Evaluation scripts
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── data/                  # Data directory
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── assets/                # Generated assets
├── demo/                  # Demo application
├── requirements.txt       # Dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to all functions and classes
- Run `black` and `ruff` for formatting

### Testing
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{audio_to_midi,
  title={Audio-to-MIDI Conversion},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Audio-to-MIDI-Conversion}
}
```

## Acknowledgments

- Librosa for audio processing
- PyTorch for deep learning framework
- Streamlit for the demo interface
- The music information retrieval community

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in configuration
2. **Audio loading errors**: Check file format and permissions
3. **MIDI conversion fails**: Verify audio contains musical content
4. **Demo won't start**: Ensure all dependencies are installed

### Getting Help

- Check the issues page for known problems
- Create a new issue with detailed error information
- Include system information and configuration

## Roadmap

- [ ] Real-time streaming conversion
- [ ] Multi-instrument separation
- [ ] Advanced deep learning architectures
- [ ] Web API for remote processing
- [ ] Mobile app integration
- [ ] Cloud deployment options

---

**Remember**: This tool is for research and educational purposes only. Use responsibly and ethically.
# Audio-to-MIDI-Conversion
