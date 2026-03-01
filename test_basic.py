"""Simple test script for Audio-to-MIDI conversion."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.baseline import BaselineAudioToMIDI, BaselineConfig
from src.data.dataset import SyntheticDataset, DataConfig
from src.utils.device import get_device, set_seed


def test_baseline_model():
    """Test the baseline Audio-to-MIDI model."""
    print("Testing baseline Audio-to-MIDI model...")
    
    # Set random seed
    set_seed(42)
    
    # Initialize model
    config = BaselineConfig()
    model = BaselineAudioToMIDI(config)
    
    # Create synthetic dataset
    data_config = DataConfig()
    dataset = SyntheticDataset(data_config, num_samples=5)
    
    # Test on a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        audio_features = sample["audio_features"]
        
        # Convert to numpy for baseline model
        audio = audio_features.numpy().T  # Transpose to (time, features)
        sr = 22050
        
        # Predict MIDI notes
        notes = model.predict(audio, sr)
        
        print(f"Sample {i}: Generated {len(notes)} notes")
        if notes:
            print(f"  First note: pitch={notes[0].pitch}, velocity={notes[0].velocity}, duration={notes[0].duration:.2f}s")
    
    print("Baseline model test completed successfully!")


def test_device_info():
    """Test device information."""
    print("\nTesting device information...")
    device = get_device()
    print(f"Using device: {device}")
    
    # Test device info
    from src.utils.device import get_device_info
    info = get_device_info()
    print(f"Device info: {info}")


def test_synthetic_dataset():
    """Test synthetic dataset generation."""
    print("\nTesting synthetic dataset...")
    
    data_config = DataConfig()
    dataset = SyntheticDataset(data_config, num_samples=3)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Audio features shape: {sample['audio_features'].shape}")
    print(f"MIDI events shape: {sample['midi_events'].shape}")
    
    print("Synthetic dataset test completed successfully!")


if __name__ == "__main__":
    print("Running Audio-to-MIDI conversion tests...")
    print("=" * 50)
    
    try:
        test_device_info()
        test_synthetic_dataset()
        test_baseline_model()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
