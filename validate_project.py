#!/usr/bin/env python3
"""Project summary and validation script."""

import sys
from pathlib import Path


def print_project_structure():
    """Print the project structure."""
    print("Audio-to-MIDI Conversion Project Structure:")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    
    def print_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        items = sorted(path.iterdir())
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and not item.name.startswith('.') and current_depth < max_depth - 1:
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, next_prefix, max_depth, current_depth + 1)
    
    print_tree(project_root)


def validate_installation():
    """Validate that the installation is working."""
    print("\nValidating Installation:")
    print("=" * 30)
    
    try:
        # Test basic imports
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from src.models.baseline import BaselineAudioToMIDI, BaselineConfig
        print("✓ Baseline model imports successful")
        
        from src.utils.device import get_device
        device = get_device()
        print(f"✓ Device detection: {device}")
        
        from src.data.dataset import SyntheticDataset, DataConfig
        print("✓ Dataset imports successful")
        
        from src.utils.midi import NoteEvent, hz_to_midi
        print("✓ MIDI utilities imports successful")
        
        from src.utils.audio import extract_features
        print("✓ Audio utilities imports successful")
        
        print("\n✓ All core modules imported successfully!")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True


def run_basic_test():
    """Run a basic functionality test."""
    print("\nRunning Basic Functionality Test:")
    print("=" * 35)
    
    try:
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from src.models.baseline import BaselineAudioToMIDI, BaselineConfig
        from src.data.dataset import SyntheticDataset, DataConfig
        from src.utils.device import set_seed
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Initialize model
        config = BaselineConfig()
        model = BaselineAudioToMIDI(config)
        print("✓ Model initialized")
        
        # Create synthetic dataset
        data_config = DataConfig()
        dataset = SyntheticDataset(data_config, num_samples=1)
        print("✓ Synthetic dataset created")
        
        # Test prediction
        sample = dataset[0]
        audio_features = sample["audio_features"]
        audio = audio_features.numpy().T
        sr = 22050
        
        notes = model.predict(audio, sr)
        print(f"✓ Generated {len(notes)} MIDI notes")
        
        if notes:
            note = notes[0]
            print(f"  - First note: pitch={note.pitch}, velocity={note.velocity}, duration={note.duration:.2f}s")
        
        print("\n✓ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_usage_instructions():
    """Print usage instructions."""
    print("\nUsage Instructions:")
    print("=" * 20)
    print("1. Run the demo application:")
    print("   python run_demo.py")
    print("   or")
    print("   streamlit run demo/app.py")
    print()
    print("2. Run basic tests:")
    print("   python test_basic.py")
    print()
    print("3. Run unit tests:")
    print("   pytest tests/")
    print()
    print("4. Train a model:")
    print("   python -m src.train.trainer --config configs/default.yaml --data_dir data/ --output_dir outputs/")
    print()
    print("5. Use in Python code:")
    print("   from src.models.baseline import BaselineAudioToMIDI")
    print("   model = BaselineAudioToMIDI()")
    print("   notes = model.predict(audio, sr)")


def print_privacy_reminder():
    """Print privacy reminder."""
    print("\nPrivacy Reminder:")
    print("=" * 15)
    print("This is a research and educational demonstration.")
    print("Please use responsibly and ethically.")
    print("- No audio data is stored or transmitted")
    print("- All processing is performed locally")
    print("- Not for biometric identification or voice cloning")
    print("- Misuse for harmful content is prohibited")


def main():
    """Main function."""
    print("Audio-to-MIDI Conversion Project")
    print("Research & Educational Demonstration")
    print("=" * 50)
    
    # Print project structure
    print_project_structure()
    
    # Validate installation
    if not validate_installation():
        print("\n❌ Installation validation failed!")
        sys.exit(1)
    
    # Run basic test
    if not run_basic_test():
        print("\n❌ Basic functionality test failed!")
        sys.exit(1)
    
    # Print instructions
    print_usage_instructions()
    print_privacy_reminder()
    
    print("\n✅ Project setup completed successfully!")
    print("Ready for research and educational use.")


if __name__ == "__main__":
    main()
