#!/usr/bin/env python3
"""Script to run the Audio-to-MIDI demo application."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the Streamlit demo application."""
    demo_path = Path(__file__).parent / "demo" / "app.py"
    
    if not demo_path.exists():
        print(f"Error: Demo file not found at {demo_path}")
        sys.exit(1)
    
    print("Starting Audio-to-MIDI Conversion Demo...")
    print("=" * 50)
    print("This is a research and educational demonstration.")
    print("Please use responsibly and ethically.")
    print("=" * 50)
    
    try:
        # Run Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"Error running demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
