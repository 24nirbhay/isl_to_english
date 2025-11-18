"""
Setup script for English-to-Konkani Translation.
Beginner-friendly installation and verification.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60 + "\n")


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("⚠️  Python 3.8 or higher required")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def install_dependencies():
    """Install required packages."""
    print("\nInstalling dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def verify_imports():
    """Verify critical imports."""
    print("\nVerifying imports...")
    
    imports = [
        ("tensorflow", "TensorFlow"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn")
    ]
    
    all_ok = True
    
    for module, name in imports:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError:
            print(f"❌ {name} not found")
            all_ok = False
    
    return all_ok


def check_gpu():
    """Check for GPU availability."""
    print("\nChecking GPU availability...")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ {len(gpus)} GPU(s) detected:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("ℹ️  No GPU detected - will use CPU")
            print("   (Training will be slower but still works)")
    
    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")


def check_files():
    """Check if required files exist."""
    print("\nChecking project files...")
    
    # Current project files
    required_files = [
        "app.py",
        "konkani_pairs.txt",
        "requirements.txt",
        "src/__init__.py",
        "src/data_loader.py",
        "src/tokenizer.py",
        "src/model_architecture.py",
        "src/train.py",
        "src/translate.py"
    ]
    
    all_ok = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
            all_ok = False
    
    # Check for isl_to_english sibling directory (for pipeline integration)
    parent_dir = Path(__file__).parent.parent
    isl_path = parent_dir / "isl_to_english"
    
    print("\nChecking for ISL-to-English integration...")
    if isl_path.exists() and isl_path.is_dir():
        print(f"✅ ISL-to-English found at: {isl_path}")
        
        # Check key ISL files
        isl_required = [
            "src/real_time_translator.py",
            "models"
        ]
        
        for file in isl_required:
            file_path = isl_path / file
            if file_path.exists():
                print(f"   ✅ {file}")
            else:
                print(f"   ⚠️  {file} (not found - pipeline integration may not work)")
    else:
        print(f"ℹ️  ISL-to-English not found at: {isl_path}")
        print("   Pipeline integration will not be available")
        print("   (This is OK if you only want standalone translation)")
    
    return all_ok


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    dirs = ["models", "models/master"]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ {dir_path}/")


def test_data_loader():
    """Test loading konkani_pairs.txt."""
    print("\nTesting data loader...")
    
    try:
        from src.data_loader import load_translation_pairs
        
        pairs = load_translation_pairs('konkani_pairs.txt')
        
        if pairs:
            print(f"✅ Loaded {len(pairs)} translation pairs")
            print(f"   Sample: {pairs[0]['english']} → {pairs[0]['konkani']}")
        else:
            print("⚠️  No translation pairs found in konkani_pairs.txt")
            print("   Please add some pairs before training")
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    return True


def show_next_steps():
    """Show next steps."""
    print_header("SETUP COMPLETE!")
    
    print("📚 Quick Start Guide:")
    print()
    print("1. View your dataset:")
    print("   python app.py stats")
    print()
    print("2. Train the model (15-30 minutes on CPU):")
    print("   python app.py train")
    print()
    print("3. Translate a sentence:")
    print("   python app.py translate --text \"hello\"")
    print()
    print("4. Translate from file:")
    print("   python app.py translate --input-file input.txt --output-file output.txt")
    print()
    print("5. Monitor file for continuous translation:")
    print("   python app.py translate --monitor input.txt")
    print()
    
    # Check if pipeline integration is possible
    parent_dir = Path(__file__).parent.parent
    if (parent_dir / "isl_to_english").exists():
        print("🔗 Pipeline Integration Available:")
        print()
        print("   Run the full ISL → English → Konkani pipeline:")
        print(f"   cd {parent_dir}")
        print("   python pipeline.py live")
        print()
    
    print("📖 For detailed documentation:")
    print("   See README.md and QUICKSTART.md")
    print()
    print("❓ For help:")
    print("   python app.py --help")
    print()


def main():
    """Run setup."""
    print_header("ENGLISH-TO-KONKANI TRANSLATION SETUP")
    
    # Check Python version
    if not check_python_version():
        print("\n⚠️  Please upgrade Python to version 3.8 or higher")
        return
    
    # Check files
    if not check_files():
        print("\n⚠️  Some files are missing. Please check your installation.")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Dependency installation failed. Please check errors above.")
        return
    
    # Verify imports
    if not verify_imports():
        print("\n⚠️  Some imports failed. Please check errors above.")
        return
    
    # Check GPU
    check_gpu()
    
    # Create directories
    create_directories()
    
    # Test data loader
    test_data_loader()
    
    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    main()
