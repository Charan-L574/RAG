"""
Setup script for OmniDoc AI
Automates the initial setup process
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """Print welcome banner"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║          🌍 OmniDoc AI Setup Wizard                           ║
║                                                                ║
║  Multilingual Intelligent Document Conversational Assistant   ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")


def check_python_version():
    """Check if Python version is compatible"""
    print("📋 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. You have Python {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def create_virtual_environment():
    """Create a virtual environment"""
    print("\n📦 Creating virtual environment...")
    venv_path = Path('venv')
    
    if venv_path.exists():
        response = input("Virtual environment already exists. Recreate? (y/n): ")
        if response.lower() != 'y':
            print("Skipping virtual environment creation...")
            return True
    
    try:
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False


def install_dependencies():
    """Install required packages"""
    print("\n📥 Installing dependencies...")
    
    # Determine the correct pip path
    if sys.platform == "win32":
        pip_path = Path('venv/Scripts/pip.exe')
    else:
        pip_path = Path('venv/bin/pip')
    
    if not pip_path.exists():
        print("⚠️ Using system pip instead of venv pip")
        pip_cmd = 'pip'
    else:
        pip_cmd = str(pip_path)
    
    try:
        # Upgrade pip
        print("  Upgrading pip...")
        subprocess.run([pip_cmd, 'install', '--upgrade', 'pip'], check=True)
        
        # Install requirements
        print("  Installing packages (this may take a few minutes)...")
        subprocess.run([pip_cmd, 'install', '-r', 'requirements.txt'], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def setup_environment():
    """Set up .env file"""
    print("\n🔑 Setting up environment variables...")
    
    env_path = Path('.env')
    
    if env_path.exists():
        response = input(".env file already exists. Keep existing? (y/n): ")
        if response.lower() == 'y':
            print("Keeping existing .env file...")
            return True
    
    # Get API key from user
    print("\n🔐 Hugging Face API Key Setup")
    print("   Get your free API key from: https://huggingface.co/settings/tokens")
    api_key = input("\nPaste your Hugging Face API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("⚠️ Skipping API key setup. You'll need to add it manually to .env")
        api_key = "your_huggingface_api_key_here"
    
    # Create .env file
    env_content = f"""# Hugging Face API Configuration
HUGGINGFACE_API_KEY={api_key}

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_MODEL=tiiuae/falcon-7b-instruct
OCR_MODEL=microsoft/trocr-base-printed
CLASSIFICATION_MODEL=facebook/bart-large-mnli
LANGUAGE_DETECTION_MODEL=papluca/xlm-roberta-base-language-detection
TRANSLATION_MODEL=Helsinki-NLP/opus-mt-mul-en

# Application Settings
MAX_UPLOAD_SIZE_MB=50
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=3
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print("✅ Environment file created")
    return True


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ['uploads', 'temp', 'test_documents']
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
    
    print("✅ Directories created")
    return True


def run_diagnostics():
    """Run system diagnostics"""
    print("\n🔍 Running diagnostics...")
    
    try:
        from utils import run_diagnostics
        run_diagnostics()
        return True
    except Exception as e:
        print(f"⚠️ Could not run diagnostics: {e}")
        return False


def print_next_steps():
    """Print next steps for user"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║                    Setup Complete! 🎉                          ║
╚════════════════════════════════════════════════════════════════╝

📋 Next Steps:

1. Activate the virtual environment:
   Windows:   venv\\Scripts\\activate
   Linux/Mac: source venv/bin/activate

2. Make sure your Hugging Face API key is set in .env file

3. Run the application:
   python app.py

4. Open your browser:
   http://localhost:7860

═══════════════════════════════════════════════════════════════

📚 Additional Commands:

  python utils.py --diagnose        # Run system diagnostics
  python utils.py --test-api        # Test API connection
  python utils.py --examples        # View usage examples
  python utils.py --create-test-docs # Create sample documents

═══════════════════════════════════════════════════════════════

💡 Tips:
  - Check README.md for detailed documentation
  - Start with sample documents in test_documents/
  - Enable multilingual translation in the UI for cross-language queries

═══════════════════════════════════════════════════════════════

Need help? Check:
  - README.md for full documentation
  - .env.example for configuration options
  - GitHub issues for known problems

Happy document chatting! 🚀
""")


def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup aborted due to incompatible Python version")
        sys.exit(1)
    
    # Ask if user wants full setup
    print("\nThis wizard will:")
    print("  1. Create a virtual environment")
    print("  2. Install dependencies")
    print("  3. Set up environment variables")
    print("  4. Create necessary directories")
    
    response = input("\nProceed with setup? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        sys.exit(0)
    
    # Run setup steps
    steps = [
        create_virtual_environment,
        install_dependencies,
        setup_environment,
        create_directories
    ]
    
    for step in steps:
        if not step():
            print("\n❌ Setup failed. Please check the errors above.")
            sys.exit(1)
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
