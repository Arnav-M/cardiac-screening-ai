"""
Setup script for easy installation
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print("Current version:", sys.version)
        return False
    
    print("✓ Python version is compatible")
    return True

def install_requirements():
    """Install required packages"""
    print_header("Installing Dependencies")
    
    try:
        print("Installing packages from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n✓ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing dependencies: {e}")
        print("\nTry installing manually:")
        print("  pip install -r requirements.txt")
        return False

def setup_env_file():
    """Setup environment file"""
    print_header("Setting Up Environment File")
    
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        print("✓ .env file already exists")
        response = input("Do you want to reconfigure it? (y/N): ").strip().lower()
        if response != 'y':
            return True
    
    if not env_example_path.exists():
        print("❌ .env.example not found")
        return False
    
    # Copy .env.example to .env
    with open(env_example_path, 'r') as f:
        content = f.read()
    
    print("\n📝 Let's configure your environment variables:")
    print("\n1. Groq API Key (Recommended - Best Quality)")
    print("   Get free API key from: https://console.groq.com/")
    groq_key = input("   Enter your Groq API key (or press Enter to skip): ").strip()
    
    if groq_key:
        content = content.replace("your_groq_api_key_here", groq_key)
        print("   ✓ Groq API key configured")
    else:
        print("   ⚠️  Skipped - You can add it later in .env file")
    
    print("\n2. Rayyan Credentials (Optional - for automatic login)")
    setup_rayyan = input("   Configure Rayyan credentials? (y/N): ").strip().lower()
    
    if setup_rayyan == 'y':
        rayyan_email = input("   Rayyan email: ").strip()
        rayyan_password = input("   Rayyan password: ").strip()
        
        if rayyan_email and rayyan_password:
            content = content.replace("your_email@example.com", rayyan_email)
            content = content.replace("your_password_here", rayyan_password)
            print("   ✓ Rayyan credentials configured")
    
    # Write .env file
    with open(env_path, 'w') as f:
        f.write(content)
    
    print(f"\n✓ Environment file created: {env_path}")
    return True

def check_ollama():
    """Check if Ollama is installed (optional)"""
    print_header("Checking Ollama (Optional)")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✓ Ollama is installed and running")
            return True
    except:
        pass
    
    print("⚠️  Ollama not detected (optional)")
    print("   Install from: https://ollama.ai/")
    print("   Required for local GPT-OSS model")
    return False

def print_next_steps():
    """Print next steps"""
    print_header("Setup Complete! 🎉")
    
    print("Next Steps:")
    print("\n1. Configure your API key (if you haven't already):")
    print("   - Edit .env file and add your Groq API key")
    print("   - Get free key from: https://console.groq.com/")
    
    print("\n2. Run the configuration helper:")
    print("   python config.py --setup")
    
    print("\n3. Start screening with the best model:")
    print("   python run_groq_rayyan.py --max-articles 50")
    
    print("\n   Or use the free Bio-ClinicalBERT model:")
    print("   python run_pure_llm_rayyan.py --max-articles 50")
    
    print("\n4. Read the documentation:")
    print("   Open README.md for detailed instructions")
    
    print("\n" + "="*60)
    print("For support, check README.md or open an issue on GitHub")
    print("="*60 + "\n")

def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("CARDIAC ARTICLE SCREENING SYSTEM - SETUP")
    print("="*60)
    print("\nThis script will help you set up your environment.")
    print("Press Ctrl+C at any time to cancel.\n")
    
    try:
        # Step 1: Check Python version
        if not check_python_version():
            sys.exit(1)
        
        # Step 2: Install requirements
        if not install_requirements():
            print("\n⚠️  Some dependencies failed to install")
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response != 'y':
                sys.exit(1)
        
        # Step 3: Setup .env file
        if not setup_env_file():
            print("\n⚠️  Environment setup incomplete")
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response != 'y':
                sys.exit(1)
        
        # Step 4: Check Ollama (optional)
        check_ollama()
        
        # Step 5: Print next steps
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
