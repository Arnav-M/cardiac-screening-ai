#!/usr/bin/env python3
"""
Setup script for local LLM integration with Rayyan
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    packages = [
        "requests",
        "selenium",
        "python-dotenv"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
        else:
            print("❌ Ollama is not responding")
            return False
    except:
        print("❌ Ollama is not running")
        return False

def setup_ollama():
    """Setup instructions for Ollama"""
    print("\n" + "="*50)
    print("OLLAMA SETUP INSTRUCTIONS")
    print("="*50)
    print("1. Download Ollama from: https://ollama.ai/download")
    print("2. Install and start Ollama")
    print("3. Run one of these commands to download a model:")
    print("   - For 70B model (requires 16GB+ RAM): ollama pull llama3.3:70b")
    print("   - For 11B model (requires 8GB+ RAM): ollama pull llama3.2:11b")
    print("   - For 3B model (requires 4GB+ RAM): ollama pull llama3.2:3b")
    print("4. Verify with: ollama list")
    print("5. Run this script again to test connection")

def setup_huggingface():
    """Setup instructions for Hugging Face"""
    print("\n" + "="*50)
    print("HUGGING FACE SETUP INSTRUCTIONS")
    print("="*50)
    print("1. Install transformers and torch:")
    print("   pip install transformers torch accelerate")
    print("2. For GPU support, install CUDA version:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("3. Models will be downloaded automatically on first use")
    print("4. Recommended models:")
    print("   - meta-llama/Llama-3.2-11B-Instruct (11GB)")
    print("   - meta-llama/Llama-3.2-3B-Instruct (3GB)")

def test_setup():
    """Test the setup"""
    print("\n" + "="*50)
    print("TESTING SETUP")
    print("="*50)
    
    # Test Ollama
    if check_ollama():
        print("✅ Ollama setup complete")
        print("You can run: python run_local_llm_rayyan.py --use-ollama --model llama3.3:70b")
    else:
        print("❌ Ollama not ready")
        setup_ollama()
    
    # Test Hugging Face
    try:
        import transformers
        import torch
        print("✅ Hugging Face setup complete")
        print("You can run: python run_local_llm_rayyan.py --use-hf --model meta-llama/Llama-3.2-11B-Instruct")
    except ImportError:
        print("❌ Hugging Face not ready")
        setup_huggingface()

def main():
    """Main setup function"""
    print("LOCAL LLM RAYYAN INTEGRATION SETUP")
    print("="*50)
    
    # Install requirements
    install_requirements()
    
    # Test setup
    test_setup()
    
    print("\n" + "="*50)
    print("SETUP COMPLETE")
    print("="*50)
    print("Choose your preferred method:")
    print("1. Ollama (easier): python run_local_llm_rayyan.py --use-ollama")
    print("2. Hugging Face: python run_local_llm_rayyan.py --use-hf")
    print("\nFor help: python run_local_llm_rayyan.py --help")

if __name__ == "__main__":
    main()
