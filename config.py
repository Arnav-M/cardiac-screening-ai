"""
Configuration Helper for Cardiac Article Screening System
Handles environment variables and model selection
"""

import os
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded environment variables from {env_path}")
except (ImportError, Exception) as e:
    logger.warning(f"Could not load .env file: {e}")

class Config:
    """Configuration class for the screening system"""
    
    # API Keys
    GROQ_API_KEY: Optional[str] = os.getenv('GROQ_API_KEY')
    HUGGINGFACE_TOKEN: Optional[str] = os.getenv('HUGGINGFACE_TOKEN')
    
    # Rayyan Credentials
    RAYYAN_EMAIL: Optional[str] = os.getenv('RAYYAN_EMAIL')
    RAYYAN_PASSWORD: Optional[str] = os.getenv('RAYYAN_PASSWORD')
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    # Screening Configuration
    MAX_ARTICLES: int = int(os.getenv('MAX_ARTICLES', '1000'))
    CONFIDENCE_THRESHOLD: float = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
    
    # Available Models Configuration
    MODELS = {
        'groq': {
            'name': 'llama-3.3-70b-versatile',
            'provider': 'groq',
            'requires_api_key': True,
            'quality': 'best',
            'speed': 'fast',
            'cost': 'free_tier'
        },
        'bioclinicalbert': {
            'name': 'emilyalsentzer/Bio_ClinicalBERT',
            'provider': 'local',
            'requires_api_key': False,
            'quality': 'excellent',
            'speed': 'fast',
            'cost': 'free'
        },
        'gpt_oss': {
            'name': 'gpt-oss:20b',
            'provider': 'ollama',
            'requires_api_key': False,
            'quality': 'good',
            'speed': 'medium',
            'cost': 'free'
        }
    }
    
    @classmethod
    def check_api_key(cls, provider: str = 'groq') -> bool:
        """Check if API key is available for the specified provider"""
        if provider == 'groq':
            if cls.GROQ_API_KEY and cls.GROQ_API_KEY != 'your_groq_api_key_here':
                logger.info("✓ Groq API key found")
                return True
            else:
                logger.warning("✗ Groq API key not found or invalid")
                return False
        elif provider == 'huggingface':
            if cls.HUGGINGFACE_TOKEN and cls.HUGGINGFACE_TOKEN != 'your_huggingface_token_here':
                logger.info("✓ Hugging Face token found")
                return True
            else:
                logger.warning("✗ Hugging Face token not found or invalid")
                return False
        return True
    
    @classmethod
    def get_recommended_model(cls) -> Dict:
        """Get recommended model based on available API keys and system"""
        # Try Groq first (best quality)
        if cls.check_api_key('groq'):
            logger.info("Recommended: Groq API (best quality)")
            return cls.MODELS['groq']
        
        # Fallback to Bio-ClinicalBERT (free, excellent for medical)
        logger.info("Recommended: Bio-ClinicalBERT (free, medical specialist)")
        return cls.MODELS['bioclinicalbert']
    
    @classmethod
    def print_configuration(cls):
        """Print current configuration"""
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        print(f"Groq API Key:        {'✓ Configured' if cls.check_api_key('groq') else '✗ Not configured'}")
        print(f"Rayyan Credentials:  {'✓ Configured' if cls.RAYYAN_EMAIL else '✗ Not configured'}")
        print(f"Max Articles:        {cls.MAX_ARTICLES}")
        print(f"Confidence Threshold: {cls.CONFIDENCE_THRESHOLD}")
        print(f"Ollama URL:          {cls.OLLAMA_BASE_URL}")
        print("\nAvailable Models:")
        for key, model in cls.MODELS.items():
            status = "✓ Ready" if not model['requires_api_key'] or cls.check_api_key(model['provider']) else "✗ Needs API key"
            print(f"  - {key:20s} {status}")
        print("="*60 + "\n")
    
    @classmethod
    def setup_wizard(cls):
        """Interactive setup wizard for first-time users"""
        print("\n" + "="*60)
        print("WELCOME TO CARDIAC ARTICLE SCREENING SYSTEM")
        print("="*60)
        print("\nSetup Wizard - Let's configure your system!\n")
        
        # Check if .env exists
        env_path = Path(__file__).parent / '.env'
        if not env_path.exists():
            print("ℹ️  No .env file found. Let's create one!")
            print("\n1. Get a FREE Groq API key (recommended for best quality):")
            print("   Visit: https://console.groq.com/")
            print("   Sign up and copy your API key")
            
            groq_key = input("\nPaste your Groq API key (or press Enter to skip): ").strip()
            
            # Create .env file
            with open(env_path, 'w') as f:
                f.write("# Cardiac Article Screening System Configuration\n\n")
                if groq_key:
                    f.write(f"GROQ_API_KEY={groq_key}\n\n")
                    print("✓ Groq API key saved!")
                else:
                    f.write("GROQ_API_KEY=your_groq_api_key_here\n\n")
                    print("✓ .env file created. You can add your API key later.")
                
                f.write("# Optional: Rayyan credentials\n")
                f.write("RAYYAN_EMAIL=your_email@example.com\n")
                f.write("RAYYAN_PASSWORD=your_password_here\n")
            
            print(f"\n✓ Configuration file created: {env_path}")
        else:
            print(f"✓ Configuration file found: {env_path}")
        
        # Show current configuration
        print("\nCurrent Configuration:")
        cls.print_configuration()
        
        # Recommend next steps
        print("\n📚 NEXT STEPS:")
        print("="*60)
        
        if cls.check_api_key('groq'):
            print("✓ You're all set! Use the Groq API for best results:")
            print("  python run_groq_rayyan.py --max-articles 50")
        else:
            print("1. Get a Groq API key (recommended):")
            print("   https://console.groq.com/")
            print("   Then add it to your .env file")
            print("\n2. Or use the free Bio-ClinicalBERT model:")
            print("   python run_pure_llm_rayyan.py --max-articles 50")
        
        print("\n📖 Read README.md for detailed instructions")
        print("="*60 + "\n")

def main():
    """Run configuration helper"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration Helper')
    parser.add_argument('--setup', action='store_true', help='Run setup wizard')
    parser.add_argument('--check', action='store_true', help='Check configuration')
    parser.add_argument('--recommend', action='store_true', help='Get model recommendation')
    
    args = parser.parse_args()
    
    if args.setup:
        Config.setup_wizard()
    elif args.check:
        Config.print_configuration()
    elif args.recommend:
        model = Config.get_recommended_model()
        print(f"\nRecommended Model: {model['name']}")
        print(f"Provider: {model['provider']}")
        print(f"Quality: {model['quality']}")
        print(f"Speed: {model['speed']}")
        print(f"Cost: {model['cost']}")
    else:
        # Default: show configuration
        Config.print_configuration()

if __name__ == "__main__":
    main()
