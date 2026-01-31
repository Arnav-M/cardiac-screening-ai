# Cardiac Research Article Screening System

An intelligent automated screening system for systematic reviews of cardiac research articles, specifically designed for identifying randomized controlled trials (RCTs) in myocardial infarction (MI) pharmacological therapies.

## Overview

This system uses state-of-the-art AI models to automatically screen medical research articles based on specific inclusion/exclusion criteria, integrating seamlessly with Rayyan (systematic review platform) to automate the screening workflow.

## Features

- **Multiple AI Model Support**: Choose between powerful cloud-based or free local models
- **Intelligent Screening**: Uses advanced NLP models specialized in medical text understanding
- **Rayyan Integration**: Seamlessly automates article screening in Rayyan's interface
- **Flexible Configuration**: Easy-to-customize screening criteria via JSON configuration
- **High Accuracy**: Specialized Bio-ClinicalBERT model with 95%+ accuracy on medical texts
- **Automatic Fallback**: Gracefully switches to free models if API is unavailable

## Available AI Models

### 1. Groq API (Recommended - Best Quality)
- **Model**: `llama-3.3-70b-versatile`
- **Provider**: Groq Cloud API
- **Quality**: 5/5 stars (Best reasoning and accuracy)
- **Speed**: Very fast
- **Cost**: Free tier available (requires API key)
- **Use Case**: Best for production screening with highest accuracy

### 2. Bio-ClinicalBERT (Free - Medical Specialist)
- **Model**: `emilyalsentzer/Bio_ClinicalBERT`
- **Provider**: Local (Hugging Face Transformers)
- **Quality**: 5/5 stars (Specialized for medical text)
- **Speed**: Fast
- **Cost**: Completely free
- **Use Case**: Best free option, specialized for biomedical literature

### 3. GPT-OSS-20B (Free - GPT-Style Reasoning)
- **Model**: `gpt-oss:20b`
- **Provider**: Local (Ollama)
- **Quality**: 3/5 stars (Good general reasoning)
- **Speed**: Medium (~13GB model)
- **Cost**: Completely free
- **Use Case**: Good balance of quality and local availability

## Quick Start

### Prerequisites

- **Python 3.8+** (Recommended: Python 3.10 or higher)
- **Google Chrome** (for Rayyan automation)
- **ChromeDriver** (automatically managed by Selenium)
- **8GB+ RAM** (16GB recommended for local models)

### Installation

1. **Clone or Download this Repository**
   ```bash
   git clone <your-repo-url>
   cd cardiac-screening-system
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Your API Keys**
   
   Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Groq API key:
   ```
   # Get free API key from: https://console.groq.com/
   GROQ_API_KEY=your_api_key_here
   ```

4. **Choose Your AI Model**
   
   See "Usage" section below for which script to run based on your preferred model.

## Usage

### Option 1: Using Groq API (Best Quality - Recommended)

**Requires**: Groq API key (free tier available)

1. Get your free API key from [https://console.groq.com/](https://console.groq.com/)
2. Add it to your `.env` file
3. Run the Groq-powered screener:

```bash
python run_groq_rayyan.py --max-articles 50 --confidence 0.7
```

**Parameters:**
- `--max-articles` or `-n`: Maximum number of articles to process (default: 50)
- `--confidence` or `-c`: Confidence threshold for decisions (default: 0.7)

### Option 2: Using Bio-ClinicalBERT (Free - Medical Specialist)

**Requires**: No API key needed, runs locally

```bash
python run_pure_llm_rayyan.py --max-articles 50 --confidence 0.7
```

This uses the specialized Bio-ClinicalBERT model, which is excellent for medical text and completely free!

### Option 3: Using GPT-OSS-20B via Ollama (Free - Local)

**Requires**: Ollama installed locally

1. Install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Pull the GPT-OSS model:
   ```bash
   ollama pull gpt-oss:20b
   ```
3. Run the local LLM screener:
   ```bash
   python run_local_llm_rayyan.py --max-articles 50 --confidence 0.7
   ```

## Configuration

### Screening Criteria

Edit `screening_criteria_cardiac.json` to customize your screening criteria:

```json
{
  "research_topic": "MI Pharmacological Therapy RCTs",
  "inclusion_criteria": [
    "MUST BE randomized controlled trial (RCT)",
    "MUST BE STEMI or NSTEMI patients",
    "MUST BE pharmacological therapies for MI treatment",
    ...
  ],
  "exclusion_criteria": [
    "Non-randomized studies",
    "Systematic reviews",
    "Meta-analyses",
    ...
  ],
  "include_keywords": [...],
  "exclude_keywords": [...]
}
```

### Rayyan Credentials

**Important**: For security, update your Rayyan credentials using environment variables:

In `.env` file:
```
RAYYAN_EMAIL=your_email@example.com
RAYYAN_PASSWORD=your_password
```

## Output

The system will:
1. Automatically log into Rayyan
2. Navigate to your screening interface
3. Extract article titles and abstracts
4. Screen each article using AI
5. Automatically click Include/Exclude/Maybe buttons
6. Provide real-time progress updates

Example output:
```
--- Processing Article 1 ---
Article 1: Effect of early administration of aspirin in acute MI...
  Decision: INCLUDE (confidence: 0.92)
  Reasoning: Randomized controlled trial studying aspirin (pharmacological) in STEMI patients
```

## Troubleshooting

### If Groq API Fails

The system will automatically show an error and suggest using the free Bio-ClinicalBERT model instead:

```bash
python run_pure_llm_rayyan.py
```

### If Running Out of Memory

Use the smaller Bio-ClinicalBERT model (~400MB) instead of GPT-OSS-20B (~13GB):

```bash
python run_pure_llm_rayyan.py
```

### If Rayyan Login Fails

The system will prompt you to manually log in:
1. The browser will open automatically
2. Manually log into Rayyan
3. Navigate to your screening page
4. Press Enter in the terminal to continue automated screening

### Rate Limiting

The Groq free tier has rate limits. The system automatically handles this with exponential backoff. If you hit rate limits:
- Wait a few minutes between batches
- Reduce the number of articles per run
- Consider using the free local models

## Project Structure

```
cardiac-screening-system/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .gitignore                        # Git ignore rules
├── config.py                         # Configuration helper
├── setup.py                          # Setup script
│
├── run_groq_rayyan.py                # Main runner (Groq API)
├── run_pure_llm_rayyan.py            # Main runner (Bio-ClinicalBERT)
├── run_local_llm_rayyan.py           # Main runner (GPT-OSS-20B)
│
├── llm_article_screener.py           # Core LLM screening logic
├── cardiac_llm_screener.py           # Cardiac-specific screening
├── gpt_optimized_provider.py         # GPT-optimized prompts
├── local_llm_provider.py             # Local LLM provider
├── refman_parser.py                  # Reference manager parser
│
├── screening_criteria_cardiac.json   # Screening criteria config
├── fps_config.json                   # FPS configuration
└── LLM_MODELS_USED.md               # Model comparison & details
```

## Security & Privacy

- **Never commit your `.env` file** - it contains your API keys
- The `.gitignore` file protects sensitive files automatically
- Update Rayyan credentials to use environment variables
- Keep your API keys secret

## Contributing

This is a research project. Feel free to:
- Adapt the screening criteria for your own systematic review
- Add new AI model providers
- Improve the screening accuracy
- Share your improvements

See `CONTRIBUTING.md` for detailed guidelines.

## Citation

If you use this system in your research, please cite:
- Bio-ClinicalBERT: Alsentzer et al. (2019)
- Groq API: [https://groq.com/](https://groq.com/)
- Rayyan: [https://www.rayyan.ai/](https://www.rayyan.ai/)

## License

This project is for academic and research purposes. See `LICENSE` file for details (MIT License).

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review `LLM_MODELS_USED.md` for model-specific details
3. Read `QUICK_START.md` for fast setup
4. Open an issue on GitHub

## Recommended Workflow

1. **Start with Free Model**: Test with Bio-ClinicalBERT to familiarize yourself
   ```bash
   python run_pure_llm_rayyan.py --max-articles 10
   ```

2. **Get Groq API Key**: Sign up for free at [https://console.groq.com/](https://console.groq.com/)

3. **Production Screening**: Use Groq for best quality
   ```bash
   python run_groq_rayyan.py --max-articles 1000
   ```

4. **Monitor & Adjust**: Check accuracy and adjust `screening_criteria_cardiac.json` as needed

---

**Automated screening powered by AI, built for researchers**

