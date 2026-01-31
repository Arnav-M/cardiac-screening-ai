# Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or use the automated setup script:

```bash
python setup.py
```

## Step 2: Get Your Free API Key

1. Visit [https://console.groq.com/](https://console.groq.com/)
2. Sign up for a free account
3. Copy your API key

## Step 3: Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```
GROQ_API_KEY=your_actual_api_key_here
```

## Step 4: Run Your First Screening

**Option A: Best Quality (Groq API)**
```bash
python run_groq_rayyan.py --max-articles 10
```

**Option B: Free Local Model (No API key needed)**
```bash
python run_pure_llm_rayyan.py --max-articles 10
```

## Step 5: Review Results

The system will:
1. Open Chrome browser
2. Navigate to Rayyan
3. Extract and screen articles
4. Automatically apply decisions (Include/Exclude/Maybe)
5. Show real-time progress

## What If Groq API Doesn't Work?

No problem! Use the free Bio-ClinicalBERT model:

```bash
python run_pure_llm_rayyan.py
```

This model is:
- Completely free
- Specialized for medical text
- Runs locally (no API needed)
- Very accurate for medical screening

## Common Issues

### "Groq API key not found"
- Check your `.env` file
- Make sure you copied the API key correctly
- Remove any extra spaces

### "ModuleNotFoundError: No module named 'dotenv'"
```bash
pip install python-dotenv
```

### "Selenium Chrome driver not found"
```bash
pip install selenium --upgrade
```

### Rate Limit Exceeded
- Wait a few minutes
- Reduce the number of articles: `--max-articles 50`
- Use the free Bio-ClinicalBERT model instead

## Need More Help?

- Read the full [README.md](README.md)
- Check [LLM_MODELS_USED.md](LLM_MODELS_USED.md) for model details
- Run `python config.py --check` to verify your configuration

---

**That's it! You're ready to start screening articles with AI!**
