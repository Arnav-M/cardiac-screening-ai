# Deployment & GitHub Setup Guide

This guide will help you prepare and deploy this project to GitHub so anyone can use it.

## Pre-Deployment Checklist

Before uploading to GitHub, ensure:

- [x] `.env` file is in `.gitignore` (prevents exposing API keys)
- [x] `.gitignore` includes all sensitive files
- [x] `README.md` is complete and user-friendly
- [x] `.env.example` provides a template for users
- [x] No hardcoded API keys in the code
- [x] No personal credentials committed
- [x] `requirements.txt` lists all dependencies
- [x] `LICENSE` file is included

## Initializing Git Repository

If you haven't already initialized a git repository:

```bash
cd cardiac-screening-system
git init
```

## Reviewing Files Before Commit

Check what will be committed:

```bash
git status
```

Make sure `.env` and sensitive files are NOT listed (should be ignored by `.gitignore`).

## Creating Your First Commit

```bash
# Add all files (sensitive files are automatically excluded by .gitignore)
git add .

# Create your first commit
git commit -m "Initial commit: Cardiac Article Screening System"
```

## Creating a GitHub Repository

1. Go to [https://github.com/new](https://github.com/new)
2. Repository name: `cardiac-screening-system` (or your preferred name)
3. Description: "AI-powered automated screening system for cardiac research systematic reviews"
4. Choose "Public" (recommended) or "Private"
5. DO NOT initialize with README (you already have one)
6. Click "Create repository"

## Pushing to GitHub

GitHub will show you commands. Use these:

```bash
# Add GitHub as remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push your code
git branch -M main
git push -u origin main
```

## Verifying Security

After pushing, visit your GitHub repository and verify:

1. `.env` file is NOT visible (should be blocked by .gitignore)
2. No API keys visible in any files
3. `.env.example` is visible (this is good - it's a template)
4. `README.md` displays correctly

## Creating a Release

Once your code is on GitHub, create a release:

1. Go to your repository on GitHub
2. Click "Releases" → "Create a new release"
3. Tag version: `v1.0.0`
4. Release title: "Cardiac Article Screening System v1.0.0"
5. Description: Brief summary of features
6. Click "Publish release"

## Repository Settings (Recommended)

1. Go to Settings → General
2. Add topics: `ai`, `medical-research`, `systematic-review`, `article-screening`, `nlp`
3. Enable "Issues" for user support
4. Add a description and website link

## Creating Professional README Badges

Add these badges to your `README.md` (replace `YOUR_USERNAME` and `YOUR_REPO`):

```markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
```

## Installation Instructions for Users

Users should follow these steps to use your project:

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your API keys

# 4. Run the setup wizard
python config.py --setup

# 5. Start screening
python run_groq_rayyan.py --max-articles 50
```

## Protecting Sensitive Information

If you accidentally committed sensitive information:

```bash
# Remove file from git history (but keep local file)
git rm --cached .env

# Update .gitignore if not already there
echo ".env" >> .gitignore

# Commit the changes
git add .gitignore
git commit -m "Remove .env from tracking"

# Force push (WARNING: Only if repository is new and no one else is using it)
git push -f origin main
```

**IMPORTANT**: If you exposed an API key, REVOKE it immediately and get a new one:
- Groq: [https://console.groq.com/keys](https://console.groq.com/keys)

## Continuous Updates

When you make changes:

```bash
# Check what changed
git status

# Add changes
git add .

# Commit with descriptive message
git commit -m "Add feature: XYZ"

# Push to GitHub
git push origin main
```

## Documentation Best Practices

Keep these files updated:
- `README.md` - Main documentation
- `CHANGELOG.md` - Record of changes
- `QUICK_START.md` - Fast getting started guide
- `LLM_MODELS_USED.md` - Model comparison and details

## Support & Issues

Enable GitHub Issues so users can:
- Report bugs
- Request features
- Ask questions

Respond promptly to maintain a healthy open-source project!

## License

Your project uses the MIT License, which allows users to:
- Use commercially
- Modify
- Distribute
- Use privately

But requires:
- License and copyright notice
- No liability/warranty

## Star Your Project

Ask users to star your repository if they find it useful! This helps others discover your work.

---

**Congratulations! Your project is now ready for the world!**

Share it with:
- Research communities
- Medical AI forums
- Systematic review researchers
- Open source communities

Good luck with your project!
