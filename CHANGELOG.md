# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-31

### Added
- Initial release of Cardiac Article Screening System
- Groq API integration with `llama-3.3-70b-versatile` model
- Bio-ClinicalBERT local model for free medical text screening
- GPT-OSS-20B local model via Ollama
- Automated Rayyan integration with Selenium
- Comprehensive screening criteria configuration (JSON)
- Environment variable management with `.env` support
- Configuration helper script (`config.py`)
- Interactive setup wizard
- Automatic fallback to free models if API fails
- Rate limiting with exponential backoff
- Real-time progress monitoring
- Detailed logging system

### Documentation
- Complete README.md with installation and usage guide
- QUICK_START.md for fast onboarding
- DEPLOYMENT.md for GitHub setup
- LLM_MODELS_USED.md for model comparisons
- Comprehensive code comments

### Security
- `.gitignore` to protect sensitive files
- `.env.example` template for safe credential storage
- Environment variable support for API keys
- Removed hardcoded credentials

### Configuration Files
- `requirements.txt` with all dependencies
- `.env.example` for easy setup
- `screening_criteria_cardiac.json` for customizable criteria
- `LICENSE` (MIT License)

## [Future] - Planned Features

### Planned Additions
- [ ] Support for additional AI models (Claude, GPT-4, etc.)
- [ ] Batch processing for large article sets
- [ ] Web interface for easier use
- [ ] Export results in multiple formats (CSV, Excel, JSON)
- [ ] Integration with other systematic review tools (Covidence, DistillerSR)
- [ ] Machine learning model training on user feedback
- [ ] Multi-language support
- [ ] Docker containerization
- [ ] REST API for programmatic access
- [ ] Progress saving and resume capability
- [ ] Advanced analytics and reporting
- [ ] Custom screening criteria templates
- [ ] Collaboration features for team screening

### Known Issues
- Rayyan automatic login may fail on some browser configurations
- Large local models (GPT-OSS-20B) require significant RAM
- Rate limiting on Groq free tier

### Performance Improvements Planned
- Caching of LLM responses
- Parallel article processing
- Optimized model loading
- Reduced memory footprint

---

## Version History

- **v1.0.0** (2025-01-31): Initial public release

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs
- Suggesting features
- Submitting pull requests
- Code style guidelines

## Support

For questions or issues:
- Open an issue on GitHub
- Check existing documentation
- Review the troubleshooting section in README.md
