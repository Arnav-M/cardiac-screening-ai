# Contributing to Cardiac Article Screening System

First off, thank you for considering contributing to this project! This is a research tool designed to help systematic review researchers, and your contributions can make a real difference.

## How Can I Contribute?

### 1. Reporting Bugs

If you find a bug, please open an issue with:
- Clear, descriptive title
- Steps to reproduce the problem
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, model used)
- Error messages or logs (if any)
- Screenshots (if applicable)

**Example:**
```
Title: "Groq API connection fails on Windows 11"

Steps to reproduce:
1. Run `python run_groq_rayyan.py`
2. System shows connection error

Expected: Should connect to Groq API
Actual: ConnectionError: [Errno 11001] getaddrinfo failed

Environment:
- OS: Windows 11
- Python: 3.10.5
- Groq API key: Configured
```

### 2. Suggesting Features

We welcome feature suggestions! Please open an issue with:
- Clear description of the feature
- Why it would be useful
- How it might work
- Any examples from other tools

**Example:**
```
Title: "Add support for Claude API"

Description: Add Claude API as an alternative to Groq for article screening

Rationale: Claude has excellent reasoning capabilities and some users may prefer it

Suggested Implementation:
- Create ClaudeProvider class
- Add API key configuration
- Update config.py with Claude option
```

### 3. Improving Documentation

Documentation improvements are always welcome:
- Fix typos or unclear explanations
- Add examples
- Improve installation instructions
- Translate to other languages
- Create video tutorials

### 4. Code Contributions

#### Before You Start

1. Check existing issues and pull requests
2. Open an issue to discuss major changes
3. Fork the repository
4. Create a feature branch

#### Setting Up Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/cardiac-screening-system.git
cd cardiac-screening-system

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest black flake8 mypy
```

#### Code Style Guidelines

- **Python Style**: Follow PEP 8
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Use type hints where possible
- **Comments**: Write clear, concise comments
- **Naming**: Use descriptive variable names

**Example:**
```python
def screen_article(self, article: Article, criteria: Dict) -> Tuple[ScreeningDecision, float, str]:
    """
    Screen an article based on inclusion/exclusion criteria.
    
    Args:
        article: Article object containing title, abstract, etc.
        criteria: Dictionary with screening criteria
        
    Returns:
        Tuple of (decision, confidence, reasoning)
        - decision: ScreeningDecision enum (INCLUDE, EXCLUDE, MAYBE)
        - confidence: Float between 0.0 and 1.0
        - reasoning: String explaining the decision
        
    Raises:
        ValueError: If article or criteria is invalid
    """
    # Implementation here
    pass
```

#### Testing

- Write tests for new features
- Ensure existing tests pass
- Test on multiple platforms if possible

```bash
# Run tests (if you've added pytest)
pytest tests/

# Check code style
flake8 .

# Format code
black .
```

#### Commit Messages

Use clear, descriptive commit messages:

```
# Good
Add Claude API support for article screening
Fix Rayyan login issue on Chrome 120+
Update documentation for Bio-ClinicalBERT setup

# Bad
Update code
Fix bug
Changes
```

#### Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new features
3. **Update CHANGELOG.md** with your changes
4. **Ensure CI passes** (once we have CI/CD)
5. **Request review** from maintainers

**Pull Request Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] CHANGELOG.md updated
```

### 5. Adding New AI Models

To add a new AI model provider:

1. Create a new provider class in `llm_article_screener.py`:
```python
class NewModelProvider(FreeLLMProvider):
    """Provider for NewModel API"""
    
    def __init__(self, model_name: str = "model-name"):
        super().__init__()
        self.model_name = model_name
        self.api_key = os.getenv('NEWMODEL_API_KEY')
    
    def screen_article(self, article: Article, criteria: Dict) -> Tuple[ScreeningDecision, float, str]:
        # Implementation here
        pass
```

2. Update `config.py` with the new model
3. Create a runner script (e.g., `run_newmodel_rayyan.py`)
4. Update `README.md` with the new model option
5. Add configuration to `.env.example`

### 6. Improving Screening Criteria

Help improve the screening accuracy:
- Refine inclusion/exclusion criteria
- Add medical domain expertise
- Test on different article types
- Share your screening criteria JSON files

## Code of Conduct

### Our Standards

- **Be respectful** and inclusive
- **Be patient** with new contributors
- **Provide constructive feedback**
- **Focus on the problem**, not the person
- **Accept constructive criticism** gracefully

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Personal attacks
- Publishing others' private information
- Unprofessional conduct

## Questions?

Don't hesitate to ask questions:
- Open an issue with the "question" label
- Check existing documentation
- Review closed issues for similar questions

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in release notes
- Thanked in the README.md

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Getting Started Checklist for New Contributors

- [ ] Read this CONTRIBUTING.md
- [ ] Read the README.md
- [ ] Set up development environment
- [ ] Run the project locally
- [ ] Pick an issue labeled "good first issue"
- [ ] Fork the repository
- [ ] Make your changes
- [ ] Submit a pull request

## Priority Areas for Contribution

Current priorities (as of 2025-01-31):
1. Testing suite development
2. Additional AI model integrations
3. Documentation improvements
4. Bug fixes for Rayyan automation
5. Performance optimizations

## Thank You!

Every contribution, no matter how small, helps make this tool better for researchers worldwide. Thank you for being part of this project!

---

**Happy Contributing!**

For urgent matters, please open an issue or contact the maintainers.
