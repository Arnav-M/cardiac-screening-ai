# System Architecture 🏗️

This document describes the architecture and design of the Cardiac Article Screening System.

## Overview

The system is designed as a modular, extensible article screening pipeline that integrates AI models with web automation.

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│              (Command Line / Python Scripts)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Runner Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Groq Runner  │  │ BERT Runner  │  │  LLM Runner  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Core Screening Engine                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Article Screener (llm_article_screener.py)   │  │
│  │  - Article extraction                                  │  │
│  │  - Criteria evaluation                                 │  │
│  │  - Decision making                                     │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  AI Provider Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Groq API     │  │ Bio-Clinical │  │ GPT-OSS      │     │
│  │ Provider     │  │ BERT         │  │ (Ollama)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Web Automation Layer                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Selenium WebDriver                        │  │
│  │  - Browser control                                     │  │
│  │  - Element extraction                                  │  │
│  │  - Rayyan integration                                  │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              External Services / Resources                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Groq Cloud   │  │ Rayyan.ai    │  │ Local Models │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Runner Layer

**Purpose**: Entry points for different AI model configurations

**Files**:
- `run_groq_rayyan.py` - Groq API runner
- `run_pure_llm_rayyan.py` - Bio-ClinicalBERT runner
- `run_local_llm_rayyan.py` - GPT-OSS runner

**Responsibilities**:
- Initialize specific AI provider
- Set up browser automation
- Manage login flow
- Orchestrate screening process
- Handle rate limiting
- Report progress

### 2. Core Screening Engine

**Purpose**: Central article screening logic

**Files**:
- `llm_article_screener.py` - Main screening engine
- `cardiac_llm_screener.py` - Cardiac-specific logic

**Key Classes**:

#### `Article` (dataclass)
```python
@dataclass
class Article:
    title: str
    abstract: str
    authors: str
    journal: str
    year: str
    doi: str
    pmid: str
    decision: Optional[ScreeningDecision] = None
    confidence: float = 0.0
    reasoning: str = ""
```

#### `ArticleScreener`
- Loads articles from various sources
- Applies screening criteria
- Manages screening workflow
- Exports results

### 3. AI Provider Layer

**Purpose**: Abstract interface for different AI models

**Base Class**: `FreeLLMProvider`

**Implementations**:
1. **GroqProvider** (Cloud API)
   - Fast, high-quality responses
   - Rate limiting with exponential backoff
   - Retry logic for robustness

2. **CardiacBioClinicalBERTProvider** (Local)
   - Medical text specialist
   - Semantic analysis
   - Pattern matching

3. **GPTOptimizedProvider** (Local Ollama)
   - GPT-style reasoning
   - Optimized prompts
   - Local inference

4. **RuleBasedProvider** (Fallback)
   - Keyword matching
   - Fast, deterministic
   - No AI required

**Interface**:
```python
def screen_article(self, article: Article, criteria: Dict) 
    -> Tuple[ScreeningDecision, float, str]:
    """
    Returns: (decision, confidence, reasoning)
    """
```

### 4. Web Automation Layer

**Purpose**: Interact with Rayyan web interface

**Technology**: Selenium WebDriver

**Key Functions**:
- `setup_driver()` - Initialize Chrome browser
- `login_to_rayyan()` - Automatic login
- `extract_current_article()` - Extract article data
- `click_decision_button()` - Apply screening decision

**Challenges Solved**:
- Dynamic content loading
- Virtualized lists
- Authentication handling
- Element selection strategies

### 5. Configuration Layer

**Purpose**: Manage settings and credentials

**Files**:
- `config.py` - Configuration helper
- `.env` - Environment variables
- `screening_criteria_cardiac.json` - Screening rules

**Configuration Hierarchy**:
1. Environment variables (`.env`)
2. Command-line arguments
3. Default values

## Data Flow

### Typical Screening Workflow

```
1. User runs: python run_groq_rayyan.py --max-articles 50
                     │
2. Runner initializes AI provider
                     │
3. Browser opens and navigates to Rayyan
                     │
4. System logs in (or prompts user)
                     │
5. For each article:
   ├─ Extract title & abstract
   ├─ Send to AI provider
   ├─ AI evaluates against criteria
   ├─ Return decision + confidence + reasoning
   ├─ Click appropriate button (Include/Exclude/Maybe)
   └─ Wait for next article
                     │
6. Display summary statistics
                     │
7. Close browser
```

### Screening Decision Flow

```
Article → AI Provider → Prompt Engineering
                            │
                    ┌───────┴───────┐
                    │               │
              Context Building   Pattern Matching
                    │               │
                    └───────┬───────┘
                            │
                      LLM Inference
                            │
                    ┌───────┴───────┐
                    │               │
              Parse Response    Extract Reasoning
                    │               │
                    └───────┬───────┘
                            │
                  Confidence Scoring
                            │
                  Return (Decision, Confidence, Reasoning)
```

## Design Patterns

### 1. Strategy Pattern
Used for AI providers - different screening strategies with common interface

### 2. Factory Pattern
Provider creation based on configuration

### 3. Adapter Pattern
Converting different AI API responses to common format

### 4. Chain of Responsibility
Fallback from paid API → free models → rule-based

## Error Handling Strategy

### Levels of Fallback

1. **Primary**: Groq API (if configured)
   - Retry with exponential backoff
   - Handle rate limits gracefully

2. **Secondary**: Bio-ClinicalBERT (always available)
   - Local inference
   - No external dependencies

3. **Tertiary**: Rule-based (always works)
   - Simple keyword matching
   - Never fails

### Error Recovery

```python
try:
    decision = groq_provider.screen_article(article, criteria)
except APIError:
    logger.warning("Groq API failed, falling back to Bio-ClinicalBERT")
    decision = bert_provider.screen_article(article, criteria)
except Exception:
    logger.error("All AI providers failed, using rule-based")
    decision = rule_provider.screen_article(article, criteria)
```

## Performance Considerations

### Bottlenecks
1. **AI Model Inference**: 1-5 seconds per article
2. **Web Automation**: 1-2 seconds per article
3. **Network Latency**: 0.5-1 second per API call

### Optimizations
1. **Rate Limiting**: Prevent API throttling
2. **Caching**: Store screening criteria in memory
3. **Batch Processing**: Process multiple articles when possible
4. **Lazy Loading**: Load models only when needed

## Security Architecture

### API Key Management
- Stored in `.env` file (never committed)
- Loaded via environment variables
- Validated at runtime

### Data Privacy
- No article data sent to external services (except chosen AI API)
- Local processing preferred
- No persistent storage of sensitive data

### Access Control
- User responsible for Rayyan credentials
- Browser automation uses user's session
- No credential storage in code

## Extensibility Points

### Adding New AI Provider

1. Inherit from `FreeLLMProvider`
2. Implement `screen_article()` method
3. Add configuration to `config.py`
4. Create runner script
5. Update documentation

### Adding New Data Source

1. Create parser class
2. Convert to `Article` objects
3. Integrate with `ArticleScreener`

### Custom Screening Criteria

1. Edit `screening_criteria_cardiac.json`
2. Add new criteria categories
3. Update prompt templates
4. Test with sample articles

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock external dependencies
- Fast execution

### Integration Tests
- Full workflow testing
- Real API calls (with test key)
- Browser automation

### Manual Testing
- Real-world article screening
- Different browsers
- Various article types

## Deployment Considerations

### Dependencies
- Python 3.8+ required
- Chrome browser for automation
- Internet connection for cloud APIs
- Optional: GPU for faster local models

### Cross-Platform Support
- Windows: Primary development
- macOS: Supported
- Linux: Supported

### Resource Requirements
- **Minimum**: 4GB RAM, 2GB disk
- **Recommended**: 8GB RAM, 5GB disk
- **Optimal**: 16GB RAM, 10GB disk (for large models)

## Future Architecture Improvements

### Planned Enhancements
1. **Microservices**: Separate AI inference from web automation
2. **API Gateway**: RESTful API for remote access
3. **Queue System**: Process articles asynchronously
4. **Database**: Persistent storage for results
5. **Web UI**: Browser-based interface
6. **Docker**: Containerized deployment
7. **CI/CD**: Automated testing and deployment

---

This architecture supports:
- ✅ Modularity and extensibility
- ✅ Multiple AI providers
- ✅ Error resilience
- ✅ Easy configuration
- ✅ Cross-platform compatibility
- ✅ Future enhancements
