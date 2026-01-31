# LLM Models Used in Cardiac Article Screening Project

## Summary of All LLMs We've Tried

### 1. **Bio-ClinicalBERT** (Primary Model)
- **Provider**: `CardiacBioClinicalBERTProvider`
- **Model**: `emilyalsentzer/Bio_ClinicalBERT`
- **Type**: Specialized biomedical BERT model
- **Status**: ✅ **Currently Active** (used in `run_pure_llm_rayyan.py`)
- **Best For**: Medical text understanding, context-aware screening
- **File**: `cardiac_llm_screener.py`

### 2. **GPT-OSS-20B** (Current GPT Model)
- **Provider**: `GPTOptimizedProvider` / `OllamaProvider`
- **Model**: `gpt-oss:20b`
- **Type**: OpenAI open-weight model (20 billion parameters)
- **Status**: ✅ **Currently Active** (used in `run_local_llm_rayyan.py`)
- **Size**: ~13 GB
- **Best For**: General-purpose medical screening with GPT-style reasoning
- **Files**: `gpt_optimized_provider.py`, `run_local_llm_rayyan.py`

### 3. **Llama 3.3 70B** (Tried but Removed)
- **Provider**: `OllamaProvider`
- **Model**: `llama3.3:70b-instruct-q4_0`
- **Type**: Quantized Llama 3.3 (4-bit quantization)
- **Status**: ❌ **Removed** (was too large/slow)
- **Size**: ~39 GB
- **Reason Removed**: Too large for machine, slow inference

### 4. **Llama 3.2 3B** (Tried but Removed)
- **Provider**: `OllamaProvider`
- **Model**: `llama3.2:3b`
- **Type**: Small Llama model
- **Status**: ❌ **Removed**
- **Size**: ~2.0 GB
- **Reason Removed**: User requested removal of all Llama instances

### 5. **Llama 3.1 8B** (Downloaded but Not Used)
- **Provider**: `OllamaProvider`
- **Model**: `llama3.1:8b`
- **Type**: Standard Llama 3.1 model
- **Status**: ⚠️ **Downloaded but Not Active**
- **Size**: ~4.9 GB
- **Note**: Still available but not currently configured

### 6. **Llama 3.2 11B** (Referenced but Not Available)
- **Provider**: `OllamaProvider`
- **Model**: `llama3.2:11b`
- **Type**: Mid-size Llama model
- **Status**: ❌ **Not Available** (tried to pull but doesn't exist in Ollama)
- **Note**: Referenced in code but model doesn't exist in Ollama registry

### 7. **Hugging Face Models** (Configured but Not Used)
- **Provider**: `HuggingFaceLocalProvider`
- **Models Referenced**:
  - `meta-llama/Llama-3.2-11B-Instruct` (fallback in code)
  - `meta-llama/Llama-3.2-3B-Instruct` (test file)
- **Status**: ⚠️ **Configured but Not Active**
- **Note**: Requires CUDA/GPU, more complex setup

### 8. **Gemma Models** (Available but Not Used)
- **Models**: 
  - `gemma2:27b` (~15 GB)
  - `gemma3:4b` (~3.3 GB)
- **Status**: ⚠️ **Available but Not Used**
- **Note**: Still in Ollama but never configured for screening

### 9. **Groq Models** (External API)
- **Provider**: `GroqProvider`
- **Model**: `llama-3.3-70b-versatile`
- **Type**: External API (not local)
- **Status**: ⚠️ **Available** (in `run_groq_rayyan.py`)
- **Note**: Requires Groq API key, not local

## Currently Active Models

1. **Bio-ClinicalBERT** - Used in `run_pure_llm_rayyan.py`
2. **GPT-OSS-20B** - Used in `run_local_llm_rayyan.py` (with GPT-optimized prompts)

## Model Comparison

| Model | Size | Speed | Medical Expertise | Status |
|-------|------|-------|-------------------|--------|
| Bio-ClinicalBERT | ~400 MB | Fast | ⭐⭐⭐⭐⭐ | Active |
| GPT-OSS-20B | ~13 GB | Medium | ⭐⭐⭐ | Active |
| Llama 3.3 70B | ~39 GB | Slow | ⭐⭐⭐⭐ | Removed |
| Llama 3.1 8B | ~4.9 GB | Fast | ⭐⭐⭐ | Available |
| Gemma2 27B | ~15 GB | Medium | ⭐⭐⭐ | Available |

## Recommendations

**For Best Medical Screening**: Use **Bio-ClinicalBERT** (`run_pure_llm_rayyan.py`)
- Best medical text understanding
- Fastest inference
- Specialized for biomedical text

**For GPT-Style Reasoning**: Use **GPT-OSS-20B** (`run_local_llm_rayyan.py`)
- Good general reasoning
- Better for complex decision-making
- Requires optimization (which we've done with `gpt_optimized_provider.py`)

