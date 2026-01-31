"""
LLM Article Screener
Automatically screens article titles and abstracts using free LLM APIs based on specific criteria.
Integrates with Rayyan workflow to automatically include/exclude/maybe articles.
"""

import pandas as pd
import json
import time
import requests
from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass
from enum import Enum
import logging
import os
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads .env file from current directory
except (ImportError, UnicodeDecodeError, Exception):
    # If python-dotenv is not installed or .env file is corrupted, skip it
    # Environment variables can still be set manually
    pass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScreeningDecision(Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    MAYBE = "maybe"

@dataclass
class Article:
    title: str
    abstract: str
    authors: str = ""
    journal: str = ""
    year: str = ""
    doi: str = ""
    pmid: str = ""
    decision: Optional[ScreeningDecision] = None
    confidence: float = 0.0
    reasoning: str = ""

class FreeLLMProvider:
    """Base class for free LLM providers"""
    
    def __init__(self):
        self.rate_limit_delay = 1.0  # Default delay between requests
    
    def screen_article(self, article: Article, criteria: Dict) -> Tuple[ScreeningDecision, float, str]:
        """Screen an article and return decision, confidence, and reasoning"""
        raise NotImplementedError

class HuggingFaceProvider(FreeLLMProvider):
    """Free Hugging Face Inference API provider"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        super().__init__()
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {}
        self.rate_limit_delay = 2.0  # Be respectful to free API
        
        # Try to get HF token from environment
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            self.headers["Authorization"] = f"Bearer {hf_token}"
            logger.info("Using Hugging Face API with authentication")
        else:
            logger.info("Using Hugging Face API without authentication (rate limited)")
    
    def screen_article(self, article: Article, criteria: Dict) -> Tuple[ScreeningDecision, float, str]:
        """Screen article using Hugging Face model"""
        try:
            prompt = self._create_screening_prompt(article, criteria)
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.1,
                    "return_full_text": False
                }
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 503:
                # Model is loading, wait and retry
                logger.info("Model loading, waiting 20 seconds...")
                time.sleep(20)
                response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '').strip()
                    return self._parse_llm_response(generated_text)
                else:
                    logger.error(f"Unexpected API response format: {result}")
                    return ScreeningDecision.MAYBE, 0.5, "API response format error"
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return ScreeningDecision.MAYBE, 0.5, f"API error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error screening article: {str(e)}")
            return ScreeningDecision.MAYBE, 0.5, f"Error: {str(e)}"
        
        finally:
            time.sleep(self.rate_limit_delay)

class GroqProvider(FreeLLMProvider):
    """Groq API provider - Fast, free, high-quality LLMs"""
    
    def __init__(self, model_name: str = "llama-3.2-11b-text-preview"):
        super().__init__()
        self.model_name = model_name
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.rate_limit_delay = 2.0  # Increased delay to avoid rate limits
        self.max_retries = 5  # More retries for rate limit handling
        self.backoff_factor = 2  # Exponential backoff multiplier
        
        # Available models on Groq (updated for current supported models 2025)
        self.available_models = {
            "llama3_3_70b": "llama-3.3-70b-versatile",  # Meta Llama 3.3 70B - Best quality
            "llama3_2_90b": "llama-3.2-90b-text-preview", # Meta Llama 3.2 90B - High performance
            "llama3_2_11b": "llama-3.2-11b-text-preview", # Meta Llama 3.2 11B - Balanced
            "llama3_2_3b": "llama-3.2-3b-preview",      # Meta Llama 3.2 3B - Fast
            "mixtral_8x7b": "mixtral-8x7b-32768",       # Mixtral 8x7B - Great reasoning
            "gemma2_9b": "gemma2-9b-it"                 # Google Gemma 2 9B - Good balance
        }
        
        # Try to get Groq API key from environment
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment. Get free key from https://console.groq.com/")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        } if self.api_key else {}
    
    def screen_article(self, article: Article, criteria: Dict) -> Tuple[ScreeningDecision, float, str]:
        """Screen article using Groq API with rate limit handling"""
        if not self.api_key:
            logger.error("Groq API key not available")
            return ScreeningDecision.MAYBE, 0.5, "Groq API key required"
        
        # Try multiple times with exponential backoff for rate limits
        for attempt in range(self.max_retries):
            try:
                prompt = self._create_medical_screening_prompt(article, criteria)
                
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are a medical research expert specializing in systematic reviews and RCT identification for cardiac research."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 200
                }
                
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    return self._parse_llm_response(content)
                
                elif response.status_code == 429:  # Rate limit error
                    wait_time = self.rate_limit_delay * (self.backoff_factor ** attempt)
                    logger.warning(f"Rate limit hit, waiting {wait_time:.1f} seconds (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code in [502, 503, 504]:  # Server errors
                    wait_time = self.rate_limit_delay * (self.backoff_factor ** attempt)
                    logger.warning(f"Server error {response.status_code}, waiting {wait_time:.1f} seconds (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                
                else:
                    logger.error(f"Groq API error: {response.status_code} - {response.text}")
                    return ScreeningDecision.MAYBE, 0.5, f"API error: {response.status_code}"
                    
            except requests.exceptions.Timeout:
                wait_time = self.rate_limit_delay * (self.backoff_factor ** attempt)
                logger.warning(f"Request timeout, waiting {wait_time:.1f} seconds (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(wait_time)
                continue
                
            except Exception as e:
                logger.error(f"Error with Groq screening: {str(e)}")
                if attempt == self.max_retries - 1:
                    return ScreeningDecision.MAYBE, 0.5, f"Error: {str(e)}"
                time.sleep(self.rate_limit_delay)
                continue
        
        # If all retries failed
        logger.error("All Groq API attempts failed")
        return ScreeningDecision.MAYBE, 0.5, "API unavailable after retries"
    
    def _parse_llm_response(self, response: str) -> Tuple[ScreeningDecision, float, str]:
        """Parse LLM response to extract decision, confidence, and reasoning"""
        try:
            # Extract decision
            decision_match = re.search(r'DECISION:\s*(INCLUDE|EXCLUDE|MAYBE)', response, re.IGNORECASE)
            if decision_match:
                decision_str = decision_match.group(1).upper()
                decision = ScreeningDecision(decision_str.lower())
            else:
                # Fallback: look for keywords in response
                response_lower = response.lower()
                if 'include' in response_lower and 'exclude' not in response_lower:
                    decision = ScreeningDecision.INCLUDE
                elif 'exclude' in response_lower:
                    decision = ScreeningDecision.EXCLUDE
                else:
                    decision = ScreeningDecision.MAYBE
            
            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            else:
                confidence = 0.7 if decision != ScreeningDecision.MAYBE else 0.5
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', response, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            else:
                reasoning = "No specific reasoning provided"
            
            return decision, confidence, reasoning
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return ScreeningDecision.MAYBE, 0.5, f"Parse error: {str(e)}"
    
    def _create_medical_screening_prompt(self, article: Article, criteria: Dict) -> str:
        """Create specialized medical research screening prompt with enhanced review detection"""
        return f"""
As a medical research expert, analyze this article for inclusion in a systematic review of PRIMARY RESEARCH STUDIES.

TITLE: {article.title}
ABSTRACT: {article.abstract[:1000]}

STRICT INCLUSION CRITERIA - ALL MUST BE PRESENT:
- Must be randomized controlled trial (RCT) - PRIMARY research only
- Must involve STEMI or NSTEMI patients (explicit MI/ACS context required)
- Must focus on pharmacological interventions during/after MI occurrence
- Must measure cardiovascular outcomes in MI patients specifically
- Must report ORIGINAL trial results (not reviews, discussions, or summaries)
- Must have EXPLICIT myocardial infarction context (not general cardiovascular disease)

STRICT EXCLUSION CRITERIA - EXCLUDE IMMEDIATELY if ANY of these:
- Review articles (systematic, narrative, literature, clinical, comprehensive)
- Meta-analyses or systematic reviews
- Opinion pieces, editorials, commentaries
- Articles that "review", "discuss", "summarize", or "examine" existing evidence
- Articles with titles containing "redefining", "standard of practice", "state of the art"
- Articles that compare treatments without reporting new trial data
- Observational studies, registries, case reports, case series
- Prevention studies (primary or secondary prevention)
- General diabetes studies without explicit MI context
- General cholesterol/dyslipidemia studies without explicit MI context
- General hypertension studies without explicit MI context
- Studies in healthy volunteers or asymptomatic patients
- General cardiovascular prevention studies
- Procedural interventions without pharmacological component (PCI-only, CABG-only, stenting-only)
- Surgical interventions without drug evaluation component
- Device-only studies (pacemaker, defibrillator, valve procedures)
- Mechanical/catheter-based interventions without drug focus
- Pure revascularization studies without pharmacological therapy evaluation

CRITICAL REVIEW DETECTION:
- Does the article REVIEW or DISCUSS existing treatments? → EXCLUDE
- Does it use phrases like "historically", "has been questioned", "evidence shows"? → EXCLUDE  
- Does it SUMMARIZE or SYNTHESIZE existing research? → EXCLUDE
- Does it mention "guidelines" or "practice recommendations"? → EXCLUDE
- Is it describing "advances" or "progress" in the field? → EXCLUDE

CRITICAL PROCEDURAL DETECTION:
- Is this a PCI/angioplasty/stenting study WITHOUT drug evaluation? → EXCLUDE
- Is this a CABG/surgical study WITHOUT pharmacological component? → EXCLUDE
- Is this a device implantation study WITHOUT drug therapy? → EXCLUDE
- Does it focus on procedural/mechanical outcomes WITHOUT drug evaluation? → EXCLUDE
- Is the primary endpoint procedural success rather than drug efficacy? → EXCLUDE

TASK: Determine if this is a PRIMARY RCT reporting NEW trial results in MI patients.

Analyze FIRST for exclusions:
1. Is this a review/discussion article? (EXCLUDE if yes)
2. Is this a general diabetes/cholesterol study without MI context? (EXCLUDE if yes)
3. Is this a procedural/surgical study without pharmacological component? (EXCLUDE if yes)
4. Is this an original RCT in MI/STEMI/NSTEMI patients? (INCLUDE only if yes)
5. Patient population (must be STEMI/NSTEMI/ACS patients, not general population)
6. Intervention type (must be PHARMACOLOGICAL during/after MI, not procedural/surgical)
7. Outcomes measured (drug efficacy outcomes in MI patients, not procedural success)

Respond in this exact format:
DECISION: [INCLUDE/EXCLUDE/MAYBE]
CONFIDENCE: [0.0-1.0]
REASONING: [Brief explanation focusing on PRIMARY research vs review/discussion]
"""

class LocalLLMProvider(FreeLLMProvider):
    """Local LLM provider using transformers library (completely free)"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        super().__init__()
        # Better free models for text analysis
        self.recommended_models = {
            "small_fast": "microsoft/DialoGPT-small",  # Fast, 117M params
            "medium": "microsoft/DialoGPT-medium",     # Balanced, 345M params  
            "large": "microsoft/DialoGPT-large",       # Better quality, 762M params
            "biomedical": "dmis-lab/biobert-base-cased-v1.1",  # Medical domain
            "clinical": "emilyalsentzer/Bio_ClinicalBERT",      # Clinical text
            "llama2_7b": "meta-llama/Llama-2-7b-chat-hf",      # Best quality (needs approval)
            "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.1", # Excellent free option
            "phi2": "microsoft/phi-2",                          # Great small model
            "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Very fast, still good
        }
        
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.rate_limit_delay = 0.1  # No rate limits for local
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            self.transformers_available = True
            logger.info("Transformers library available for local LLM")
        except ImportError:
            self.transformers_available = False
            logger.warning("Transformers library not available. Install with: pip install transformers torch")
    
    def _load_model(self):
        """Load the model and tokenizer"""
        if not self.transformers_available:
            raise ImportError("Transformers library not available")
        
        if self.model is None:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            logger.info(f"Loading local model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            logger.info("Model loaded successfully")
    
    def screen_article(self, article: Article, criteria: Dict) -> Tuple[ScreeningDecision, float, str]:
        """Screen article using local model"""
        try:
            if not self.transformers_available:
                return ScreeningDecision.MAYBE, 0.5, "Transformers library not available"
            
            self._load_model()
            prompt = self._create_screening_prompt(article, criteria)
            
            # Generate response using local model
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            return self._parse_llm_response(response)
            
        except Exception as e:
            logger.error(f"Error with local LLM: {str(e)}")
            return ScreeningDecision.MAYBE, 0.5, f"Local LLM error: {str(e)}"

class RuleBasedProvider(FreeLLMProvider):
    """Rule-based screening as fallback (completely free and fast)"""
    
    def __init__(self):
        super().__init__()
        self.rate_limit_delay = 0.01  # Very fast
    
    def screen_article(self, article: Article, criteria: Dict) -> Tuple[ScreeningDecision, float, str]:
        """Screen article using rule-based approach"""
        try:
            include_terms = criteria.get('include_keywords', [])
            exclude_terms = criteria.get('exclude_keywords', [])
            study_types_exclude = criteria.get('study_types_exclude', [])
            
            text = f"{article.title} {article.abstract}".lower()
            
            # Check for exclusion terms first
            exclude_score = 0
            exclude_reasons = []
            
            for term in exclude_terms:
                if term.lower() in text:
                    exclude_score += 1
                    exclude_reasons.append(f"Contains exclusion term: '{term}'")
            
            for study_type in study_types_exclude:
                if study_type.lower() in text:
                    exclude_score += 2  # Weight study types more heavily
                    exclude_reasons.append(f"Excluded study type: '{study_type}'")
            
            # Check for inclusion terms
            include_score = 0
            include_reasons = []
            
            for term in include_terms:
                if term.lower() in text:
                    include_score += 1
                    include_reasons.append(f"Contains inclusion term: '{term}'")
            
            # Make decision based on scores
            if exclude_score > 2:
                confidence = min(0.9, 0.6 + (exclude_score * 0.1))
                reasoning = "; ".join(exclude_reasons[:3])  # Top 3 reasons
                return ScreeningDecision.EXCLUDE, confidence, reasoning
            
            elif include_score >= 2:
                confidence = min(0.9, 0.6 + (include_score * 0.1))
                reasoning = "; ".join(include_reasons[:3])  # Top 3 reasons
                return ScreeningDecision.INCLUDE, confidence, reasoning
            
            elif exclude_score > 0 or include_score > 0:
                reasoning = f"Mixed signals: {exclude_score} exclude, {include_score} include terms"
                return ScreeningDecision.MAYBE, 0.5, reasoning
            
            else:
                return ScreeningDecision.MAYBE, 0.3, "No clear inclusion or exclusion signals"
                
        except Exception as e:
            logger.error(f"Error in rule-based screening: {str(e)}")
            return ScreeningDecision.MAYBE, 0.5, f"Rule-based error: {str(e)}"

    def _create_screening_prompt(self, article: Article, criteria: Dict) -> str:
        """Create a structured prompt for LLM screening with enhanced review detection"""
        prompt = f"""
You are a systematic review expert screening for PRIMARY RESEARCH STUDIES only.

ARTICLE TITLE: {article.title}

ARTICLE ABSTRACT: {article.abstract}

INCLUSION CRITERIA:
{chr(10).join(f"- {criterion}" for criterion in criteria.get('inclusion_criteria', []))}

EXCLUSION CRITERIA:
{chr(10).join(f"- {criterion}" for criterion in criteria.get('exclusion_criteria', []))}

CRITICAL: EXCLUDE IMMEDIATELY if this article:
- Reviews, discusses, or summarizes existing research
- Uses language like "redefining", "standard of practice", "historically"
- Compares treatments without reporting NEW trial data
- Contains phrases like "evidence shows", "guidelines recommend", "practice recommendations"
- Describes "advances", "progress", or "current status" in the field
- Is a narrative review, systematic review, or meta-analysis
- Focuses on procedural interventions without pharmacological evaluation
- Studies PCI, angioplasty, stenting, or CABG without drug component
- Evaluates device implantations without drug therapy
- Measures procedural success without drug efficacy evaluation

ONLY INCLUDE if this reports ORIGINAL trial results from a NEW study.

Please provide your decision in this exact format:
DECISION: [INCLUDE/EXCLUDE/MAYBE]
CONFIDENCE: [0.0-1.0]
REASONING: [Brief explanation focusing on PRIMARY research vs review/discussion]

Decision:"""
        return prompt
    
    def _parse_llm_response(self, response: str) -> Tuple[ScreeningDecision, float, str]:
        """Parse LLM response to extract decision, confidence, and reasoning"""
        try:
            # Extract decision
            decision_match = re.search(r'DECISION:\s*(INCLUDE|EXCLUDE|MAYBE)', response, re.IGNORECASE)
            if decision_match:
                decision_str = decision_match.group(1).upper()
                decision = ScreeningDecision(decision_str.lower())
            else:
                # Fallback: look for keywords in response
                response_lower = response.lower()
                if 'include' in response_lower and 'exclude' not in response_lower:
                    decision = ScreeningDecision.INCLUDE
                elif 'exclude' in response_lower:
                    decision = ScreeningDecision.EXCLUDE
                else:
                    decision = ScreeningDecision.MAYBE
            
            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            else:
                confidence = 0.7 if decision != ScreeningDecision.MAYBE else 0.5
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', response, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            else:
                reasoning = "No specific reasoning provided"
            
            return decision, confidence, reasoning
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return ScreeningDecision.MAYBE, 0.5, f"Parse error: {str(e)}"

class ArticleScreener:
    """Main article screening system"""
    
    def __init__(self, provider: FreeLLMProvider = None):
        self.provider = provider or RuleBasedProvider()
        self.results = []
    
    def load_articles_from_csv(self, file_path: str, title_col: str = 'title', 
                              abstract_col: str = 'abstract') -> List[Article]:
        """Load articles from CSV file"""
        try:
            df = pd.read_csv(file_path)
            articles = []
            
            for _, row in df.iterrows():
                article = Article(
                    title=str(row.get(title_col, '')),
                    abstract=str(row.get(abstract_col, '')),
                    authors=str(row.get('authors', '')),
                    journal=str(row.get('journal', '')),
                    year=str(row.get('year', '')),
                    doi=str(row.get('doi', '')),
                    pmid=str(row.get('pmid', ''))
                )
                articles.append(article)
            
            logger.info(f"Loaded {len(articles)} articles from {file_path}")
            return articles
            
        except Exception as e:
            logger.error(f"Error loading articles from CSV: {str(e)}")
            return []
    
    def load_criteria_from_json(self, file_path: str) -> Dict:
        """Load screening criteria from JSON file"""
        try:
            with open(file_path, 'r') as f:
                criteria = json.load(f)
            logger.info(f"Loaded screening criteria from {file_path}")
            return criteria
        except Exception as e:
            logger.error(f"Error loading criteria: {str(e)}")
            return {}
    
    def screen_articles(self, articles: List[Article], criteria: Dict, 
                       batch_size: int = 10) -> List[Article]:
        """Screen a list of articles"""
        screened_articles = []
        total = len(articles)
        
        logger.info(f"Starting screening of {total} articles...")
        
        for i, article in enumerate(articles):
            try:
                decision, confidence, reasoning = self.provider.screen_article(article, criteria)
                
                article.decision = decision
                article.confidence = confidence
                article.reasoning = reasoning
                
                screened_articles.append(article)
                
                if (i + 1) % batch_size == 0:
                    logger.info(f"Processed {i + 1}/{total} articles ({((i + 1)/total)*100:.1f}%)")
                
            except KeyboardInterrupt:
                logger.info(f"Screening interrupted after {i + 1} articles")
                break
            except Exception as e:
                logger.error(f"Error screening article {i + 1}: {str(e)}")
                article.decision = ScreeningDecision.MAYBE
                article.confidence = 0.5
                article.reasoning = f"Screening error: {str(e)}"
                screened_articles.append(article)
        
        self.results = screened_articles
        logger.info(f"Screening completed. Processed {len(screened_articles)} articles")
        return screened_articles
    
    def save_results_to_csv(self, output_file: str):
        """Save screening results to CSV"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        try:
            data = []
            for article in self.results:
                data.append({
                    'title': article.title,
                    'abstract': article.abstract,
                    'authors': article.authors,
                    'journal': article.journal,
                    'year': article.year,
                    'doi': article.doi,
                    'pmid': article.pmid,
                    'decision': article.decision.value if article.decision else '',
                    'confidence': article.confidence,
                    'reasoning': article.reasoning
                })
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            
            # Print summary
            summary = df['decision'].value_counts()
            logger.info(f"Results saved to {output_file}")
            logger.info("Screening Summary:")
            for decision, count in summary.items():
                logger.info(f"  {decision}: {count} articles")
                
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of screening results"""
        if not self.results:
            return {}
        
        decisions = [article.decision.value for article in self.results if article.decision]
        confidences = [article.confidence for article in self.results]
        
        from collections import Counter
        decision_counts = Counter(decisions)
        
        return {
            'total_articles': len(self.results),
            'include_count': decision_counts.get('include', 0),
            'exclude_count': decision_counts.get('exclude', 0),
            'maybe_count': decision_counts.get('maybe', 0),
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'high_confidence_count': sum(1 for c in confidences if c >= 0.8)
        }

def create_sample_criteria_file():
    """Create a sample criteria file for labiaplasty research"""
    criteria = {
        "inclusion_criteria": [
            "Randomized controlled trial (RCT)",
            "Clinical trial involving human subjects",
            "Labiaplasty or labial reduction surgery",
            "Cosmetic genital surgery",
            "Female genital cosmetic surgery",
            "Vulvar surgery",
            "Patient satisfaction outcomes",
            "Surgical outcomes",
            "Quality of life measures"
        ],
        "exclusion_criteria": [
            "Animal studies",
            "In vitro studies",
            "Case reports with fewer than 10 patients",
            "Review articles",
            "Meta-analyses",
            "Systematic reviews",
            "Non-human subjects",
            "Pediatric patients",
            "Male genital surgery",
            "Non-surgical treatments only"
        ],
        "include_keywords": [
            "labiaplasty",
            "labial reduction",
            "vulvar surgery",
            "genital cosmetic surgery",
            "labia minora",
            "labia majora",
            "vulvoplasty",
            "randomized controlled trial",
            "clinical trial",
            "patient satisfaction",
            "surgical outcomes"
        ],
        "exclude_keywords": [
            "animal",
            "rat",
            "mouse",
            "mice",
            "in vitro",
            "cell culture",
            "systematic review",
            "meta-analysis",
            "case report",
            "review article",
            "pediatric",
            "children",
            "male"
        ],
        "study_types_exclude": [
            "systematic review",
            "meta-analysis",
            "narrative review",
            "case report",
            "case series",
            "animal study",
            "in vitro study",
            "retrospective cohort",
            "cross-sectional",
            "observational"
        ]
    }
    
    with open('screening_criteria_labiaplasty.json', 'w') as f:
        json.dump(criteria, f, indent=2)
    
    logger.info("Sample criteria file created: screening_criteria_labiaplasty.json")

def main():
    """Main function to demonstrate the screening system"""
    print("=" * 60)
    print("LLM Article Screener for Systematic Reviews")
    print("=" * 60)
    
    # Create sample criteria if it doesn't exist
    if not os.path.exists('screening_criteria_labiaplasty.json'):
        create_sample_criteria_file()
    
    # Choose provider
    print("\nAvailable LLM Providers:")
    print("1. Rule-based screening (Fast, Free, Good baseline)")
    print("2. Hugging Face API (Free tier, Internet required)")
    print("3. Local LLM (Free, Requires transformers library)")
    
    choice = input("\nChoose provider (1-3) [1]: ").strip() or "1"
    
    if choice == "1":
        provider = RuleBasedProvider()
        print("Using rule-based screening")
    elif choice == "2":
        provider = HuggingFaceProvider()
        print("Using Hugging Face API")
    elif choice == "3":
        provider = LocalLLMProvider()
        print("Using local LLM")
    else:
        provider = RuleBasedProvider()
        print("Invalid choice, using rule-based screening")
    
    # Initialize screener
    screener = ArticleScreener(provider)
    
    # Load criteria
    criteria = screener.load_criteria_from_json('screening_criteria_labiaplasty.json')
    if not criteria:
        print("Error loading criteria file")
        return
    
    # Get input file
    input_file = input("\nEnter CSV file path with articles: ").strip()
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    
    # Load articles
    articles = screener.load_articles_from_csv(input_file)
    if not articles:
        print("No articles loaded")
        return
    
    print(f"\nLoaded {len(articles)} articles")
    
    # Confirm before processing
    confirm = input(f"\nProceed with screening {len(articles)} articles? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Screening cancelled")
        return
    
    # Screen articles
    try:
        screened_articles = screener.screen_articles(articles, criteria)
        
        # Save results
        output_file = input_file.replace('.csv', '_screened.csv')
        screener.save_results_to_csv(output_file)
        
        # Show summary
        stats = screener.get_summary_stats()
        print(f"\nScreening Summary:")
        print(f"Total articles: {stats['total_articles']}")
        print(f"Include: {stats['include_count']}")
        print(f"Exclude: {stats['exclude_count']}")
        print(f"Maybe: {stats['maybe_count']}")
        print(f"Average confidence: {stats['avg_confidence']:.2f}")
        print(f"High confidence decisions (≥0.8): {stats['high_confidence_count']}")
        
        print(f"\nResults saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\nScreening interrupted by user")
    except Exception as e:
        print(f"Error during screening: {str(e)}")

if __name__ == "__main__":
    main()
