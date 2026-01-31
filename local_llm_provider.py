#!/usr/bin/env python3
"""
Local LLM provider for Rayyan integration
Supports Ollama and Hugging Face Transformers
"""

import requests
import time
import logging
from typing import Dict, Tuple
from llm_article_screener import FreeLLMProvider, Article, ScreeningDecision

logger = logging.getLogger(__name__)

class OllamaProvider(FreeLLMProvider):
    """Ollama local provider - easiest setup with comprehensive cardiac screening"""
    
    def __init__(self, model_name: str = "gpt-oss:20b", base_url: str = "http://localhost:11434"):
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Test connection
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"[CORRECT] Connected to Ollama at {base_url}")
                print(f"Using model: {model_name}")
            else:
                raise Exception("Ollama not responding")
        except Exception as e:
            print(f"[ERROR] Error connecting to Ollama: {e}")
            print("\nTo set up Ollama:")
            print("1. Download from: https://ollama.ai/download")
            print("2. Install and start Ollama")
            print("3. Run: ollama pull gpt-oss:20b")
            print("4. Or use a smaller model: ollama pull llama3.1:8b")
            raise
    
    def screen_article(self, article: Article, criteria: Dict) -> Tuple[ScreeningDecision, float, str]:
        """Screen article using Ollama"""
        try:
            prompt = self._create_medical_screening_prompt(article, criteria)
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 200
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                return self._parse_llm_response(generated_text)
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return ScreeningDecision.MAYBE, 0.5, f"API error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error with Ollama screening: {str(e)}")
            return ScreeningDecision.MAYBE, 0.5, f"Error: {str(e)}"
    
    def _create_medical_screening_prompt(self, article: Article, criteria: Dict) -> str:
        """Create a medical screening prompt for the article"""
        prompt = f"""
You are a medical research expert specializing in systematic reviews and randomized controlled trials (RCTs) for myocardial infarction (MI) pharmacological therapy research.

ARTICLE TO SCREEN:
Title: {article.title}
Abstract: {article.abstract}
Authors: {article.authors}
Year: {article.year}
DOI: {article.doi}

COMPREHENSIVE SCREENING CRITERIA:

STUDY DESIGN REQUIREMENTS:
- Must be a randomized controlled trial (RCT) or randomized clinical trial
- Look for: "randomized controlled trial", "randomised controlled trial", "RCT", "double-blind", "placebo-controlled", "randomly assigned", "randomization", "treatment group", "control group"
- EXCLUDE: systematic reviews, meta-analyses, observational studies, case reports, registry studies, database analyses, surveys, cohort studies, case-control studies, cross-sectional studies

POPULATION REQUIREMENTS:
- Must study patients with myocardial infarction (MI) or acute coronary syndrome (ACS)
- Look for: "myocardial infarction", "MI", "STEMI", "NSTEMI", "ST-elevation myocardial infarction", "non-ST-elevation myocardial infarction", "acute coronary syndrome", "ACS", "heart attack", "post-MI", "acute MI"
- EXCLUDE: studies focused on diabetes, kidney disease, cancer, stroke, lung disease, psychiatric, neurological, orthopedic, or dermatologic conditions

INTERVENTION REQUIREMENTS:
- Must study pharmacological interventions (drugs/medications) for MI treatment
- Look for: "medication", "drug", "pharmaceutical", "therapy", "treatment", "antiplatelet", "statin", "ACE inhibitor", "beta blocker", "aspirin", "clopidogrel", "atorvastatin", "metoprolol", "dose", "dosage", "mg", "pharmacological", "medical therapy", "drug therapy"
- ALLOW: Combined studies (procedures + drugs) if they specifically study the drug component
- EXCLUDE: Pure procedural studies (surgery, PCI, stenting, bypass), health system interventions, adherence programs, lifestyle modifications, exercise programs, dietary interventions

EXCLUSION CRITERIA (AUTOMATIC EXCLUDE):
- Review articles, meta-analyses, systematic reviews
- Observational studies, registry studies, database analyses
- Prevention studies (primary/secondary prevention)
- Non-cardiac populations
- Pure procedural interventions without drug study component
- Health system/technology interventions
- Adherence/behavioral interventions

ANALYSIS STEPS:
1. Check study design - is it clearly an RCT?
2. Check population - does it study MI/ACS patients?
3. Check intervention - does it study pharmacological treatment of MI?
4. Check for exclusion criteria - is it a review, observational, or prevention study?
5. For combined studies - is the drug component being specifically studied?

Please analyze this article and provide your decision in the following format:

DECISION: [INCLUDE/EXCLUDE/MAYBE]
CONFIDENCE: [0.0-1.0]
REASONING: [Detailed explanation of your decision based on the criteria above]

Focus on the specific requirements for MI pharmacological RCTs. Be strict about exclusions.
"""
        return prompt.strip()
    
    def _parse_llm_response(self, response: str) -> Tuple[ScreeningDecision, float, str]:
        """Parse LLM response to extract decision, confidence, and reasoning"""
        try:
            import re
            
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

class HuggingFaceLocalProvider(FreeLLMProvider):
    """Hugging Face Transformers local provider"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-11B-Instruct"):
        super().__init__()
        self.model_name = model_name
        self.device = "cuda" if self._check_cuda() else "cpu"
        
        print(f"Loading {model_name} on {self.device}...")
        print("This may take several minutes...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            print(f"❌ Error: {e}")
            print("\nTry using Ollama instead (easier setup)")
            raise
    
    def _check_cuda(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def screen_article(self, article: Article, criteria: Dict) -> Tuple[ScreeningDecision, float, str]:
        """Screen article using local Hugging Face model"""
        try:
            import torch
            
            prompt = self._create_medical_screening_prompt(article, criteria)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part (remove input prompt)
            generated_text = response[len(prompt):].strip()
            
            return self._parse_llm_response(generated_text)
            
        except Exception as e:
            logger.error(f"Error with local model screening: {str(e)}")
            return ScreeningDecision.MAYBE, 0.5, f"Error: {str(e)}"
    
    def _create_medical_screening_prompt(self, article: Article, criteria: Dict) -> str:
        """Create a medical screening prompt for the article"""
        prompt = f"""
You are a medical research expert specializing in systematic reviews and randomized controlled trials (RCTs) for myocardial infarction (MI) pharmacological therapy research.

ARTICLE TO SCREEN:
Title: {article.title}
Abstract: {article.abstract}
Authors: {article.authors}
Year: {article.year}
DOI: {article.doi}

COMPREHENSIVE SCREENING CRITERIA:

STUDY DESIGN REQUIREMENTS:
- Must be a randomized controlled trial (RCT) or randomized clinical trial
- Look for: "randomized controlled trial", "randomised controlled trial", "RCT", "double-blind", "placebo-controlled", "randomly assigned", "randomization", "treatment group", "control group"
- EXCLUDE: systematic reviews, meta-analyses, observational studies, case reports, registry studies, database analyses, surveys, cohort studies, case-control studies, cross-sectional studies

POPULATION REQUIREMENTS:
- Must study patients with myocardial infarction (MI) or acute coronary syndrome (ACS)
- Look for: "myocardial infarction", "MI", "STEMI", "NSTEMI", "ST-elevation myocardial infarction", "non-ST-elevation myocardial infarction", "acute coronary syndrome", "ACS", "heart attack", "post-MI", "acute MI"
- EXCLUDE: studies focused on diabetes, kidney disease, cancer, stroke, lung disease, psychiatric, neurological, orthopedic, or dermatologic conditions

INTERVENTION REQUIREMENTS:
- Must study pharmacological interventions (drugs/medications) for MI treatment
- Look for: "medication", "drug", "pharmaceutical", "therapy", "treatment", "antiplatelet", "statin", "ACE inhibitor", "beta blocker", "aspirin", "clopidogrel", "atorvastatin", "metoprolol", "dose", "dosage", "mg", "pharmacological", "medical therapy", "drug therapy"
- ALLOW: Combined studies (procedures + drugs) if they specifically study the drug component
- EXCLUDE: Pure procedural studies (surgery, PCI, stenting, bypass), health system interventions, adherence programs, lifestyle modifications, exercise programs, dietary interventions

EXCLUSION CRITERIA (AUTOMATIC EXCLUDE):
- Review articles, meta-analyses, systematic reviews
- Observational studies, registry studies, database analyses
- Prevention studies (primary/secondary prevention)
- Non-cardiac populations
- Pure procedural interventions without drug study component
- Health system/technology interventions
- Adherence/behavioral interventions

ANALYSIS STEPS:
1. Check study design - is it clearly an RCT?
2. Check population - does it study MI/ACS patients?
3. Check intervention - does it study pharmacological treatment of MI?
4. Check for exclusion criteria - is it a review, observational, or prevention study?
5. For combined studies - is the drug component being specifically studied?

Please analyze this article and provide your decision in the following format:

DECISION: [INCLUDE/EXCLUDE/MAYBE]
CONFIDENCE: [0.0-1.0]
REASONING: [Detailed explanation of your decision based on the criteria above]

Focus on the specific requirements for MI pharmacological RCTs. Be strict about exclusions.
"""
        return prompt.strip()
    
    def _parse_llm_response(self, response: str) -> Tuple[ScreeningDecision, float, str]:
        """Parse LLM response to extract decision, confidence, and reasoning"""
        try:
            import re
            
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
