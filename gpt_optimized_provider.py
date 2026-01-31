#!/usr/bin/env python3
"""
GPT-Optimized Medical Article Screener
Optimized specifically for GPT models with clearer, more focused prompts
"""

import requests
import logging
from typing import Dict, Tuple
from refman_parser import Article, ScreeningDecision

logger = logging.getLogger(__name__)

class GPTOptimizedProvider:
    """GPT-optimized provider with clearer prompts and better decision logic"""
    
    def __init__(self, model_name: str = "gpt-oss:20b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Test connection
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"[CORRECT] Connected to Ollama at {base_url}")
                print(f"Using GPT-optimized model: {model_name}")
            else:
                raise Exception("Ollama not responding")
        except Exception as e:
            print(f"[ERROR] Error connecting to Ollama: {e}")
            raise
    
    def screen_article(self, article: Article, criteria: Dict) -> Tuple[ScreeningDecision, float, str]:
        """Screen article using GPT-optimized prompt"""
        try:
            prompt = self._create_gpt_optimized_prompt(article, criteria)
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 150
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                return self._parse_gpt_response(generated_text)
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return ScreeningDecision.MAYBE, 0.5, f"API error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error with GPT screening: {e}")
            return ScreeningDecision.MAYBE, 0.5, f"Error: {str(e)}"
    
    def _create_gpt_optimized_prompt(self, article: Article, criteria: Dict) -> str:
        """Create a clear, focused prompt optimized for GPT models"""
        prompt = f"""You are a medical research expert. Analyze this article for inclusion in a systematic review of drug treatments for heart attack patients.

ARTICLE:
Title: {article.title}
Abstract: {article.abstract}
Year: {article.year}

INCLUSION CRITERIA (ALL 3 MUST BE TRUE):
1. RANDOMIZED TRIAL: Must be a randomized controlled trial (RCT) or randomized clinical trial
   - Look for: "randomized", "RCT", "randomly assigned", "double-blind", "placebo-controlled"
   - EXCLUDE: reviews, meta-analyses, observational studies, case reports

2. HEART ATTACK PATIENTS: Must study patients who had a heart attack (myocardial infarction)
   - Look for: "myocardial infarction", "MI", "STEMI", "NSTEMI", "heart attack", "acute coronary syndrome"
   - EXCLUDE: diabetes, cancer, stroke, or other non-heart conditions

3. DRUG TREATMENT: Must study medications/drugs for heart attack treatment
   - Look for: "medication", "drug", "aspirin", "clopidogrel", "statin", "beta blocker", "ACE inhibitor"
   - EXCLUDE: surgery, procedures, lifestyle changes, or non-drug treatments

DECISION RULES:
- INCLUDE: If ALL 3 criteria are clearly met
- EXCLUDE: If ANY criterion is clearly not met
- MAYBE: If unclear or mixed evidence

Respond with ONLY this format:
DECISION: [INCLUDE/EXCLUDE/MAYBE]
CONFIDENCE: [0.0-1.0]
REASONING: [Brief explanation]"""
        
        return prompt.strip()
    
    def _parse_gpt_response(self, response: str) -> Tuple[ScreeningDecision, float, str]:
        """Parse GPT response with improved error handling"""
        try:
            import re
            
            # Clean up response
            response = response.strip()
            
            # Extract decision
            decision_match = re.search(r'DECISION:\s*(INCLUDE|EXCLUDE|MAYBE)', response, re.IGNORECASE)
            if decision_match:
                decision_str = decision_match.group(1).upper()
                if decision_str == 'INCLUDE':
                    decision = ScreeningDecision.INCLUDE
                elif decision_str == 'EXCLUDE':
                    decision = ScreeningDecision.EXCLUDE
                else:
                    decision = ScreeningDecision.MAYBE
            else:
                decision = ScreeningDecision.MAYBE
            
            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', response)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            else:
                confidence = 0.5
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            else:
                reasoning = "No reasoning provided"
            
            return decision, confidence, reasoning
            
        except Exception as e:
            logger.error(f"Error parsing GPT response: {e}")
            return ScreeningDecision.MAYBE, 0.5, f"Parse error: {str(e)}"
