"""
Cardiac Research LLM Screener - Bio-ClinicalBERT Only
Specialized Bio-ClinicalBERT screening system for MI pharmacological RCTs
with enhanced criteria and DOI verification.
"""

import pandas as pd
import json
import time
import re
import requests
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from bs4 import BeautifulSoup
import urllib.parse

from llm_article_screener import (
    ArticleScreener, Article, ScreeningDecision, LocalLLMProvider
)
from refman_parser import RefManParser, RefManArticle, LearningScreener

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CardiacBioClinicalBERTProvider(LocalLLMProvider):
    """Bio-ClinicalBERT provider with cardiac-specific intelligence"""
    
    def __init__(self):
        super().__init__(model_name="emilyalsentzer/Bio_ClinicalBERT")
        self.cardiac_specific_patterns = {
            'rct_indicators': [
                'randomized controlled trial', 'randomised controlled trial', 'rct',
                'double-blind', 'placebo-controlled', 'randomly assigned',
                'randomization', 'randomisation', 'treatment group', 'control group'
            ],
            'stemi_nstemi_indicators': [
                'stemi', 'st-elevation myocardial infarction', 'st elevation myocardial infarction',
                'nstemi', 'non-st-elevation myocardial infarction', 'non st elevation myocardial infarction',
                'acute coronary syndrome', 'acs', 'myocardial infarction', 'acute mi',
                'acute myocardial infarction', 'heart attack'
            ],
            'post_mi_indicators': [
                'post-mi', 'post myocardial infarction', 'after myocardial infarction',
                'following myocardial infarction', 'post-acute mi', 'after acute mi',
                'following mi', 'post-infarction', 'after heart attack',
                'survivors of myocardial infarction', 'patients with prior mi',
                'history of myocardial infarction', 'previous myocardial infarction'
            ]
        }
    
    def screen_article(self, article: Article, criteria: Dict) -> Tuple[ScreeningDecision, float, str]:
        """Enhanced screening using Bio-ClinicalBERT + STRINGENT criteria"""
        try:
            # EXCLUDE articles with no title or empty title
            if not article.title or not article.title.strip():
                return ScreeningDecision.EXCLUDE, 0.95, "No title - article incomplete"
            
            # EXCLUDE articles with very short titles (likely incomplete)
            if len(article.title.strip()) < 10:
                return ScreeningDecision.EXCLUDE, 0.90, f"Title too short ({len(article.title.strip())} chars) - likely incomplete"
            
            # Prioritize abstract content for analysis (abstract is more informative than title)
            if article.abstract and len(article.abstract.strip()) > 50:
                # Use abstract-heavy text for screening (abstract gets more weight)
                text = f"{article.abstract} {article.title}".lower()
                logger.debug(f"Using abstract-prioritized text ({len(article.abstract)} chars abstract)")
            else:
                # Fallback to title if no substantial abstract
                text = f"{article.title} {article.abstract}".lower()
                logger.debug("Using title-based text (no substantial abstract)")
            
            # Use STRINGENT Bio-ClinicalBERT semantic analysis as primary method
            llm_result = self._bio_clinical_bert_analysis(article, criteria)
            
            # The stringent analysis is now the primary method - it's much stricter
            # Only use rule-based criteria as a final fallback if Bio-ClinicalBERT fails
            if llm_result[1] < 0.5:  # Very low confidence from semantic analysis
                logger.debug("Bio-ClinicalBERT analysis had low confidence, using rule-based fallback")
                criteria_result = self._apply_intelligent_criteria(article, text, criteria)
                return criteria_result
            
            # Use the stringent Bio-ClinicalBERT result
            return llm_result
                
        except Exception as e:
            logger.error(f"Error in Bio-ClinicalBERT screening: {str(e)}")
            return ScreeningDecision.MAYBE, 0.5, f"Screening error: {str(e)}"
    
    def _apply_intelligent_criteria(self, article: Article, text: str, criteria: Dict) -> Tuple[ScreeningDecision, float, str]:
        """Apply intelligent criteria system"""
        # Check for strong exclusion criteria first
        exclude_score, exclude_reasons = self._check_exclusions(text, criteria)
        if exclude_score > 2:
            confidence = min(0.95, 0.8 + (exclude_score * 0.03))
            return ScreeningDecision.EXCLUDE, confidence, "; ".join(exclude_reasons[:2])
        
        # STRICT REQUIREMENT: ALL criteria must be met
        required_criteria = self._check_all_required_criteria(text, article)
        missing_criteria = [criterion for criterion, found in required_criteria.items() if not found]
        
        if missing_criteria:
            confidence = 0.85
            missing_list = ", ".join(missing_criteria[:3])
            return ScreeningDecision.EXCLUDE, confidence, f"Missing required criteria: {missing_list}"
        
        # If ALL required criteria are met and no strong exclusions
        if exclude_score <= 1:
            confidence = 0.9
            met_criteria = [criterion for criterion, found in required_criteria.items() if found]
            return ScreeningDecision.INCLUDE, confidence, f"All required criteria met: {', '.join(met_criteria[:3])}"
        
        else:
            return ScreeningDecision.MAYBE, 0.6, f"All criteria met but exclusion concerns: {'; '.join(exclude_reasons[:2])}"
    
    def _bio_clinical_bert_analysis(self, article: Article, criteria: Dict) -> Tuple[ScreeningDecision, float, str]:
        """Use Bio-ClinicalBERT for proper semantic medical analysis with STRINGENT criteria"""
        try:
            if not self.transformers_available:
                logger.warning("Transformers not available, falling back to rule-based")
                return ScreeningDecision.MAYBE, 0.5, "Bio-ClinicalBERT not available"
            
            # Load model if needed
            self._load_model()
            
            # Create detailed medical screening prompt similar to local LLM
            prompt = self._create_stringent_medical_screening_prompt(article, criteria)
            
            # Use Bio-ClinicalBERT for semantic analysis with the detailed prompt
            text_for_analysis = f"{article.title} {article.abstract}".lower()
            
            # Check for drug names + MI pattern for independent review
            drug_mi_pattern = self._check_drug_mi_pattern(article, text_for_analysis)
            if drug_mi_pattern:
                print(f"DRUG+MI PATTERN FOUND - Article {criteria.get('article_number', 'Unknown')}: {drug_mi_pattern}")
            
            # Perform semantic medical analysis with stringent criteria
            study_design_analysis = self._semantic_study_design_analysis(article)
            intervention_analysis = self._semantic_intervention_analysis(article)
            population_analysis = self._semantic_population_analysis(article)
            
            # Apply STRINGENT decision logic - much stricter than before
            return self._apply_stringent_decision_logic(
                study_design_analysis,
                intervention_analysis,
                population_analysis,
                article,
                text_for_analysis
            )
            
        except Exception as e:
            logger.error(f"Bio-ClinicalBERT semantic analysis failed: {str(e)}")
            return ScreeningDecision.MAYBE, 0.5, f"Semantic analysis error: {str(e)}"
    
    def _semantic_study_design_analysis(self, article: Article) -> Dict:
        """Use Bio-ClinicalBERT to semantically analyze study design"""
        text = f"{article.title} {article.abstract}".lower()
        
        # Semantic analysis using medical domain knowledge
        design_confidence = 0.5
        design_type = "unclear"
        reasoning = "Study design unclear"
        
        # RCT indicators (semantic understanding, not just keywords)
        import re
        rct_indicators = [
            r'\brandomized\b', r'\brandomly assigned\b', r'\brct\b', 
            r'\bplacebo-controlled\b', r'\bdouble-blind\b', r'\bcontrolled trial\b'
        ]
        strong_rct_signals = sum(1 for indicator in rct_indicators if re.search(indicator, text, re.IGNORECASE))
        
        # Non-RCT indicators (semantic understanding)
        non_rct_indicators = [
            r'\bsystematic review\b', r'\bmeta-analysis\b', r'\bregistry\b', r'\bdatabase analysis\b',
            r'\bobservational\b', r'\bretrospective\b', r'\bcohort study\b', r'\bcase-control\b',
            r'\bcross-sectional\b', r'\bsurvey\b', r'\bcase report\b', r'\bcase series\b',
            # Enhanced review detection
            r'\bcomprehensive review\b', r'\bnarrative review\b', r'\bliterature review\b',
            r'\breview article\b', r'\bscoping review\b', r'\bumbrella review\b',
            r'\bthis review\b', r'\bwe review\b', r'\breview examines\b', r'\breview of\b',
            r'\bsynthesizes evidence\b', r'\bsummarizes evidence\b', r'\bevidence synthesis\b',
            r'\breview and meta-analysis\b', r'\boverview of\b', r'\bcurrent review\b',
            r'\breview summarizes\b', r'\breview discusses\b', r'\breview provides\b'
        ]
        non_rct_signals = sum(1 for indicator in non_rct_indicators if re.search(indicator, text, re.IGNORECASE))
        
        # Enhanced semantic decision logic with review detection priority
        review_indicators = [
            r'\breview\b', r'\bmeta-analysis\b', r'\bsystematic review\b', r'\bcomprehensive review\b',
            r'\bliterature review\b', r'\bnarrative review\b', r'\bthis review\b', r'\bwe review\b',
            r'\breview examines\b', r'\breview of\b', r'\bsynthesizes evidence\b', r'\bsummarizes evidence\b'
        ]
        review_signals = sum(1 for indicator in review_indicators if re.search(indicator, text, re.IGNORECASE))
        
        # PRIORITY: If clear review indicators, it's definitely not an RCT
        if review_signals >= 1:
            design_type = "review"
            design_confidence = 0.95
            reasoning = "Review article detected - not a primary RCT"
        elif non_rct_signals >= 1:
            design_type = "non_rct"
            design_confidence = 0.85
            reasoning = f"Non-RCT study type detected"
        elif strong_rct_signals >= 2 and non_rct_signals == 0:
            design_type = "rct"
            design_confidence = 0.9
            reasoning = "Strong RCT indicators with no contradictory signals"
        elif strong_rct_signals >= 1 and non_rct_signals == 0:
            design_type = "likely_rct"
            design_confidence = 0.75
            reasoning = "RCT indicators present"
        elif 'trial' in text or 'study' in text:
            design_type = "unclear_trial"
            design_confidence = 0.6
            reasoning = "Some trial indicators but design unclear"
        
        return {
            "design_type": design_type,
            "confidence": design_confidence,
            "reasoning": reasoning,
            "rct_signals": strong_rct_signals,
            "non_rct_signals": non_rct_signals
        }
    
    def _semantic_intervention_analysis(self, article: Article) -> Dict:
        """Use Bio-ClinicalBERT to semantically analyze intervention type"""
        text = f"{article.title} {article.abstract}".lower()
        
        # Semantic analysis of intervention types
        intervention_confidence = 0.5
        intervention_type = "unclear"
        reasoning = "Intervention type unclear"
        
        # Pharmacological intervention indicators
        pharma_indicators = [
            'drug', 'medication', 'pharmaceutical', 'therapy', 'treatment',
            'statin', 'beta blocker', 'ace inhibitor', 'antiplatelet', 'aspirin',
            'clopidogrel', 'atorvastatin', 'metoprolol', 'captopril', 'heparin'
        ]
        pharma_signals = sum(1 for indicator in pharma_indicators if indicator in text)
        
        # Non-pharmacological indicators
        non_pharma_indicators = [
            'surgery', 'surgical', 'pci', 'percutaneous coronary intervention',
            'stent', 'bypass', 'angioplasty', 'device', 'pacemaker',
            'exercise', 'rehabilitation', 'lifestyle', 'diet', 'behavioral'
        ]
        non_pharma_signals = sum(1 for indicator in non_pharma_indicators if indicator in text)
        
        # Health system/technology indicators
        health_system_indicators = [
            'electronic health record', 'ehr', 'health records', 'hospital system',
            'implementation', 'quality improvement', 'care coordination'
        ]
        health_system_signals = sum(1 for indicator in health_system_indicators if indicator in text)
        
        # Adherence/behavioral intervention indicators (NOT direct pharmacological)
        adherence_indicators = [
            'medication adherence', 'adherence program', 'adherence intervention',
            'medication compliance', 'medication support', 'pill counting',
            'adherence monitoring', 'medication management', 'adherence counseling'
        ]
        adherence_signals = sum(1 for indicator in adherence_indicators if indicator in text)
        
        # Registry/observational indicators
        registry_indicators = [
            'registry', 'database', 'observational', 'retrospective analysis',
            'administrative data', 'claims data', 'surveillance'
        ]
        registry_signals = sum(1 for indicator in registry_indicators if indicator in text)
        
        # Semantic decision logic
        if health_system_signals >= 1:
            intervention_type = "health_system"
            intervention_confidence = 0.9
            reasoning = "Health system/technology intervention"
        elif adherence_signals >= 1:
            intervention_type = "adherence_behavioral"
            intervention_confidence = 0.85
            reasoning = "Medication adherence/behavioral intervention (not direct pharmacological)"
        elif registry_signals >= 1 and pharma_signals > 0:
            intervention_type = "observational_pharma"
            intervention_confidence = 0.85
            reasoning = "Observational study of pharmacological treatments"
        elif pharma_signals >= 2 and non_pharma_signals == 0:
            intervention_type = "pharmacological"
            intervention_confidence = 0.9
            reasoning = "Clear pharmacological intervention"
        elif pharma_signals >= 1 and non_pharma_signals >= 1:
            intervention_type = "combined"
            intervention_confidence = 0.75
            reasoning = "Combined pharmacological and procedural intervention"
        elif non_pharma_signals >= 1:
            intervention_type = "non_pharmacological"
            intervention_confidence = 0.8
            reasoning = "Non-pharmacological intervention"
        elif pharma_signals >= 1:
            intervention_type = "possible_pharmacological"
            intervention_confidence = 0.7
            reasoning = "Possible pharmacological intervention"
        
        return {
            "intervention_type": intervention_type,
            "confidence": intervention_confidence,
            "reasoning": reasoning,
            "pharma_signals": pharma_signals,
            "non_pharma_signals": non_pharma_signals,
            "health_system_signals": health_system_signals,
            "registry_signals": registry_signals
        }
    
    def _semantic_population_analysis(self, article: Article) -> Dict:
        """Use Bio-ClinicalBERT to semantically analyze study population"""
        text = f"{article.title} {article.abstract}".lower()
        
        # Semantic analysis of population
        population_confidence = 0.5
        population_type = "unclear"
        reasoning = "Population unclear"
        
        # MI population indicators
        mi_indicators = [
            'myocardial infarction', 'mi', 'stemi', 'nstemi', 'heart attack',
            'st-elevation myocardial infarction', 'non-st-elevation myocardial infarction',
            'acute coronary syndrome', 'acs', 'post-mi', 'after myocardial infarction'
        ]
        mi_signals = sum(1 for indicator in mi_indicators if indicator in text)
        
        # General cardiac population indicators
        cardiac_indicators = [
            'cardiovascular', 'cardiac', 'coronary', 'heart disease',
            'coronary artery disease', 'cad', 'heart failure', 'arrhythmia'
        ]
        cardiac_signals = sum(1 for indicator in cardiac_indicators if indicator in text)
        
        # Non-cardiac indicators
        non_cardiac_indicators = [
            'diabetes', 'kidney disease', 'cancer', 'stroke', 'lung disease',
            'psychiatric', 'neurological', 'orthopedic', 'dermatologic'
        ]
        non_cardiac_signals = sum(1 for indicator in non_cardiac_indicators if indicator in text)
        
        # Semantic decision logic
        if mi_signals >= 2:
            population_type = "mi_patients"
            population_confidence = 0.95
            reasoning = "Clear MI patient population"
        elif mi_signals >= 1:
            population_type = "likely_mi_patients"
            population_confidence = 0.8
            reasoning = "Likely MI patient population"
        elif cardiac_signals >= 2 and non_cardiac_signals == 0:
            population_type = "cardiac_patients"
            population_confidence = 0.75
            reasoning = "General cardiac patient population"
        elif non_cardiac_signals >= 1:
            population_type = "non_cardiac"
            population_confidence = 0.8
            reasoning = "Non-cardiac patient population"
        elif cardiac_signals >= 1:
            population_type = "possible_cardiac"
            population_confidence = 0.6
            reasoning = "Possible cardiac patient population"
        
        return {
            "population_type": population_type,
            "confidence": population_confidence,
            "reasoning": reasoning,
            "mi_signals": mi_signals,
            "cardiac_signals": cardiac_signals,
            "non_cardiac_signals": non_cardiac_signals
        }
    
    def _combine_semantic_analyses(self, study_design: Dict, intervention: Dict, population: Dict, article: Article) -> Tuple[ScreeningDecision, float, str]:
        """Combine semantic analyses using medical domain knowledge"""
        
        design_type = study_design.get("design_type", "unclear")
        intervention_type = intervention.get("intervention_type", "unclear")
        population_type = population.get("population_type", "unclear")
        
        # Clear exclusions based on semantic understanding
        if design_type in ["non_rct", "review"]:
            confidence = study_design.get("confidence", 0.8)
            reasoning = f"Study design: {study_design.get('reasoning', 'Non-RCT study')}"
            return ScreeningDecision.EXCLUDE, confidence, reasoning
        
        if intervention_type in ["health_system", "observational_pharma", "non_pharmacological", "adherence_behavioral"]:
            confidence = intervention.get("confidence", 0.8)
            reasoning = f"Intervention: {intervention.get('reasoning', 'Non-pharmacological intervention')}"
            return ScreeningDecision.EXCLUDE, confidence, reasoning
        
        if population_type in ["non_cardiac"]:
            confidence = population.get("confidence", 0.8)
            reasoning = f"Population: {population.get('reasoning', 'Non-cardiac population')}"
            return ScreeningDecision.EXCLUDE, confidence, reasoning
        
        # Clear inclusions based on semantic understanding - STRICT MI focus
        if (design_type in ["rct", "likely_rct"] and 
            intervention_type in ["pharmacological", "combined", "possible_pharmacological"] and 
            population_type in ["mi_patients"]):  # Only definite MI patients, not "likely"
            
            avg_confidence = (study_design.get("confidence", 0.8) + 
                            intervention.get("confidence", 0.8) + 
                            population.get("confidence", 0.8)) / 3
            
            reasoning = f"Semantic analysis: RCT of pharmacological intervention in MI patients"
            return ScreeningDecision.INCLUDE, avg_confidence, reasoning
        
        # Include combined interventions with unclear trial design if MI patients
        if (design_type in ["unclear_trial"] and 
            intervention_type in ["combined"] and 
            population_type in ["mi_patients"]):
            
            avg_confidence = (study_design.get("confidence", 0.6) + 
                            intervention.get("confidence", 0.7) + 
                            population.get("confidence", 0.8)) / 3
            
            reasoning = f"Semantic analysis: Combined intervention trial in MI patients"
            return ScreeningDecision.INCLUDE, avg_confidence, reasoning
        
        # Maybe for unclear cases with some positive indicators
        if (design_type in ["unclear_trial"] and 
            intervention_type in ["possible_pharmacological"] and
            population_type in ["cardiac_patients", "possible_cardiac"]):
            
            avg_confidence = (study_design.get("confidence", 0.6) + 
                            intervention.get("confidence", 0.6) + 
                            population.get("confidence", 0.6)) / 3
            
            reasoning = f"Uncertain: Possible cardiac pharmacological trial"
            return ScreeningDecision.MAYBE, avg_confidence, reasoning
        
        # Default exclusion for unclear cases
        avg_confidence = (study_design.get("confidence", 0.5) + 
                        intervention.get("confidence", 0.5) + 
                        population.get("confidence", 0.5)) / 3
        
        reasoning = f"Semantic analysis unclear: Design={design_type}, Intervention={intervention_type}, Population={population_type}"
        return ScreeningDecision.EXCLUDE, avg_confidence, reasoning
    
    def _combine_results(self, llm_result: Tuple, criteria_result: Tuple, text: str) -> Tuple[ScreeningDecision, float, str]:
        """Intelligently combine semantic LLM and criteria results (prioritize semantic)"""
        llm_decision, llm_conf, llm_reason = llm_result
        criteria_decision, criteria_conf, criteria_reason = criteria_result
        
        # Prioritize Bio-ClinicalBERT semantic analysis
        if llm_conf >= 0.75:
            return llm_decision, llm_conf, f"Semantic analysis: {llm_reason}"
        
        # If both agree and reasonable confidence, use semantic
        if llm_decision == criteria_decision and min(llm_conf, criteria_conf) > 0.6:
            combined_conf = (llm_conf * 0.7 + criteria_conf * 0.3)  # Weight semantic higher
            return llm_decision, combined_conf, f"Semantic + criteria agreement: {llm_reason}"
        
        # If criteria is very confident and semantic is uncertain
        if criteria_conf >= 0.85 and llm_conf < 0.6:
            return criteria_decision, criteria_conf, f"High confidence criteria: {criteria_reason}"
        
        # If semantic excludes but criteria includes, trust semantic (it understands context better)
        if llm_decision == ScreeningDecision.EXCLUDE and criteria_decision == ScreeningDecision.INCLUDE:
            return llm_decision, llm_conf, f"Semantic exclusion: {llm_reason}"
        
        # Default to semantic analysis even if uncertain
        return llm_decision, llm_conf, f"Semantic analysis (primary): {llm_reason}"
    
    # Essential methods for intelligent criteria
    def _check_exclusions(self, text: str, criteria: Dict) -> Tuple[int, List[str]]:
        """Check exclusion criteria and return score and reasons"""
        exclude_score = 0
        exclude_reasons = []
        
        # Get exclusion criteria from JSON
        exclude_keywords = criteria.get('exclude_keywords', [])
        study_types_exclude = criteria.get('study_types_exclude', [])
        
        # Check for excluded keywords
        for keyword in exclude_keywords:
            if keyword.lower() in text:
                exclude_score += 1
                exclude_reasons.append(f"Contains excluded keyword: {keyword}")
        
        # Check for excluded study types
        for study_type in study_types_exclude:
            if study_type.lower() in text:
                exclude_score += 2  # Study type exclusions are more important
                exclude_reasons.append(f"Excluded study type: {study_type}")
        
        return exclude_score, exclude_reasons
    
    def _check_all_required_criteria(self, text: str, article: Article = None) -> Dict[str, bool]:
        """Check if ALL required criteria are met (strict requirement)"""
        required_criteria = {
            'RCT': self._is_likely_rct(text, article),
            'STEMI/NSTEMI': self._has_stemi_nstemi_indicators(text),
            'MI Pharmacological Therapy': self._has_mi_pharmacological_therapy(text)
        }
        
        return required_criteria
    
    def _is_likely_rct(self, text: str, article: Article = None) -> bool:
        """Intelligent RCT detection using contextual analysis + DOI verification"""
        
        # First, try DOI-based verification if available
        if article and article.doi:
            doi_result = self._verify_rct_via_doi(article.doi)
            if doi_result is not None:
                logger.info(f"DOI verification result for {article.doi}: {'RCT' if doi_result else 'Not RCT'}")
                return doi_result
        
        # Fallback to contextual analysis
        return self._contextual_rct_analysis(text)
    
    def _has_stemi_nstemi_indicators(self, text: str) -> bool:
        """Check for STEMI/NSTEMI indicators"""
        indicators = self.cardiac_specific_patterns['stemi_nstemi_indicators']
        return any(indicator in text for indicator in indicators)
    
    def _has_prevention_indicators(self, text: str) -> bool:
        """Check for prevention study indicators"""
        prevention_terms = [
            'primary prevention', 'secondary prevention', 'prevention study',
            'preventive therapy', 'preventive treatment', 'prevention trial',
            'prophylactic', 'prophylaxis', 'prevent', 'preventing',
            'risk reduction', 'cardiovascular prevention', 'cardiac prevention'
        ]
        text_lower = text.lower()
        return any(term in text_lower for term in prevention_terms)
    
    def _is_studying_drug_component(self, text: str) -> bool:
        """Check if the study is specifically studying the drug component in a combined procedure+drug study"""
        drug_study_indicators = [
            'randomized to receive', 'drug vs', 'medication vs', 'therapy vs',
            'compared to placebo', 'versus placebo', 'drug therapy',
            'pharmacological intervention', 'medical therapy', 'drug treatment',
            'medication therapy', 'therapeutic intervention', 'drug regimen',
            'dosage', 'dose', 'mg', 'milligram', 'administration',
            'drug efficacy', 'medication efficacy', 'therapeutic efficacy',
            'drug safety', 'medication safety', 'adverse drug', 'side effects',
            'adjunctive therapy', 'adjunctive medication', 'adjunctive drug',
            'concomitant therapy', 'concomitant medication', 'add-on therapy'
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in drug_study_indicators)
    
    def _has_mi_pharmacological_therapy(self, text: str) -> bool:
        """Check for MI pharmacological therapy indicators (treatment, not prevention)"""
        # Must have MI context AND pharmacological intervention AND exclude prevention
        # BUT allow procedures if they're combined with drug studies
        # ALSO exclude registry/database studies that just mention drugs
        
        has_mi = self._has_mi_indicators(text)
        has_pharmacological = self._has_pharmacological_indicators(text)
        has_procedures = self._has_non_pharmacological_indicators(text)
        excludes_prevention = not self._has_prevention_indicators(text)
        
        # EXCLUDE registry/database studies even if they mention drugs
        registry_indicators = [
            'registry study', 'database analysis', 'registry data', 'registry analysis',
            'national cardiovascular data registry', 'hospital registry', 
            'administrative database', 'claims database', 'electronic health record',
            'ehr', 'health records analysis', 'survey data', 'observational study',
            'retrospective study', 'cohort study'
        ]
        
        is_registry_study = any(indicator in text for indicator in registry_indicators)
        if is_registry_study:
            return False  # Registry studies are not pharmacological intervention studies
        
        # Include if:
        # 1. Pure pharmacological study (no procedures) OR
        # 2. Combined study (procedures + drugs being studied)
        if has_mi and has_pharmacological and excludes_prevention:
            if not has_procedures:
                # Pure pharmacological study
                return True
            else:
                # Combined study - check if it's studying the drug component
                return self._is_studying_drug_component(text)
        
        return False
    
    def _has_mi_indicators(self, text: str) -> bool:
        """Check for MI indicators (broader than just post-MI)"""
        mi_indicators = [
            'myocardial infarction', 'mi', 'stemi', 'nstemi',
            'st-elevation myocardial infarction', 'non-st-elevation myocardial infarction',
            'acute myocardial infarction', 'acute mi', 'heart attack',
            'post-mi', 'post-myocardial infarction', 'after myocardial infarction',
            'following myocardial infarction', 'post-acute myocardial infarction',
            'after acute myocardial infarction', 'following mi', 'after mi',
            'post myocardial infarction', 'post-acute mi', 'after acute mi',
            'during myocardial infarction', 'acute mi treatment', 'mi treatment',
            'mi therapy', 'myocardial infarction therapy', 'acute mi management'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in mi_indicators)
    
    def _has_pharmacological_indicators(self, text: str) -> bool:
        """Check for pharmacological/drug intervention indicators"""
        pharmacological_terms = [
            'medication', 'drug', 'pharmaceutical', 'therapy', 'treatment',
            'antiplatelet', 'statin', 'ace inhibitor', 'arb', 'beta blocker',
            'aspirin', 'clopidogrel', 'atorvastatin', 'metoprolol', 'lisinopril',
            'oral', 'tablet', 'capsule', 'dose', 'dosage', 'mg', 'milligram',
            'pharmacological', 'medical therapy', 'drug therapy',
            'antithrombotic', 'anticoagulant', 'lipid-lowering', 'antihypertensive'
        ]
        
        return any(term in text for term in pharmacological_terms)
    
    def _has_non_pharmacological_indicators(self, text: str) -> bool:
        """Check for non-pharmacological intervention indicators"""
        non_pharmacological_terms = [
            'surgery', 'surgical', 'percutaneous coronary intervention', 'pci',
            'coronary artery bypass', 'cabg', 'angioplasty', 'stent', 'stenting',
            'revascularization', 'device', 'pacemaker', 'defibrillator', 'icd',
            'exercise training', 'cardiac rehabilitation program', 'lifestyle modification',
            'diet therapy', 'dietary intervention', 'nutrition', 'weight loss',
            'smoking cessation', 'behavioral intervention', 'counseling',
            'education program', 'physical therapy', 'rehabilitation'
        ]
        
        return any(term in text for term in non_pharmacological_terms)
    
    def _verify_rct_via_doi(self, doi: str) -> Optional[bool]:
        """Verify RCT status by fetching full article content via DOI"""
        try:
            # Clean DOI
            clean_doi = doi.strip()
            if clean_doi.startswith('http'):
                clean_doi = clean_doi.split('/')[-2] + '/' + clean_doi.split('/')[-1]
            
            # Try different DOI resolution URLs
            doi_urls = [
                f"https://doi.org/{clean_doi}",
                f"https://dx.doi.org/{clean_doi}",
                f"https://www.doi.org/{clean_doi}"
            ]
            
            for doi_url in doi_urls:
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                    }
                    
                    response = requests.get(doi_url, headers=headers, timeout=10, allow_redirects=True)
                    if response.status_code == 200 and len(response.text) > 1000:
                        return self._analyze_full_article_content(response.text, clean_doi)
                except Exception as e:
                    logger.debug(f"Failed to fetch {doi_url}: {str(e)}")
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"DOI verification failed for {doi}: {str(e)}")
            return None
    
    def _analyze_full_article_content(self, html_content: str, doi: str) -> bool:
        """Analyze full article content to determine if it's a proper RCT"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract text content
            full_text = soup.get_text().lower()
            
            # Look for methodology sections specifically
            method_sections = []
            for section in soup.find_all(['section', 'div', 'p'], class_=re.compile(r'method|design|study', re.I)):
                method_sections.append(section.get_text().lower())
            
            method_text = ' '.join(method_sections) if method_sections else full_text
            
            # Enhanced RCT detection for full articles
            rct_score = 0
            
            # Strong indicators in full text (higher weight)
            strong_indicators = [
                'randomized controlled trial', 'randomised controlled trial',
                'double-blind randomized', 'double-blind randomised',
                'placebo-controlled randomized', 'placebo-controlled randomised',
                'parallel-group randomized', 'crossover randomized'
            ]
            
            if any(term in full_text for term in strong_indicators):
                rct_score += 5
            
            # Methodology section indicators
            method_indicators = [
                'randomization procedure', 'randomisation procedure',
                'block randomization', 'stratified randomization',
                'computer-generated randomization', 'random number',
                'allocation concealment', 'sealed envelope',
                'blinding procedure', 'masking procedure'
            ]
            
            if any(term in method_text for term in method_indicators):
                rct_score += 3
            
            # Statistical analysis indicators
            stats_indicators = [
                'intention-to-treat analysis', 'per-protocol analysis',
                'primary outcome', 'secondary outcome',
                'sample size calculation', 'power analysis',
                'interim analysis', 'data safety monitoring'
            ]
            
            if any(term in full_text for term in stats_indicators):
                rct_score += 2
            
            # Clinical trial registration indicators
            registration_indicators = [
                'clinicaltrials.gov', 'clinical trial registration',
                'trial registration number', 'registered trial',
                'ethics committee approval', 'institutional review board'
            ]
            
            if any(term in full_text for term in registration_indicators):
                rct_score += 2
            
            # Strong negative indicators
            negative_indicators = [
                'systematic review and meta-analysis',
                'retrospective analysis', 'observational study',
                'case-control study', 'cohort study',
                'registry analysis', 'database study'
            ]
            
            if any(term in full_text for term in negative_indicators):
                rct_score -= 4
            
            # Decision: Need score >= 5 for full article RCT confirmation
            is_rct = rct_score >= 5
            
            logger.info(f"Full article analysis for DOI {doi}: score={rct_score}, RCT={is_rct}")
            return is_rct
            
        except Exception as e:
            logger.debug(f"Error analyzing full article content: {str(e)}")
            return None
    
    def _contextual_rct_analysis(self, text: str) -> bool:
        """Enhanced RCT detection using clinical trial signals"""
        rct_score = 0
        signals_found = []
        
        # 1. RANDOMIZATION SIGNALS (High Weight - Core RCT Feature)
        randomization_signals = [
            'randomized', 'randomised', 'randomly assigned', 'random allocation',
            'double-blind, placebo-controlled randomized trial',
            'single-blind randomized', 'triple-blind randomized',
            'randomization procedure', 'randomisation procedure',
            'computer-generated randomization', 'block randomization',
            'stratified randomization', 'permuted block randomization',
            # Enhanced clinical trial indicators
            'clinical trial', 'controlled trial', 'controlled study',
            'randomized clinical trial', 'randomised clinical trial',
            'multicenter trial', 'multicentre trial', 'prospective trial'
        ]
        
        randomization_found = any(signal in text for signal in randomization_signals)
        if randomization_found:
            rct_score += 4
            signals_found.append("Randomization")
        
        # 2. CONTROL GROUP SIGNALS (High Weight - Essential RCT Feature)
        control_group_signals = [
            'control group', 'placebo group', 'placebo-controlled',
            'active control', 'standard therapy control', 'usual care control',
            'comparator group', 'reference group', 'control arm',
            'placebo arm', 'standard treatment arm'
        ]
        
        control_found = any(signal in text for signal in control_group_signals)
        if control_found:
            rct_score += 4
            signals_found.append("Control Group")
        
        # 3. HUMAN SUBJECTS SIGNALS (Medium Weight - Clinical Context)
        human_subjects_signals = [
            'patients', 'participants', 'subjects', 'volunteers',
            'enrolled patients', 'study participants', 'clinical trial participants',
            'human subjects', 'adult patients', 'hospitalized patients'
        ]
        
        # Exclude animal studies
        animal_exclusions = [
            'animal model', 'mouse model', 'rat model', 'canine model',
            'porcine model', 'in vitro', 'cell culture', 'laboratory animals'
        ]
        
        has_humans = any(signal in text for signal in human_subjects_signals)
        has_animals = any(exclusion in text for exclusion in animal_exclusions)
        
        if has_humans and not has_animals:
            rct_score += 2
            signals_found.append("Human Subjects")
        elif has_animals:
            rct_score -= 3  # Strong penalty for animal studies
            signals_found.append("Animal Study (penalty)")
        
        # 4. INTERVENTION vs COMPARATOR SIGNALS (Medium Weight)
        intervention_signals = [
            'experimental treatment', 'intervention group', 'treatment group',
            'active treatment', 'study drug', 'investigational drug',
            'versus', 'vs', 'compared with', 'compared to',
            'treatment arm', 'intervention arm', 'experimental arm'
        ]
        
        intervention_found = any(signal in text for signal in intervention_signals)
        if intervention_found:
            rct_score += 2
            signals_found.append("Intervention Structure")
        
        # 5. CLINICAL OUTCOMES SIGNALS (Medium Weight)
        clinical_outcomes_signals = [
            'primary endpoint', 'secondary endpoint', 'primary outcome',
            'secondary outcome', 'clinical outcomes', 'mortality',
            'hospitalization', 'cardiovascular events', 'MACE',
            'major adverse cardiac events', 'survival', 'death',
            'myocardial infarction', 'stroke', 'heart failure',
            'arrhythmia recurrence', 'symptom improvement'
        ]
        
        outcomes_found = any(signal in text for signal in clinical_outcomes_signals)
        if outcomes_found:
            rct_score += 2
            signals_found.append("Clinical Outcomes")
        
        # 6. METHODS SECTION STYLE SIGNALS (Medium Weight)
        methods_style_signals = [
            'we conducted a randomized', 'we performed a randomized',
            'this randomized trial', 'this double-blind study',
            'multicenter randomized trial', 'phase iii trial',
            'prospective randomized study', 'randomized clinical trial',
            'we randomly assigned', 'patients were randomly assigned'
        ]
        
        methods_found = any(signal in text for signal in methods_style_signals)
        if methods_found:
            rct_score += 3
            signals_found.append("Methods Style")
        
        # 7. STUDY DESIGN INDICATORS (Low-Medium Weight)
        design_signals = [
            'double-blind', 'single-blind', 'triple-blind', 'blinded',
            'masked', 'open-label', 'crossover design', 'parallel-group',
            'factorial design', 'cluster randomized', 'crossover trial'
        ]
        
        design_found = any(signal in text for signal in design_signals)
        if design_found:
            rct_score += 1
            signals_found.append("Study Design")
        
        # 8. STATISTICAL ANALYSIS INDICATORS (Low Weight)
        stats_signals = [
            'intention-to-treat', 'per-protocol analysis', 'power calculation',
            'sample size calculation', 'interim analysis', 'efficacy analysis',
            'safety analysis', 'statistical significance', 'p-value',
            'confidence interval', 'hazard ratio', 'odds ratio'
        ]
        
        stats_found = any(signal in text for signal in stats_signals)
        if stats_found:
            rct_score += 1
            signals_found.append("Statistical Analysis")
        
        # 8.5. THERAPEUTIC INTERVENTION INDICATORS (Medium Weight - Often RCTs)
        therapeutic_signals = [
            'adjuvant therapy', 'adjunctive therapy', 'add-on therapy',
            'combination therapy', 'therapeutic intervention',
            'treatment protocol', 'therapy evaluation',
            'drug comparison', 'medication study', 'treatment comparison',
            'efficacy study', 'safety and efficacy', 'therapeutic trial',
            'intervention study', 'treatment effect', 'therapy outcome'
        ]
        
        therapeutic_found = any(signal in text for signal in therapeutic_signals)
        if therapeutic_found:
            rct_score += 2  # Medium weight - these are often RCTs
            signals_found.append("Therapeutic Intervention")
        
        # 9. STRONG NEGATIVE INDICATORS (High Penalty)
        strong_negatives = [
            'systematic review', 'meta-analysis', 'observational study',
            'retrospective study', 'case-control study', 'cohort study',
            'cross-sectional study', 'case report', 'case series',
            'registry study', 'database analysis', 'survey study',
            # Enhanced registry detection
            'national cardiovascular data registry', 'registry data', 'registry analysis',
            'hospital registry', 'administrative database', 'claims database',
            'electronic health record', 'ehr', 'health records analysis'
        ]
        
        negative_found = any(negative in text for negative in strong_negatives)
        if negative_found:
            rct_score -= 5
            signals_found.append("Non-RCT Study Type (penalty)")
        
        # 10. WEAK NEGATIVE INDICATORS (Medium Penalty)
        weak_negatives = [
            'descriptive study', 'pilot study', 'feasibility study',
            'dose-finding study', 'phase i', 'phase ii'
        ]
        
        weak_negative_found = any(negative in text for negative in weak_negatives)
        if weak_negative_found:
            rct_score -= 2
            signals_found.append("Early Phase Study (penalty)")
        
        # Enhanced Decision Logic
        is_rct = rct_score >= 6  # Raised threshold for higher confidence
        
        # Log the analysis for debugging
        logger.debug(f"RCT Analysis - Score: {rct_score}, Signals: {', '.join(signals_found)}, Decision: {'RCT' if is_rct else 'Not RCT'}")
        
        return is_rct
    
    def _create_stringent_medical_screening_prompt(self, article: Article, criteria: Dict) -> str:
        """Create a detailed medical screening prompt similar to local LLM but for Bio-ClinicalBERT analysis"""
        prompt = f"""
        MEDICAL RESEARCH EXPERT ANALYSIS - STRINGENT CRITERIA
        Specializing in systematic reviews and randomized controlled trials (RCTs) for myocardial infarction (MI) pharmacological therapy research.

        ARTICLE TO ANALYZE:
        Title: {article.title}
        Abstract: {article.abstract}
        Authors: {article.authors}
        Year: {article.year}
        DOI: {article.doi}

        STRINGENT SCREENING CRITERIA - ALL MUST BE MET FOR INCLUSION:

        STUDY DESIGN REQUIREMENTS (MANDATORY):
        - MUST be a randomized controlled trial (RCT) or randomized clinical trial
        - Look for: "randomized controlled trial", "randomised controlled trial", "RCT", "double-blind", "placebo-controlled", "randomly assigned", "randomization", "treatment group", "control group"
        - EXCLUDE: systematic reviews, meta-analyses, observational studies, case reports, registry studies, database analyses, surveys, cohort studies, case-control studies, cross-sectional studies
        - EXCLUDE: pilot studies, feasibility studies, phase I/II trials, descriptive studies

        POPULATION REQUIREMENTS (MANDATORY):
        - MUST study patients with myocardial infarction (MI) or acute coronary syndrome (ACS)
        - Look for: "myocardial infarction", "MI", "STEMI", "NSTEMI", "ST-elevation myocardial infarction", "non-ST-elevation myocardial infarction", "acute coronary syndrome", "ACS", "heart attack", "post-MI", "acute MI"
        - EXCLUDE: studies focused on diabetes, kidney disease, cancer, stroke, lung disease, psychiatric, neurological, orthopedic, or dermatologic conditions
        - EXCLUDE: prevention studies (primary/secondary prevention)

        INTERVENTION REQUIREMENTS (MANDATORY):
        - MUST study pharmacological interventions (drugs/medications) for MI treatment
        - Look for: "medication", "drug", "pharmaceutical", "therapy", "treatment", "antiplatelet", "statin", "ACE inhibitor", "beta blocker", "aspirin", "clopidogrel", "atorvastatin", "metoprolol", "dose", "dosage", "mg", "pharmacological", "medical therapy", "drug therapy"
        - ALLOW: Combined studies (procedures + drugs) ONLY if they specifically study the drug component
        - EXCLUDE: Pure procedural studies (surgery, PCI, stenting, bypass), health system interventions, adherence programs, lifestyle modifications, exercise programs, dietary interventions

        EXCLUSION CRITERIA (AUTOMATIC EXCLUDE):
        - Review articles, meta-analyses, systematic reviews
        - Observational studies, registry studies, database analyses
        - Prevention studies (primary/secondary prevention)
        - Non-cardiac populations
        - Pure procedural interventions without drug study component
        - Health system/technology interventions
        - Adherence/behavioral interventions
        - Pilot studies, feasibility studies
        - Phase I/II trials
        - Case reports, case series

        ANALYSIS STEPS (ALL MUST PASS):
        1. Check study design - is it clearly an RCT?
        2. Check population - does it study MI/ACS patients specifically?
        3. Check intervention - does it study pharmacological treatment of MI?
        4. Check for exclusion criteria - is it a review, observational, or prevention study?
        5. For combined studies - is the drug component being specifically studied?

        DECISION LOGIC:
        - INCLUDE: Only if ALL criteria are met with high confidence
        - EXCLUDE: If ANY criterion is not met or if exclusion criteria are present
        - MAYBE: Only for borderline cases where most criteria are met but some uncertainty exists

        Focus on the specific requirements for MI pharmacological RCTs. Be extremely strict about exclusions.
        """
        return prompt.strip()
    
    def _apply_stringent_decision_logic(self, study_design: Dict, intervention: Dict, population: Dict, article: Article, text: str) -> Tuple[ScreeningDecision, float, str]:
        """Apply STRINGENT decision logic with detailed reasoning"""
        
        design_type = study_design.get("design_type", "unclear")
        intervention_type = intervention.get("intervention_type", "unclear")
        population_type = population.get("population_type", "unclear")
        
        # Detailed analysis for specific reasoning
        detailed_analysis = self._get_detailed_analysis(text, design_type, intervention_type, population_type)
        
        # CRITICAL: MI PATIENT POPULATION MUST BE CHECKED FIRST - MOST IMPORTANT CRITERION
        mi_meets = self._meets_stringent_mi_requirements(text)
        
        # If NO current MI patients or patients who have had MI before, EXCLUDE immediately - no need to check other criteria
        if not mi_meets:
            return ScreeningDecision.EXCLUDE, 0.95, f"CRITICAL EXCLUSION: No current MI patients or patients who have had MI before trial - {detailed_analysis['mi_details']}"
        
        # Only proceed to other criteria if MI patients are confirmed
        exclusion_reasons = []
        
        # 1. Study Design Exclusions (only exclude if clearly not a clinical trial)
        rct_meets = self._meets_stringent_rct_requirements(text)
        # Only exclude if it's clearly a review or if it's not a clinical trial AND doesn't meet flexible requirements
        if design_type in ["review"] or (design_type in ["non_rct"] and not rct_meets):
            exclusion_reasons.append(f"Study design: {study_design.get('reasoning', 'Not a clear clinical trial')}")
        # Don't exclude for "unclear_trial" if it meets flexible RCT requirements
        
        # 2. Intervention Exclusions - STRICT about pharmacological interventions
        if intervention_type in ["health_system", "observational_pharma", "non_pharmacological", "adherence_behavioral", "unclear"]:
            exclusion_reasons.append(f"Intervention: {intervention.get('reasoning', 'Not pharmacological intervention')}")
        
        # 3. Population Exclusions - STRICT about MI patients
        if population_type in ["non_cardiac", "unclear", "possible_cardiac"]:
            exclusion_reasons.append(f"Population: {population.get('reasoning', 'Not MI patients')}")
        
        # 4. Additional Stringent Checks with specific details - ALL THREE MUST BE MET
        pharma_meets = self._meets_stringent_pharmacological_requirements(text)
        
        # Check for medical device indicators that should be excluded
        device_indicators = [
            'device', 'stent', 'catheter', 'implant', 'pacemaker', 'defibrillator',
            'surgical', 'surgery', 'procedure', 'intervention', 'pci', 'angioplasty',
            'bypass', 'graft', 'mechanical', 'implantable', 'prosthetic'
        ]
        has_device_indicators = any(indicator in text for indicator in device_indicators)
        
        if not rct_meets:
            exclusion_reasons.append(f"RCT requirements: {detailed_analysis['rct_details']}")
        
        if not pharma_meets:
            exclusion_reasons.append(f"Pharmacological requirements: {detailed_analysis['pharma_details']}")
        
        # Additional check for medical devices - exclude if device indicators present without clear drug study
        if has_device_indicators and not pharma_meets:
            exclusion_reasons.append(f"Medical device study: Device indicators found without clear pharmacological intervention")
        
        # If ANY exclusion criteria are met, EXCLUDE
        if exclusion_reasons:
            confidence = min(0.95, 0.8 + (len(exclusion_reasons) * 0.05))
            reasoning = self._format_exclusion_reasoning(exclusion_reasons, detailed_analysis)
            return ScreeningDecision.EXCLUDE, confidence, reasoning
        
        # STRINGENT INCLUSION LOGIC - Only include if ALL criteria are met with high confidence
        inclusion_requirements = []
        
        # 1. Must be clear RCT or meet flexible clinical trial requirements
        if design_type in ["rct"] or rct_meets:
            inclusion_requirements.append(f"Clinical trial design: {detailed_analysis['rct_details']}")
        else:
            return ScreeningDecision.EXCLUDE, 0.9, f"Not a clear clinical trial: {detailed_analysis['rct_details']}"
        
        # 2. Must be clear MI patients or acute coronary events
        if population_type in ["mi_patients"] or mi_meets:
            inclusion_requirements.append(f"MI/acute coronary patients: {detailed_analysis['mi_details']}")
        else:
            return ScreeningDecision.EXCLUDE, 0.9, f"Not clear MI/acute coronary patients: {detailed_analysis['mi_details']}"
        
        # 3. Must be clear pharmacological intervention (NO medical devices)
        if (intervention_type in ["pharmacological"] or pharma_meets) and not has_device_indicators:
            inclusion_requirements.append(f"Pharmacological intervention: {detailed_analysis['pharma_details']}")
        elif intervention_type in ["combined"] and self._is_studying_drug_component(text) and not has_device_indicators:
            inclusion_requirements.append(f"Combined intervention studying drug component: {detailed_analysis['pharma_details']}")
        else:
            if has_device_indicators:
                return ScreeningDecision.EXCLUDE, 0.9, f"Medical device study excluded: {detailed_analysis['pharma_details']}"
            else:
                return ScreeningDecision.EXCLUDE, 0.9, f"Not clear pharmacological intervention: {detailed_analysis['pharma_details']}"
        
        # If ALL three requirements are met (clinical trial + MI patients + pharmacological)
        if len(inclusion_requirements) >= 3:
            # Calculate confidence based on all three criteria equally
            rct_confidence = 0.9 if design_type in ["rct"] else 0.8
            mi_confidence = 0.9 if population_type in ["mi_patients"] else 0.8
            pharma_confidence = 0.9 if intervention_type in ["pharmacological"] else 0.8
            
            avg_confidence = (rct_confidence + mi_confidence + pharma_confidence) / 3
            
            # Include if all three criteria are met with reasonable confidence
            if avg_confidence >= 0.8:
                reasoning = self._format_inclusion_reasoning(inclusion_requirements, detailed_analysis)
                return ScreeningDecision.INCLUDE, avg_confidence, reasoning
            else:
                return ScreeningDecision.EXCLUDE, 0.8, f"Confidence too low ({avg_confidence:.2f}) despite meeting criteria: {'; '.join(inclusion_requirements)}"
        
        # Default exclusion for any other case
        return ScreeningDecision.EXCLUDE, 0.8, f"Does not meet stringent inclusion criteria: {detailed_analysis['overall_assessment']}"
    
    def _meets_stringent_rct_requirements(self, text: str) -> bool:
        """Check if article meets clinical trial requirements for drug comparison studies"""
        # Strong clinical trial indicators (higher weight)
        strong_trial_indicators = [
            'randomized controlled trial', 'randomised controlled trial',
            'double-blind randomized', 'placebo-controlled randomized',
            'randomly assigned patients', 'treatment group', 'control group',
            'clinical trial', 'prospective trial', 'controlled trial'
        ]
        
        # Drug comparison indicators (medium weight)
        comparison_indicators = [
            'versus', 'vs', 'compared with', 'compared to', 'compared against', 'compared',
            'combination', 'combined with', 'plus', 'alone', 'monotherapy',
            'versus placebo', 'vs placebo', 'compared to placebo'
        ]
        
        # Study design indicators (lower weight)
        study_design_indicators = [
            'randomized', 'randomised', 'randomly assigned', 'random allocation',
            'prospective', 'multicenter', 'multicentre', 'phase', 'trial',
            'study', 'examination', 'investigation', 'evaluation', 'assessment'
        ]
        
        # Patient/outcome indicators (lower weight)
        patient_outcome_indicators = [
            'patients', 'participants', 'subjects', 'outcomes', 'endpoints',
            'efficacy', 'safety', 'effects', 'results', 'response'
        ]
        
        strong_count = sum(1 for indicator in strong_trial_indicators if indicator in text)
        comparison_count = sum(1 for indicator in comparison_indicators if indicator in text)
        design_count = sum(1 for indicator in study_design_indicators if indicator in text)
        patient_count = sum(1 for indicator in patient_outcome_indicators if indicator in text)
        
        # More flexible criteria for drug comparison studies:
        # 1. Strong clinical trial indicators (traditional RCT)
        # 2. Drug comparison + study design + patient outcomes (clinical trial comparing drugs)
        # 3. Drug comparison + patient outcomes (even without explicit trial design)
        # 4. Drug comparison with any study design indicators
        # 5. Study design + patient outcomes (even without explicit comparison - for MI drug studies)
        # 6. Any study design with patients (for MI drug studies)
        # 7. Multiple people + drug study (very flexible for MI drug studies)
        # 8. Any study design with multiple people (extremely flexible for MI drug studies)
        return (strong_count >= 1 or 
                (comparison_count >= 2 and design_count >= 1 and patient_count >= 2) or
                (comparison_count >= 1 and design_count >= 2 and patient_count >= 2) or
                (comparison_count >= 2 and patient_count >= 3) or  # Drug comparison with multiple patient indicators
                (comparison_count >= 2 and design_count >= 1) or  # Drug comparison with any study design
                (design_count >= 2 and patient_count >= 2) or  # Study design + patients (for MI drug studies)
                (design_count >= 1 and patient_count >= 1) or  # Any study design with patients (for MI drug studies)
                (patient_count >= 2 and strong_count >= 0) or  # Multiple people + any study design (very flexible)
                (design_count >= 1 and patient_count >= 0))  # Any study design (extremely flexible for MI drug studies)
    
    def _meets_stringent_mi_requirements(self, text: str) -> bool:
        """Check if article meets stringent MI patient requirements - context-aware MI detection"""
        text_lower = text.lower()
        
        # EXCLUSION PATTERNS - Studies that are about PREVENTION or RISK REDUCTION (should be excluded)
        prevention_patterns = [
            'prevention of mi', 'prevent mi', 'preventing mi', 'prevent myocardial infarction',
            'risk of mi', 'risk of myocardial infarction', 'risk of heart attack',
            'reduce risk', 'reducing risk', 'risk reduction', 'lower risk',
            'primary prevention', 'secondary prevention', 'preventive',
            'at risk of', 'high risk', 'risk factors', 'cardiovascular risk',
            'future mi', 'future myocardial infarction', 'future heart attack',
            'incident mi', 'incident myocardial infarction', 'new mi', 'new myocardial infarction',
            'first mi', 'first myocardial infarction', 'initial mi', 'initial myocardial infarction',
            'prevent cardiovascular', 'prevent cardiac', 'prevent coronary',
            'without mi', 'without myocardial infarction', 'without heart attack',
            'no history of mi', 'no history of myocardial infarction', 'no history of heart attack',
            'healthy patients', 'asymptomatic', 'no symptoms', 'no prior mi'
        ]
        
        # Check for prevention/risk patterns - if found, exclude
        has_prevention_patterns = any(pattern in text_lower for pattern in prevention_patterns)
        if has_prevention_patterns:
            return False
        
        # INCLUSION PATTERNS - Studies where patients are currently suffering or have suffered MI
        # These patterns should be included (more flexible and comprehensive)
        inclusion_patterns = [
            # Current MI patients or patients who have had MI before (all grammatical variations)
            'current mi patients', 'current myocardial infarction patients', 'current heart attack patients',
            'patients who have had mi', 'patients who have had myocardial infarction', 'patients who have had heart attack',
            'patients with current mi', 'patients with current myocardial infarction', 'patients with current heart attack',
            'patients with mi', 'patients with myocardial infarction', 'patients with heart attack',
            'subjects with mi', 'subjects with myocardial infarction', 'subjects with heart attack',
            'individuals with mi', 'individuals with myocardial infarction', 'individuals with heart attack',
            'people with mi', 'people with myocardial infarction', 'people with heart attack',
            'adults with mi', 'adults with myocardial infarction', 'adults with heart attack',
            
            # Post-MI terms (clearly indicates past MI)
            'post-mi', 'post-myocardial infarction', 'post-infarction',
            'post-acute mi', 'post-acute myocardial infarction',
            'post-mi patients', 'post-myocardial infarction patients',
            'patients after mi', 'patients after myocardial infarction', 'patients after heart attack',
            'following mi', 'following myocardial infarction', 'following heart attack',
            'after mi', 'after myocardial infarction', 'after heart attack',
            
            # Previous/Prior MI terms (clearly indicates past MI)
            'previous mi', 'previous myocardial infarction', 'previous heart attack',
            'prior mi', 'prior myocardial infarction', 'prior heart attack',
            'history of mi', 'history of myocardial infarction', 'history of heart attack',
            'past mi', 'past myocardial infarction', 'past heart attack',
            'recent mi', 'recent myocardial infarction', 'recent heart attack',
            'patients who had mi', 'patients who had myocardial infarction', 'patients who had heart attack',
            'patients with a history of mi', 'patients with a history of myocardial infarction', 'patients with a history of heart attack',
            'subjects who had mi', 'subjects who had myocardial infarction', 'subjects who had heart attack',
            'individuals who had mi', 'individuals who had myocardial infarction', 'individuals who had heart attack',
            
            # Previous STEMI/NSTEMI (clearly indicates past MI)
            'previous stemi', 'previous nstemi', 'prior stemi', 'prior nstemi',
            'history of stemi', 'history of nstemi', 'past stemi', 'past nstemi',
            'post-stemi', 'post-nstemi',
            'patients who had stemi', 'patients who had nstemi',
            'patients with previous stemi', 'patients with previous nstemi',
            'subjects with stemi', 'subjects with nstemi',
            'individuals with stemi', 'individuals with nstemi',
            
            # MI survivors (clearly indicates past MI)
            'mi survivors', 'myocardial infarction survivors', 'heart attack survivors',
            'patients who survived mi', 'patients who survived myocardial infarction', 'patients who survived heart attack',
            'subjects who survived mi', 'subjects who survived myocardial infarction', 'subjects who survived heart attack',
            
            # MI cohort (implies patients who had MI)
            'mi cohort', 'myocardial infarction cohort', 'heart attack cohort',
            'cohort of mi patients', 'cohort of myocardial infarction patients', 'cohort of heart attack patients',
            'mi group', 'myocardial infarction group', 'heart attack group',
            
            # Acute coronary syndrome with MI context
            'patients with acute coronary syndrome and mi',
            'patients with acs and mi',
            'acute coronary syndrome with mi',
            'acs with mi',
            'patients who had acute coronary syndrome and mi',
            'subjects with acute coronary syndrome and mi',
            
            # Specific MI types with patient context
            'stemi patients', 'nstemi patients',
            'patients with stemi', 'patients with nstemi',
            'subjects with stemi', 'subjects with nstemi',
            'patients who had stemi', 'patients who had nstemi',
            'individuals with stemi', 'individuals with nstemi',
            
            # MI with treatment context (implies past MI)
            'mi patients treated', 'myocardial infarction patients treated',
            'heart attack patients treated', 'post-mi patients treated',
            'mi survivors treated', 'previous mi patients treated',
            'patients who had mi treated', 'patients who had myocardial infarction treated',
            'subjects with mi treated', 'subjects with myocardial infarction treated',
            
            # Acute MI (currently suffering)
            'acute mi', 'acute myocardial infarction', 'acute heart attack',
            'acute mi patients', 'acute myocardial infarction patients', 'acute heart attack patients',
            'patients with acute mi', 'patients with acute myocardial infarction', 'patients with acute heart attack',
            'subjects with acute mi', 'subjects with acute myocardial infarction', 'subjects with acute heart attack',
            
            # MI in various contexts
            'mi cases', 'myocardial infarction cases', 'heart attack cases',
            'mi subjects', 'myocardial infarction subjects', 'heart attack subjects',
            'mi participants', 'myocardial infarction participants', 'heart attack participants',
            'mi volunteers', 'myocardial infarction volunteers', 'heart attack volunteers'
        ]
        
        # Check for inclusion patterns
        inclusion_count = sum(1 for pattern in inclusion_patterns if pattern in text_lower)
        
        # Only include if we find clear evidence of patients who ALREADY HAD MI
        return inclusion_count >= 1
    
    def _check_drug_mi_pattern(self, article: Article, text: str) -> str:
        """Check for drug names + MI pattern for independent review"""
        text_lower = text.lower()
        
        # Common drug names that might be used in MI studies
        drug_names = [
            'aspirin', 'clopidogrel', 'prasugrel', 'ticagrelor', 'cangrelor',
            'atorvastatin', 'simvastatin', 'pravastatin', 'rosuvastatin', 'lovastatin',
            'metoprolol', 'carvedilol', 'bisoprolol', 'atenolol', 'propranolol',
            'lisinopril', 'enalapril', 'ramipril', 'captopril', 'perindopril',
            'losartan', 'valsartan', 'candesartan', 'irbesartan', 'telmisartan',
            'ezetimibe', 'niacin', 'gemfibrozil', 'fenofibrate', 'colesevelam',
            'warfarin', 'dabigatran', 'rivaroxaban', 'apixaban', 'edoxaban',
            'nitroglycerin', 'isosorbide', 'amlodipine', 'nifedipine', 'diltiazem',
            'verapamil', 'digoxin', 'furosemide', 'spironolactone', 'eplerenone',
            'morphine', 'fentanyl', 'midazolam', 'propofol', 'diazepam',
            'heparin', 'bivalirudin', 'eptifibatide', 'tirofiban', 'abciximab',
            'alteplase', 'tenecteplase', 'reteplase', 'streptokinase',
            'omeprazole', 'pantoprazole', 'lansoprazole', 'esomeprazole',
            'metformin', 'insulin', 'glipizide', 'glyburide', 'pioglitazone',
            'clopidogrel', 'prasugrel', 'ticagrelor', 'cangrelor', 'vorapaxar'
        ]
        
        # MI indicators
        mi_indicators = [
            'mi', 'myocardial infarction', 'heart attack', 'stemi', 'nstemi',
            'acute coronary syndrome', 'acs', 'acute mi', 'post-mi', 'post-myocardial infarction'
        ]
        
        found_drugs = []
        found_mi = []
        
        # Check for drug names
        for drug in drug_names:
            if drug in text_lower:
                found_drugs.append(drug)
        
        # Check for MI indicators
        for mi in mi_indicators:
            if mi in text_lower:
                found_mi.append(mi)
        
        # If both drug and MI found, return pattern
        if found_drugs and found_mi:
            return f"Drugs: {', '.join(found_drugs[:3])} | MI: {', '.join(found_mi[:3])}"
        
        return None

    def _meets_stringent_pharmacological_requirements(self, text: str) -> bool:
        """Check if article meets stringent pharmacological intervention requirements"""
        # Strong pharmacological indicators (higher weight)
        strong_pharma_indicators = [
            'medication', 'drug', 'pharmaceutical', 'therapy', 'treatment',
            'antiplatelet', 'statin', 'ace inhibitor', 'beta blocker',
            'aspirin', 'clopidogrel', 'atorvastatin', 'metoprolol', 'ezetimibe',
            'simvastatin', 'dose', 'dosage', 'mg', 'pharmacological'
        ]
        
        # Additional pharmacological indicators (medium weight)
        medium_pharma_indicators = [
            'combined with', 'plus', 'versus', 'vs', 'compared with',
            'suppression', 'enhanced', 'results in', 'greater',
            'oral', 'tablet', 'capsule', 'milligram'
        ]
        
        strong_count = sum(1 for indicator in strong_pharma_indicators if indicator in text)
        medium_count = sum(1 for indicator in medium_pharma_indicators if indicator in text)
        
        # More flexible: either 2+ strong indicators OR 1 strong + 1+ medium indicators OR 1+ strong indicators (for single drug studies)
        return strong_count >= 2 or (strong_count >= 1 and medium_count >= 1) or strong_count >= 1
    
    def _get_detailed_analysis(self, text: str, design_type: str, intervention_type: str, population_type: str) -> Dict:
        """Get detailed analysis for specific reasoning"""
        
        # Clinical Trial Analysis (more flexible for drug comparison studies)
        strong_trial_indicators = [
            'randomized controlled trial', 'randomised controlled trial',
            'double-blind randomized', 'placebo-controlled randomized',
            'randomly assigned patients', 'treatment group', 'control group',
            'clinical trial', 'prospective trial', 'controlled trial'
        ]
        comparison_indicators = [
            'versus', 'vs', 'compared with', 'compared to', 'compared against', 'compared',
            'combination', 'combined with', 'plus', 'alone', 'monotherapy',
            'versus placebo', 'vs placebo', 'compared to placebo'
        ]
        study_design_indicators = [
            'randomized', 'randomised', 'randomly assigned', 'random allocation',
            'prospective', 'multicenter', 'multicentre', 'phase', 'trial',
            'study', 'examination', 'investigation', 'evaluation', 'assessment'
        ]
        patient_outcome_indicators = [
            'patients', 'participants', 'subjects', 'outcomes', 'endpoints',
            'efficacy', 'safety', 'effects', 'results', 'response'
        ]
        
        found_strong_trial = [ind for ind in strong_trial_indicators if ind in text]
        found_comparison = [ind for ind in comparison_indicators if ind in text]
        found_design = [ind for ind in study_design_indicators if ind in text]
        found_patients = [ind for ind in patient_outcome_indicators if ind in text]
        
        # Check if meets clinical trial requirements
        strong_count = len(found_strong_trial)
        comparison_count = len(found_comparison)
        design_count = len(found_design)
        patient_count = len(found_patients)
        
        meets_requirements = (strong_count >= 1 or 
                            (comparison_count >= 2 and design_count >= 1 and patient_count >= 2) or
                            (comparison_count >= 1 and design_count >= 2 and patient_count >= 2) or
                            (comparison_count >= 2 and patient_count >= 3) or
                            (comparison_count >= 2 and design_count >= 1) or
                            (design_count >= 2 and patient_count >= 2) or
                            (design_count >= 1 and patient_count >= 1) or
                            (patient_count >= 2 and strong_count >= 0) or
                            (design_count >= 1 and patient_count >= 0))
        
        if meets_requirements:
            if found_strong_trial:
                rct_details = f"Strong clinical trial indicators: {', '.join(found_strong_trial[:3])}"
            else:
                rct_details = f"Drug comparison study: {', '.join(found_comparison[:2])} with {', '.join(found_design[:2])} design in {', '.join(found_patients[:2])}"
        else:
            rct_details = f"Insufficient clinical trial indicators. Found: trial={found_strong_trial[:2]}, comparison={found_comparison[:2]}, design={found_design[:2]}, patients={found_patients[:2]}"
        
        # MI Analysis - Context-aware MI detection
        text_lower = text.lower()
        
        # Check for prevention/risk patterns (should be excluded)
        prevention_patterns = [
            'prevention of mi', 'prevent mi', 'preventing mi', 'prevent myocardial infarction',
            'risk of mi', 'risk of myocardial infarction', 'risk of heart attack',
            'reduce risk', 'reducing risk', 'risk reduction', 'lower risk',
            'primary prevention', 'secondary prevention', 'preventive',
            'at risk of', 'high risk', 'risk factors', 'cardiovascular risk',
            'future mi', 'future myocardial infarction', 'future heart attack',
            'incident mi', 'incident myocardial infarction', 'new mi', 'new myocardial infarction',
            'first mi', 'first myocardial infarction', 'initial mi', 'initial myocardial infarction',
            'prevent cardiovascular', 'prevent cardiac', 'prevent coronary',
            'without mi', 'without myocardial infarction', 'without heart attack',
            'no history of mi', 'no history of myocardial infarction', 'no history of heart attack',
            'healthy patients', 'asymptomatic', 'no symptoms', 'no prior mi'
        ]
        
        # Check for inclusion patterns (patients who are currently suffering or have suffered MI)
        inclusion_patterns = [
            # Current MI patients or patients who have had MI before (all grammatical variations)
            'current mi patients', 'current myocardial infarction patients', 'current heart attack patients',
            'patients who have had mi', 'patients who have had myocardial infarction', 'patients who have had heart attack',
            'patients with current mi', 'patients with current myocardial infarction', 'patients with current heart attack',
            'patients with mi', 'patients with myocardial infarction', 'patients with heart attack',
            'subjects with mi', 'subjects with myocardial infarction', 'subjects with heart attack',
            'individuals with mi', 'individuals with myocardial infarction', 'individuals with heart attack',
            'people with mi', 'people with myocardial infarction', 'people with heart attack',
            'adults with mi', 'adults with myocardial infarction', 'adults with heart attack',
            'post-mi', 'post-myocardial infarction', 'post-infarction',
            'post-acute mi', 'post-acute myocardial infarction',
            'post-mi patients', 'post-myocardial infarction patients',
            'patients after mi', 'patients after myocardial infarction', 'patients after heart attack',
            'following mi', 'following myocardial infarction', 'following heart attack',
            'after mi', 'after myocardial infarction', 'after heart attack',
            'previous mi', 'previous myocardial infarction', 'previous heart attack',
            'prior mi', 'prior myocardial infarction', 'prior heart attack',
            'history of mi', 'history of myocardial infarction', 'history of heart attack',
            'past mi', 'past myocardial infarction', 'past heart attack',
            'recent mi', 'recent myocardial infarction', 'recent heart attack',
            'patients who had mi', 'patients who had myocardial infarction', 'patients who had heart attack',
            'patients with a history of mi', 'patients with a history of myocardial infarction', 'patients with a history of heart attack',
            'subjects who had mi', 'subjects who had myocardial infarction', 'subjects who had heart attack',
            'individuals who had mi', 'individuals who had myocardial infarction', 'individuals who had heart attack',
            'previous stemi', 'previous nstemi', 'prior stemi', 'prior nstemi',
            'history of stemi', 'history of nstemi', 'past stemi', 'past nstemi',
            'post-stemi', 'post-nstemi',
            'patients who had stemi', 'patients who had nstemi',
            'patients with previous stemi', 'patients with previous nstemi',
            'subjects with stemi', 'subjects with nstemi',
            'individuals with stemi', 'individuals with nstemi',
            'mi survivors', 'myocardial infarction survivors', 'heart attack survivors',
            'patients who survived mi', 'patients who survived myocardial infarction', 'patients who survived heart attack',
            'subjects who survived mi', 'subjects who survived myocardial infarction', 'subjects who survived heart attack',
            'mi cohort', 'myocardial infarction cohort', 'heart attack cohort',
            'cohort of mi patients', 'cohort of myocardial infarction patients', 'cohort of heart attack patients',
            'mi group', 'myocardial infarction group', 'heart attack group',
            'patients with acute coronary syndrome and mi',
            'patients with acs and mi',
            'acute coronary syndrome with mi',
            'acs with mi',
            'patients who had acute coronary syndrome and mi',
            'subjects with acute coronary syndrome and mi',
            'stemi patients', 'nstemi patients',
            'patients with stemi', 'patients with nstemi',
            'subjects with stemi', 'subjects with nstemi',
            'patients who had stemi', 'patients who had nstemi',
            'individuals with stemi', 'individuals with nstemi',
            'mi patients treated', 'myocardial infarction patients treated',
            'heart attack patients treated', 'post-mi patients treated',
            'mi survivors treated', 'previous mi patients treated',
            'patients who had mi treated', 'patients who had myocardial infarction treated',
            'subjects with mi treated', 'subjects with myocardial infarction treated',
            'acute mi', 'acute myocardial infarction', 'acute heart attack',
            'acute mi patients', 'acute myocardial infarction patients', 'acute heart attack patients',
            'patients with acute mi', 'patients with acute myocardial infarction', 'patients with acute heart attack',
            'subjects with acute mi', 'subjects with acute myocardial infarction', 'subjects with acute heart attack',
            'mi cases', 'myocardial infarction cases', 'heart attack cases',
            'mi subjects', 'myocardial infarction subjects', 'heart attack subjects',
            'mi participants', 'myocardial infarction participants', 'heart attack participants',
            'mi volunteers', 'myocardial infarction volunteers', 'heart attack volunteers'
        ]
        
        found_prevention = [pattern for pattern in prevention_patterns if pattern in text_lower]
        found_inclusion = [pattern for pattern in inclusion_patterns if pattern in text_lower]
        
        if found_prevention:
            mi_details = f"PREVENTION STUDY EXCLUDED: {', '.join(found_prevention[:3])}"
        elif found_inclusion:
            mi_details = f"Current MI patients or patients who have had MI before: {', '.join(found_inclusion[:3])}"
        else:
            mi_details = f"No clear evidence of current MI patients or patients who have had MI before. Text may be about prevention or risk reduction."
        
        # Pharmacological Analysis
        strong_pharma_indicators = [
            'medication', 'drug', 'pharmaceutical', 'therapy', 'treatment',
            'antiplatelet', 'statin', 'ace inhibitor', 'beta blocker',
            'aspirin', 'clopidogrel', 'atorvastatin', 'metoprolol', 'ezetimibe',
            'simvastatin', 'dose', 'dosage', 'mg', 'pharmacological'
        ]
        medium_pharma_indicators = [
            'combined with', 'plus', 'versus', 'vs', 'compared with',
            'suppression', 'enhanced', 'results in', 'greater',
            'oral', 'tablet', 'capsule', 'milligram'
        ]
        
        found_strong_pharma = [ind for ind in strong_pharma_indicators if ind in text]
        found_medium_pharma = [ind for ind in medium_pharma_indicators if ind in text]
        
        if len(found_strong_pharma) >= 2 or (len(found_strong_pharma) >= 1 and len(found_medium_pharma) >= 1) or len(found_strong_pharma) >= 1:
            pharma_details = f"Strong pharmacological indicators: {', '.join(found_strong_pharma[:3])}"
        else:
            pharma_details = f"Insufficient pharmacological indicators. Found: {', '.join(found_strong_pharma + found_medium_pharma[:3])}"
        
        # Overall Assessment
        overall_score = 0
        if meets_requirements:
            overall_score += 1
        if found_inclusion:
            overall_score += 1
        if len(found_strong_pharma) >= 2 or (len(found_strong_pharma) >= 1 and len(found_medium_pharma) >= 1):
            overall_score += 1
        
        if overall_score == 3:
            overall_assessment = "Meets all three criteria for inclusion"
        elif overall_score == 2:
            overall_assessment = "Meets 2/3 criteria - borderline case"
        elif overall_score == 1:
            overall_assessment = "Meets only 1/3 criteria - insufficient"
        else:
            overall_assessment = "Meets 0/3 criteria - clear exclusion"
        
        return {
            'rct_details': rct_details,
            'mi_details': mi_details,
            'pharma_details': pharma_details,
            'overall_assessment': overall_assessment,
            'found_strong_trial': found_strong_trial,
            'found_comparison': found_comparison,
            'found_design': found_design,
            'found_patients': found_patients,
            'found_prevention': found_prevention,
            'found_inclusion': found_inclusion,
            'found_strong_pharma': found_strong_pharma,
            'found_medium_pharma': found_medium_pharma
        }
    
    def _format_exclusion_reasoning(self, exclusion_reasons: List[str], detailed_analysis: Dict) -> str:
        """Format detailed exclusion reasoning"""
        if len(exclusion_reasons) == 1:
            return f"EXCLUDED: {exclusion_reasons[0]}"
        else:
            primary_reason = exclusion_reasons[0]
            additional_reasons = exclusion_reasons[1:3]  # Show up to 2 additional reasons
            return f"EXCLUDED: {primary_reason}. Additional issues: {'; '.join(additional_reasons)}"
    
    def _format_inclusion_reasoning(self, inclusion_requirements: List[str], detailed_analysis: Dict) -> str:
        """Format detailed inclusion reasoning"""
        return f"INCLUDED: {'; '.join(inclusion_requirements)}. Overall assessment: {detailed_analysis['overall_assessment']}"


class CardiacArticleScreener(ArticleScreener):
    """Specialized article screener for cardiac research - Bio-ClinicalBERT only"""
    
    def __init__(self):
        # Only use Bio-ClinicalBERT provider
        super().__init__(CardiacBioClinicalBERTProvider())
        self.refman_parser = RefManParser()
        self.learning_screener = None
    
    def load_articles_from_refman(self, file_path: str) -> List[Article]:
        """Load articles from RefMan format file"""
        try:
            refman_articles = self.refman_parser.parse_refman_file(file_path)
            articles = []
            
            for refman_article in refman_articles:
                article = Article(
                    title=refman_article.title,
                    abstract=refman_article.abstract,
                    authors=refman_article.authors,
                    journal=refman_article.journal,
                    year=refman_article.year,
                    doi=refman_article.doi,
                    pmid=refman_article.pmid
                )
                articles.append(article)
            
            logger.info(f"Loaded {len(articles)} articles from RefMan file: {file_path}")
            return articles
            
        except Exception as e:
            logger.error(f"Error loading articles from RefMan file: {str(e)}")
            return []
    

def main():
    """Main function for Bio-ClinicalBERT cardiac screening"""
    print("=" * 60)
    print("Cardiac Research Bio-ClinicalBERT Screener")
    print("MI Pharmacological Therapy RCTs")
    print("=" * 60)
    
    # Initialize screener with Bio-ClinicalBERT only
    screener = CardiacArticleScreener()
    
    # Load criteria
    criteria_file = 'screening_criteria_cardiac.json'
    try:
        with open(criteria_file, 'r') as f:
            criteria = json.load(f)
    except FileNotFoundError:
        print(f"Criteria file not found: {criteria_file}")
        return
    
    print(f"\nLoaded screening criteria for: {criteria.get('research_topic', 'Cardiac Research')}")
    print("Using Bio-ClinicalBERT with intelligent criteria and DOI verification")
    
    # Choose input format
    print("\nInput format options:")
    print("1. RefMan format file")
    print("2. CSV file")
    
    choice = input("Choose option (1-2) [1]: ").strip() or "1"
    
    articles = []
    
    if choice == "1":
        refman_file = input("Enter RefMan file path [articles.ris]: ").strip() or "articles.ris"
        if Path(refman_file).exists():
            articles = screener.load_articles_from_refman(refman_file)
        else:
            print(f"File not found: {refman_file}")
            return
    
    elif choice == "2":
        csv_file = input("Enter CSV file path: ").strip()
        if Path(csv_file).exists():
            articles = screener.load_articles_from_csv(csv_file)
        else:
            print(f"File not found: {csv_file}")
            return
    
    if not articles:
        print("No articles loaded")
        return
    
    print(f"\nLoaded {len(articles)} articles")
    
    # Screen articles
    print(f"\nScreening {len(articles)} articles with Bio-ClinicalBERT...")
    
    screened_articles = screener.screen_articles(articles, criteria)
    
    # Save results
    output_file = 'bioclinicalbert_screening_results.csv'
    screener.save_results_to_csv(output_file)
    
    # Show detailed results
    print(f"\nDetailed Results:")
    print("-" * 60)
    
    for i, article in enumerate(screened_articles, 1):
        print(f"\n{i}. {article.title[:70]}...")
        print(f"   Decision: {article.decision.value.upper()}")
        print(f"   Confidence: {article.confidence:.2f}")
        print(f"   Reasoning: {article.reasoning}")
        if article.journal:
            print(f"   Journal: {article.journal}")
        if article.year:
            print(f"   Year: {article.year}")
    
    # Summary statistics
    stats = screener.get_summary_stats()
    print(f"\n{'='*60}")
    print("BIO-CLINICALBERT SCREENING SUMMARY")
    print("="*60)
    print(f"Total articles processed: {stats['total_articles']}")
    print(f"INCLUDE: {stats['include_count']} articles")
    print(f"EXCLUDE: {stats['exclude_count']} articles")
    print(f"MAYBE: {stats['maybe_count']} articles")
    print(f"Average confidence: {stats['avg_confidence']:.2f}")
    print(f"High confidence decisions (≥0.8): {stats['high_confidence_count']}")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
