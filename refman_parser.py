"""
RefMan Format Parser and Learning System
Parses RefMan format files and creates a learning system for article screening criteria.
"""

import re
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from llm_article_screener import Article, ScreeningDecision

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RefManArticle:
    """Represents an article parsed from RefMan format"""
    record_type: str = ""
    title: str = ""
    authors: str = ""
    journal: str = ""
    year: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    abstract: str = ""
    keywords: str = ""
    doi: str = ""
    pmid: str = ""
    url: str = ""
    language: str = ""
    publication_type: str = ""
    notes: str = ""
    raw_record: str = ""

class RefManParser:
    """Parser for RefMan format files"""
    
    def __init__(self):
        self.field_mappings = {
            'TY': 'record_type',
            'TI': 'title', 
            'T1': 'title',
            'AU': 'authors',
            'A1': 'authors',
            'JO': 'journal',
            'JF': 'journal',
            'JA': 'journal',
            'PY': 'year',
            'Y1': 'year',
            'VL': 'volume',
            'IS': 'issue',
            'SP': 'pages',
            'EP': 'pages',
            'AB': 'abstract',
            'N2': 'abstract',
            'KW': 'keywords',
            'DO': 'doi',
            'AN': 'pmid',
            'UR': 'url',
            'LA': 'language',
            'PT': 'publication_type',
            'N1': 'notes'
        }
    
    def parse_refman_file(self, file_path: str) -> List[RefManArticle]:
        """Parse a RefMan format file and return list of articles"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into individual records
            records = re.split(r'\n\s*\n(?=TY\s+-)', content)
            articles = []
            
            for record in records:
                if not record.strip():
                    continue
                    
                article = self._parse_single_record(record.strip())
                if article and article.title:  # Only include records with titles
                    articles.append(article)
            
            logger.info(f"Parsed {len(articles)} articles from {file_path}")
            return articles
            
        except Exception as e:
            logger.error(f"Error parsing RefMan file {file_path}: {str(e)}")
            return []
    
    def _parse_single_record(self, record: str) -> Optional[RefManArticle]:
        """Parse a single RefMan record"""
        try:
            article = RefManArticle()
            article.raw_record = record
            
            lines = record.split('\n')
            current_field = None
            current_value = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is a field line (starts with field code)
                field_match = re.match(r'^([A-Z][A-Z0-9])\s*-\s*(.*)', line)
                
                if field_match:
                    # Save previous field if exists
                    if current_field and current_value:
                        self._set_field_value(article, current_field, current_value.strip())
                    
                    # Start new field
                    current_field = field_match.group(1)
                    current_value = field_match.group(2)
                else:
                    # Continuation of previous field
                    if current_value:
                        current_value += " " + line
                    else:
                        current_value = line
            
            # Don't forget the last field
            if current_field and current_value:
                self._set_field_value(article, current_field, current_value.strip())
            
            return article
            
        except Exception as e:
            logger.error(f"Error parsing single record: {str(e)}")
            return None
    
    def _set_field_value(self, article: RefManArticle, field_code: str, value: str):
        """Set field value in article object"""
        if field_code in self.field_mappings:
            field_name = self.field_mappings[field_code]
            
            # Handle special cases
            if field_name == 'authors':
                # Combine multiple author entries
                existing = getattr(article, field_name)
                if existing:
                    setattr(article, field_name, existing + "; " + value)
                else:
                    setattr(article, field_name, value)
            elif field_name == 'pages':
                # Combine start and end pages
                existing = getattr(article, field_name)
                if existing:
                    setattr(article, field_name, existing + "-" + value)
                else:
                    setattr(article, field_name, value)
            elif field_name == 'keywords':
                # Combine multiple keyword entries
                existing = getattr(article, field_name)
                if existing:
                    setattr(article, field_name, existing + "; " + value)
                else:
                    setattr(article, field_name, value)
            else:
                setattr(article, field_name, value)
    
    def convert_to_csv(self, articles: List[RefManArticle], output_file: str):
        """Convert RefMan articles to CSV format"""
        try:
            data = []
            for article in articles:
                data.append({
                    'title': article.title,
                    'abstract': article.abstract,
                    'authors': article.authors,
                    'journal': article.journal,
                    'year': article.year,
                    'volume': article.volume,
                    'issue': article.issue,
                    'pages': article.pages,
                    'doi': article.doi,
                    'pmid': article.pmid,
                    'keywords': article.keywords,
                    'language': article.language,
                    'publication_type': article.publication_type,
                    'record_type': article.record_type,
                    'url': article.url,
                    'notes': article.notes
                })
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            logger.info(f"Converted {len(articles)} articles to CSV: {output_file}")
            
        except Exception as e:
            logger.error(f"Error converting to CSV: {str(e)}")

class LearningScreener:
    """Learning system that improves screening criteria based on example articles"""
    
    def __init__(self, base_criteria: Dict):
        self.base_criteria = base_criteria.copy()
        self.learned_patterns = {
            'include_patterns': [],
            'exclude_patterns': [],
            'title_patterns': [],
            'abstract_patterns': [],
            'journal_patterns': [],
            'author_patterns': []
        }
        self.training_examples = []
    
    def add_training_example(self, article: RefManArticle, decision: ScreeningDecision, 
                           reasoning: str = ""):
        """Add a training example to learn from"""
        example = {
            'article': article,
            'decision': decision,
            'reasoning': reasoning,
            'title': article.title.lower(),
            'abstract': article.abstract.lower(),
            'journal': article.journal.lower(),
            'authors': article.authors.lower(),
            'keywords': article.keywords.lower(),
            'record_type': article.record_type.lower(),
            'publication_type': article.publication_type.lower()
        }
        
        self.training_examples.append(example)
        logger.info(f"Added training example: {decision.value} - {article.title[:50]}...")
    
    def learn_from_examples(self):
        """Analyze training examples and update screening criteria"""
        logger.info(f"Learning from {len(self.training_examples)} training examples...")
        
        include_examples = [ex for ex in self.training_examples if ex['decision'] == ScreeningDecision.INCLUDE]
        exclude_examples = [ex for ex in self.training_examples if ex['decision'] == ScreeningDecision.EXCLUDE]
        
        # Learn inclusion patterns
        self._extract_patterns(include_examples, 'include')
        
        # Learn exclusion patterns
        self._extract_patterns(exclude_examples, 'exclude')
        
        # Update base criteria with learned patterns
        self._update_criteria()
        
        logger.info("Learning completed. Updated screening criteria.")
    
    def _extract_patterns(self, examples: List[Dict], pattern_type: str):
        """Extract common patterns from examples"""
        if not examples:
            return
        
        # Extract common words and phrases
        all_titles = " ".join([ex['title'] for ex in examples])
        all_abstracts = " ".join([ex['abstract'] for ex in examples])
        all_journals = [ex['journal'] for ex in examples if ex['journal']]
        all_record_types = [ex['record_type'] for ex in examples if ex['record_type']]
        all_pub_types = [ex['publication_type'] for ex in examples if ex['publication_type']]
        
        # Find frequent terms (simple approach)
        title_words = self._extract_frequent_terms(all_titles)
        abstract_words = self._extract_frequent_terms(all_abstracts)
        
        # Store patterns
        pattern_key = f'{pattern_type}_patterns'
        self.learned_patterns[pattern_key].extend([
            {'type': 'title_terms', 'terms': title_words[:10]},  # Top 10
            {'type': 'abstract_terms', 'terms': abstract_words[:10]},
            {'type': 'journals', 'values': list(set(all_journals))},
            {'type': 'record_types', 'values': list(set(all_record_types))},
            {'type': 'publication_types', 'values': list(set(all_pub_types))}
        ])
    
    def _extract_frequent_terms(self, text: str, min_length: int = 4) -> List[str]:
        """Extract frequent terms from text"""
        if not text:
            return []
        
        # Simple word frequency analysis
        words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + r',}\b', text.lower())
        
        # Count frequencies
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return most frequent words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words if count >= 2]  # At least 2 occurrences
    
    def _update_criteria(self):
        """Update base criteria with learned patterns"""
        # Add learned inclusion keywords
        for pattern in self.learned_patterns['include_patterns']:
            if pattern['type'] == 'title_terms':
                self.base_criteria.setdefault('learned_include_keywords', []).extend(pattern['terms'])
            elif pattern['type'] == 'abstract_terms':
                self.base_criteria.setdefault('learned_include_abstract_terms', []).extend(pattern['terms'])
        
        # Add learned exclusion keywords
        for pattern in self.learned_patterns['exclude_patterns']:
            if pattern['type'] == 'title_terms':
                self.base_criteria.setdefault('learned_exclude_keywords', []).extend(pattern['terms'])
            elif pattern['type'] == 'abstract_terms':
                self.base_criteria.setdefault('learned_exclude_abstract_terms', []).extend(pattern['terms'])
        
        # Remove duplicates
        for key in ['learned_include_keywords', 'learned_include_abstract_terms', 
                   'learned_exclude_keywords', 'learned_exclude_abstract_terms']:
            if key in self.base_criteria:
                self.base_criteria[key] = list(set(self.base_criteria[key]))
    
    def get_updated_criteria(self) -> Dict:
        """Get the updated criteria after learning"""
        return self.base_criteria.copy()
    
    def save_learned_criteria(self, output_file: str):
        """Save the learned criteria to a file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.base_criteria, f, indent=2)
            logger.info(f"Saved learned criteria to {output_file}")
        except Exception as e:
            logger.error(f"Error saving learned criteria: {str(e)}")

def create_sample_refman_file():
    """Create a sample RefMan file for testing"""
    sample_content = """TY  - JOUR
TI  - Effect of atorvastatin on cardiovascular outcomes in patients with acute coronary syndromes: a randomized controlled trial
AU  - Smith, John A.
AU  - Johnson, Mary B.
AU  - Williams, Robert C.
JO  - New England Journal of Medicine
PY  - 2023
VL  - 389
IS  - 12
SP  - 1123
EP  - 1135
AB  - Background: High-intensity statin therapy is recommended for secondary prevention after acute coronary syndromes. This randomized controlled trial evaluated the efficacy of atorvastatin 80mg versus placebo in patients with STEMI and NSTEMI. Methods: We randomized 2,847 patients with acute MI to receive either atorvastatin 80mg daily or matching placebo. Primary endpoint was major adverse cardiovascular events (MACE) at 12 months. Results: Atorvastatin significantly reduced MACE compared to placebo (12.3% vs 18.7%, HR 0.64, 95% CI 0.52-0.79, p<0.001). Conclusion: High-intensity atorvastatin therapy improves cardiovascular outcomes in post-MI patients.
KW  - atorvastatin; myocardial infarction; secondary prevention; randomized controlled trial; STEMI; NSTEMI
DO  - 10.1056/NEJMoa2023001
AN  - 37123456
LA  - English
PT  - Journal Article
ER  - 

TY  - JOUR
TI  - Systematic review and meta-analysis of dual antiplatelet therapy duration after percutaneous coronary intervention
AU  - Brown, Lisa M.
AU  - Davis, Michael R.
JO  - Circulation
PY  - 2023
VL  - 147
IS  - 8
SP  - 642
EP  - 655
AB  - This systematic review and meta-analysis examined optimal duration of dual antiplatelet therapy (DAPT) after PCI. We searched MEDLINE, Embase, and Cochrane databases through March 2023. Included studies were randomized controlled trials comparing different DAPT durations. Twenty-three trials with 89,234 patients were included. Extended DAPT (>12 months) reduced ischemic events but increased bleeding risk. The optimal duration depends on individual patient bleeding and thrombotic risk profiles.
KW  - dual antiplatelet therapy; percutaneous coronary intervention; meta-analysis; systematic review
DO  - 10.1161/CIRCULATIONAHA.122.063789
AN  - 36987654
LA  - English
PT  - Review
ER  - 

TY  - JOUR
TI  - Cardiac regeneration using stem cells in a murine model of myocardial infarction
AU  - Garcia, Carlos E.
AU  - Lee, Sarah H.
JO  - Nature Biotechnology
PY  - 2022
VL  - 40
IS  - 15
SP  - 1891
EP  - 1902
AB  - We investigated cardiac regeneration potential using induced pluripotent stem cell-derived cardiomyocytes in a mouse model of myocardial infarction. Male C57BL/6 mice underwent left anterior descending artery ligation followed by intramyocardial injection of stem cells or vehicle control. Echocardiography and histological analysis were performed at 4 and 12 weeks. Stem cell therapy improved left ventricular function and reduced infarct size compared to controls.
KW  - stem cells; cardiac regeneration; myocardial infarction; animal model; mice
DO  - 10.1038/s41587-022-01456-2
AN  - 35789123
LA  - English
PT  - Journal Article
ER  - 

TY  - JOUR
TI  - Case report: Rare presentation of takotsubo cardiomyopathy following emotional stress
AU  - Wilson, Patricia K.
AU  - Anderson, David J.
JO  - American Journal of Cardiology
PY  - 2023
VL  - 195
SP  - 89
EP  - 92
AB  - We report a case of a 67-year-old woman who developed takotsubo cardiomyopathy following the death of her spouse. She presented with chest pain and ST-elevation on ECG, initially suspected to be STEMI. Coronary angiography revealed normal coronaries and characteristic apical ballooning on ventriculography. She recovered completely with supportive care. This case highlights the importance of considering takotsubo cardiomyopathy in the differential diagnosis of acute coronary syndromes.
KW  - takotsubo cardiomyopathy; case report; stress cardiomyopathy; acute coronary syndrome
DO  - 10.1016/j.amjcard.2023.02.015
AN  - 36854321
LA  - English
PT  - Case Reports
ER  - """
    
    with open('sample_refman.txt', 'w') as f:
        f.write(sample_content)
    
    logger.info("Created sample RefMan file: sample_refman.txt")

def main():
    """Main function to demonstrate RefMan parsing and learning"""
    print("=" * 60)
    print("RefMan Parser and Learning System")
    print("=" * 60)
    
    # Create sample RefMan file if it doesn't exist
    if not Path('sample_refman.txt').exists():
        create_sample_refman_file()
    
    # Parse RefMan file
    parser = RefManParser()
    articles = parser.parse_refman_file('sample_refman.txt')
    
    if not articles:
        print("No articles found in RefMan file")
        return
    
    print(f"\nParsed {len(articles)} articles from RefMan file:")
    for i, article in enumerate(articles, 1):
        print(f"\n{i}. {article.title[:80]}...")
        print(f"   Journal: {article.journal}")
        print(f"   Year: {article.year}")
        print(f"   Type: {article.record_type}")
        print(f"   Abstract: {article.abstract[:100]}...")
    
    # Convert to CSV
    parser.convert_to_csv(articles, 'refman_articles.csv')
    
    # Demonstrate learning system
    print("\n" + "=" * 60)
    print("Learning System Demonstration")
    print("=" * 60)
    
    # Load base criteria
    try:
        with open('screening_criteria_cardiac.json', 'r') as f:
            base_criteria = json.load(f)
    except FileNotFoundError:
        print("Base criteria file not found")
        return
    
    # Initialize learning screener
    learner = LearningScreener(base_criteria)
    
    # Add training examples (manually label the sample articles)
    training_labels = [
        (0, ScreeningDecision.INCLUDE, "RCT with STEMI/NSTEMI patients, secondary prevention"),
        (1, ScreeningDecision.EXCLUDE, "Systematic review and meta-analysis"),
        (2, ScreeningDecision.EXCLUDE, "Animal study using mice"),
        (3, ScreeningDecision.EXCLUDE, "Case report, not an RCT")
    ]
    
    for idx, decision, reasoning in training_labels:
        if idx < len(articles):
            learner.add_training_example(articles[idx], decision, reasoning)
            print(f"Training example {idx+1}: {decision.value.upper()}")
            print(f"  Title: {articles[idx].title[:60]}...")
            print(f"  Reasoning: {reasoning}")
    
    # Learn from examples
    print(f"\nLearning from {len(training_labels)} examples...")
    learner.learn_from_examples()
    
    # Save learned criteria
    learner.save_learned_criteria('learned_criteria_cardiac.json')
    
    print("\nLearning completed! Check 'learned_criteria_cardiac.json' for updated criteria.")
    print("Files created:")
    print("- sample_refman.txt (sample RefMan format)")
    print("- refman_articles.csv (converted to CSV)")
    print("- learned_criteria_cardiac.json (learned criteria)")

if __name__ == "__main__":
    main()
