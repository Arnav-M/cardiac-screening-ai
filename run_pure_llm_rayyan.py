#!/usr/bin/env python3
"""
Pure LLM Cardiac Rayyan Runner
Uses Bio-ClinicalBERT directly without any training data
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Our imports
from llm_article_screener import Article, ScreeningDecision
from cardiac_llm_screener import CardiacBioClinicalBERTProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PureLLMCardiacRayyanRunner:
    """Run Rayyan screening using pure Bio-ClinicalBERT without training data"""
    
    def __init__(self, max_articles: int = 1000, confidence_threshold: float = 0.7):
        self.max_articles = max_articles
        self.confidence_threshold = confidence_threshold
        self.driver = None
        self.rayyan_url = "https://new.rayyan.ai/reviews/1643911/screening"
        
        # Login credentials
        self.username = "vedantkasmalkarmd@gmail.com"
        self.password = "Masvadi123"
        
        # Initialize Bio-ClinicalBERT provider
        self.provider = CardiacBioClinicalBERTProvider()
        
        # Load base criteria
        self.base_criteria = self.load_base_criteria()
    
    def load_base_criteria(self) -> Dict:
        """Load base screening criteria"""
        try:
            with open('screening_criteria_cardiac.json', 'r') as f:
                criteria = json.load(f)
            logger.info("Loaded base screening criteria")
            return criteria
        except Exception as e:
            logger.error(f"Error loading criteria: {str(e)}")
            return {}
    
    def setup_driver(self):
        """Set up Chrome WebDriver"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.maximize_window()
            
            logger.info("Chrome driver initialized for Rayyan")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up Chrome driver: {str(e)}")
            return False
    
    def login_to_rayyan(self) -> bool:
        """Navigate to Rayyan and automatically log in"""
        try:
            logger.info("Opening Rayyan login page...")
            # Go to login page first
            self.driver.get("https://new.rayyan.ai/login")
            
            # Wait for login form to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email'], input[name='email'], input[placeholder*='email' i]"))
            )
            
            logger.info("Attempting automatic login...")
            
            # Find and fill email field
            email_selectors = [
                "input[type='email']",
                "input[name='email']",
                "input[placeholder*='email' i]",
                "input[id*='email' i]",
                "#email",
                "[data-testid*='email']"
            ]
            
            email_field = None
            for selector in email_selectors:
                try:
                    email_field = self.driver.find_element(By.CSS_SELECTOR, selector)
                    break
                except:
                    continue
            
            if not email_field:
                logger.error("Could not find email field")
                return self._fallback_manual_login()
            
            email_field.clear()
            email_field.send_keys(self.username)
            logger.info("Email entered")
            
            # Find and fill password field
            password_selectors = [
                "input[type='password']",
                "input[name='password']",
                "input[placeholder*='password' i]",
                "input[id*='password' i]",
                "#password",
                "[data-testid*='password']"
            ]
            
            password_field = None
            for selector in password_selectors:
                try:
                    password_field = self.driver.find_element(By.CSS_SELECTOR, selector)
                    break
                except:
                    continue
            
            if not password_field:
                logger.error("Could not find password field")
                return self._fallback_manual_login()
            
            password_field.clear()
            password_field.send_keys(self.password)
            logger.info("Password entered")
            
            # Find and click login button
            login_button_selectors = [
                "button[type='submit']",
                "input[type='submit']",
                "button:contains('Log in')",
                "button:contains('Sign in')",
                "[data-testid*='login']",
                "[data-testid*='submit']",
                ".login-button",
                "#login-button"
            ]
            
            login_button = None
            for selector in login_button_selectors:
                try:
                    login_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    break
                except:
                    continue
            
            if not login_button:
                # Try finding button by text content
                buttons = self.driver.find_elements(By.TAG_NAME, "button")
                for button in buttons:
                    if any(text in button.text.lower() for text in ['log in', 'sign in', 'login', 'submit']):
                        login_button = button
                        break
            
            if not login_button:
                logger.error("Could not find login button")
                return self._fallback_manual_login()
            
            login_button.click()
            logger.info("Login button clicked")
            
            # Wait for login to complete and redirect
            time.sleep(3)
            
            print("\n" + "="*80)
            print("LOGIN COMPLETED")
            print("="*80)
            print("[SUCCESS] Successfully logged into Rayyan!")
            print()
            print("[INFO] Please navigate manually to your screening page:")
            print("1. Choose the correct project/review")
            print("2. Make sure articles are visible")
            print("3. Set any filters as desired")
            print("4. Come back here when ready")
            print()
            print("[WAITING] Waiting for you to navigate to the screening page...")
            
            # Simple confirmation - let user navigate completely manually
            while True:
                confirm = input("\n[READY] Ready to start article processing? Type 'y' or 'yes' to begin: ").strip().lower()
                if confirm in ['y', 'yes']:
                    # Just verify we can find some kind of screening interface
                    try:
                        # Look for any of the common screening interface elements
                        interface_found = (
                            self.driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="decisions-bar"]') or 
                            self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="article"]') or
                            self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="virtuoso-scroller"]') or
                            self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="result-item-base"]')
                        )
                        
                        if interface_found:
                            logger.info("[SUCCESS] Screening interface detected - ready to start processing!")
                            print("[SUCCESS] Starting article processing...")
                            return True
                        else:
                            print("[WARNING] Could not detect screening interface elements.")
                            print("   Please make sure you're on a screening page with articles visible.")
                            retry = input("   Try again? (y/n): ").strip().lower()
                            if retry not in ['y', 'yes']:
                                logger.error("User chose not to retry - stopping")
                                return False
                            continue
                            
                    except Exception as e:
                        logger.debug(f"Interface detection error: {str(e)}")
                        print("[WARNING] Could not verify screening interface.")
                        print("   The tool will try to proceed anyway.")
                        proceed = input("   Continue? (y/n): ").strip().lower()
                        if proceed in ['y', 'yes']:
                            return True
                        else:
                            return False
                else:
                    print("   Please type 'y' or 'yes' when you're ready to start.")
                    continue
            
        except Exception as e:
            logger.error(f"Automatic login failed: {str(e)}")
            return self._fallback_manual_login()
    
    def _fallback_manual_login(self) -> bool:
        """Fallback to manual login if automatic login fails"""
        try:
            logger.info("Falling back to manual login...")
            
            print("\n" + "="*80)
            print("MANUAL LOGIN REQUIRED")
            print("="*80)
            print("Automatic login failed. Please log in manually.")
            print(f"Username: {self.username}")
            print(f"Password: {self.password}")
            print()
            print("[INFO] Please:")
            print("1. Log in to Rayyan manually")
            print("2. Navigate to your screening page")
            print("3. Make sure articles are visible")
            print("4. Check that filters are set as desired")
            print()
            
            # Wait for user confirmation
            while True:
                confirm = input("[READY] Ready to start automation? Type 'y' or 'yes' to begin: ").strip().lower()
                if confirm in ['y', 'yes']:
                    break
            
            print("[SUCCESS] Starting automation...")
            
            # Verify we can find the screening interface
            WebDriverWait(self.driver, 10).until(
                lambda driver: driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="decisions-bar"]') or 
                               driver.find_elements(By.CSS_SELECTOR, '[data-testid="article"]')
            )
            
            logger.info("[SUCCESS] Rayyan screening interface detected")
            return True
            
        except Exception as e:
            logger.error(f"Manual login fallback failed: {str(e)}")
            return False
    
    
    def extract_current_article(self) -> Optional[Dict]:
        """Extract information from the currently selected article in the virtualized list"""
        try:
            # First, try to find the currently selected article in the list
            selected_article_selectors = [
                '[data-testid="result-item-base"][data-selected="true"]',
                'div[data-selected="true"]',
                '.mr-1\\.5.group.rounded-2xl[data-selected="true"]'
            ]
            
            article_element = None
            for selector in selected_article_selectors:
                try:
                    article_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    logger.debug(f"Found selected article using selector: {selector}")
                    break
                except:
                    continue
            
            # If no selected article found, try to find the decisions bar (old method)
            if not article_element:
                try:
                    article_element = self.driver.find_element(By.CSS_SELECTOR, 'div[data-testid="decisions-bar"]')
                    logger.debug("Found article using decisions-bar selector")
                except:
                    logger.debug("Could not find article using any method")
                    return None
            
            # Extract title - try multiple selectors based on the HTML structure
            title = ""
            title_selectors = [
                '[data-testid="result-item-title"] p',
                'div.font-semibold p',
                '[data-testid="result-item-title"] div p',
                '.truncate p',
                'p'  # Last resort
            ]
            
            for selector in title_selectors:
                try:
                    title_elem = article_element.find_element(By.CSS_SELECTOR, selector)
                    title = title_elem.text.strip()
                    if title and len(title) > 10:  # Make sure we got a meaningful title
                        logger.debug(f"Found title using selector: {selector}")
                        break
                except:
                    continue
            
            # Enhanced abstract extraction with multiple selectors
            abstract = ""
            abstract_selectors = [
                'p[data-testid="abstract"]',
                '[data-testid="abstract"]',
                '.abstract',
                'p.abstract',
                'div.abstract',
                '[class*="abstract"]',
                'p[class*="abstract"]',
                # Try to find any paragraph with substantial text
                'div[data-testid="decisions-bar"] p:not(.font-semibold)',
                'div[data-testid="decisions-bar"] div p:nth-of-type(2)',
                'div[data-testid="decisions-bar"] div p:last-of-type'
            ]
            
            for selector in abstract_selectors:
                try:
                    abstract_elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                    potential_abstract = abstract_elem.text.strip()
                    
                    # Check if this looks like an abstract (longer text, not just title)
                    if (potential_abstract and 
                        len(potential_abstract) > 50 and 
                        potential_abstract != title and
                        not potential_abstract.startswith('Author')):
                        abstract = potential_abstract
                        logger.debug(f"Found abstract using selector: {selector}")
                        break
                except:
                    continue
            
            if not abstract:
                logger.debug("No abstract found with any selector, trying to get all text")
                try:
                    # Try to get all text content and extract potential abstract
                    all_text = article_element.text
                    lines = [line.strip() for line in all_text.split('\n') if line.strip()]
                    
                    # Look for lines that might be abstract (longer than title, contains common abstract words)
                    abstract_indicators = ['study', 'patient', 'method', 'result', 'conclusion', 'background', 'objective']
                    for line in lines:
                        if (len(line) > 100 and 
                            line != title and
                            any(indicator in line.lower() for indicator in abstract_indicators)):
                            abstract = line
                            logger.debug("Found potential abstract in full text")
                            break
                except:
                    pass
            
            # Extract additional metadata from the article element
            authors = ""
            year = ""
            journal = ""
            
            try:
                # Extract authors - look for text that appears to be author names
                author_selectors = [
                    'span.block.w-max',  # Based on the HTML structure you provided
                    '.text-detailsText span',
                    'span:contains(",")',  # Authors usually have commas
                ]
                
                for selector in author_selectors:
                    try:
                        author_elem = article_element.find_element(By.CSS_SELECTOR, selector)
                        potential_authors = author_elem.text.strip()
                        if potential_authors and ',' in potential_authors:
                            authors = potential_authors
                            logger.debug(f"Found authors: {authors[:50]}...")
                            break
                    except:
                        continue
                        
                # Extract year from date field
                date_selectors = [
                    'span:contains("Date:")',
                    '.text-detailsText:contains("Date:")'
                ]
                
                for selector in date_selectors:
                    try:
                        date_elem = article_element.find_element(By.CSS_SELECTOR, selector)
                        date_text = date_elem.text.strip()
                        # Extract year from date string like "Date: 1998-11-01"
                        if 'Date:' in date_text:
                            year = date_text.split('Date:')[1].strip()[:4]
                            logger.debug(f"Found year: {year}")
                            break
                    except:
                        continue
                
                # Try to extract journal from citation attribute or text
                if hasattr(article_element, 'get_attribute'):
                    citation = article_element.get_attribute('citation')
                    if citation:
                        # Extract journal name from citation
                        parts = citation.split(' - ')
                        if len(parts) > 0:
                            journal = parts[0].strip()
                            logger.debug(f"Found journal: {journal[:50]}...")
                            
            except Exception as e:
                logger.debug(f"Error extracting metadata: {str(e)}")
            
            # EXCLUDE articles with no title or very short titles
            if not title or not title.strip():
                logger.debug("Skipping article with no title")
                return None
            
            if len(title.strip()) < 10:
                logger.debug(f"Skipping article with short title ({len(title.strip())} chars): {title}")
                return None
            
            # Log what we extracted
            logger.info(f"Extracted - Title: {title[:60]}... Abstract: {len(abstract)} chars")
            if abstract:
                logger.debug(f"Abstract preview: {abstract[:100]}...")
            if authors:
                logger.debug(f"Authors: {authors[:50]}...")
            if year:
                logger.debug(f"Year: {year}")
            
            # Return article with meaningful title and available metadata
            return {
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'journal': journal,
                'year': year
            }
                
        except Exception as e:
            logger.debug(f"Error extracting current article: {str(e)}")
            return None
    
    def click_decision_button(self, decision: ScreeningDecision) -> bool:
        """Click the appropriate decision button (Accept/Reject/Maybe)"""
        try:
            # Map decisions to button actions
            if decision == ScreeningDecision.INCLUDE:
                # Look for Accept/Include button
                button_selectors = [
                    'button[data-testid*="include"]',
                    'button[aria-label*="Include"]',
                    'button[title*="Include"]',
                    'button[data-testid*="accept"]',
                    'button[aria-label*="Accept"]'
                ]
            elif decision == ScreeningDecision.EXCLUDE:
                # Look for Reject/Exclude button
                button_selectors = [
                    'button[data-testid*="exclude"]',
                    'button[aria-label*="Exclude"]',
                    'button[title*="Exclude"]',
                    'button[data-testid*="reject"]',
                    'button[aria-label*="Reject"]'
                ]
            else:  # MAYBE
                # Look for Maybe/Undecided button
                button_selectors = [
                    'button[data-testid*="maybe"]',
                    'button[aria-label*="Maybe"]',
                    'button[title*="Maybe"]',
                    'button[data-testid*="undecided"]',
                    'button[aria-label*="Undecided"]'
                ]
            
            # Try each selector until one works
            for selector in button_selectors:
                try:
                    button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    button.click()
                    logger.info(f"Clicked {decision.value} button")
                    return True
                except:
                    continue
            
            # If no selector worked, log available buttons for debugging
            logger.error(f"Could not find {decision.value} button. Available buttons:")
            buttons = self.driver.find_elements(By.TAG_NAME, 'button')
            for btn in buttons[:10]:  # Show first 10 buttons
                try:
                    text = btn.text.strip()
                    data_testid = btn.get_attribute('data-testid')
                    aria_label = btn.get_attribute('aria-label')
                    logger.error(f"  Button: text='{text}', data-testid='{data_testid}', aria-label='{aria_label}'")
                except:
                    pass
            
            return False
                
        except Exception as e:
            logger.error(f"Error clicking decision button: {str(e)}")
            return False
    
    def process_articles(self):
        """Process articles using pure LLM screening - exclude articles that can't be processed"""
        processed = 0
        include_count = 0
        exclude_count = 0
        maybe_count = 0
        skipped_count = 0
        
        print(f"Step 4: Processing up to {self.max_articles} articles with STRINGENT Bio-ClinicalBERT...")
        
        for i in range(1, self.max_articles + 1):
            logger.info(f"Processing article {i}/{self.max_articles}")
            
            # Extract current article
            article_data = self.extract_current_article()
            
            if not article_data:
                logger.warning(f"Could not extract article {i} - excluding/skipping")
                skipped_count += 1
                
                # Automatically exclude articles that can't be processed
                try:
                    if self.click_decision_button(ScreeningDecision.EXCLUDE):
                        logger.info("Successfully excluded unprocessable article")
                        exclude_count += 1
                        processed += 1
                    else:
                        logger.warning("Failed to exclude unprocessable article")
                except Exception as e:
                    logger.error(f"Error excluding unprocessable article: {str(e)}")
                
                # Brief pause before next article
                time.sleep(1.5)
                continue
            
            # Create Article object
            article = Article(
                title=article_data['title'],
                abstract=article_data['abstract'],
                authors=article_data['authors'],
                journal=article_data['journal'],
                year=article_data['year']
            )
            
            # Screen with Bio-ClinicalBERT
            try:
                decision, confidence, reasoning = self.provider.screen_article(article, self.base_criteria)
                
                # Display result
                print(f"\nArticle {i}: {article.title[:60]}...")
                print(f"  Decision: {decision.value.upper()} (confidence: {confidence:.2f})")
                print(f"  Reasoning: {reasoning[:100]}...")
                
                # Apply decision in Rayyan
                if self.click_decision_button(decision):
                    logger.info(f"Successfully applied {decision.value} decision")
                    processed += 1
                    
                    if decision == ScreeningDecision.INCLUDE:
                        include_count += 1
                    elif decision == ScreeningDecision.EXCLUDE:
                        exclude_count += 1
                    else:
                        maybe_count += 1
                else:
                    logger.warning(f"Failed to apply decision for article {i}")
                
            except Exception as e:
                logger.error(f"Error screening article {i}: {str(e)}")
                # If screening fails, exclude the article
                try:
                    if self.click_decision_button(ScreeningDecision.EXCLUDE):
                        logger.info("Excluded article due to screening error")
                        exclude_count += 1
                        processed += 1
                    else:
                        logger.warning("Failed to exclude article with screening error")
                except:
                    logger.error("Failed to exclude article after screening error")
            
            # Wait before next article (clicking should auto-advance)
            time.sleep(1.5)
            
            # Progress update every 50 articles
            if i % 50 == 0:
                print(f"\n[PROGRESS] Processed {processed} articles so far:")
                print(f"   Include: {include_count}, Exclude: {exclude_count}, Maybe: {maybe_count}")
                print(f"   Skipped (unprocessable): {skipped_count}")
        
        return {
            'total_processed': processed,
            'include_count': include_count,
            'exclude_count': exclude_count,
            'maybe_count': maybe_count,
            'skipped_count': skipped_count
        }
    
    def run(self):
        """Main execution flow"""
        print("=" * 80)
        print("STRINGENT BIO-CLINICALBERT RAYYAN RUNNER")
        print("Using Enhanced Bio-ClinicalBERT with Stringent Criteria")
        print("=" * 80)
        
        # Step 1: Setup
        print("Step 1: Setting up browser...")
        if not self.setup_driver():
            print("[ERROR] Failed to setup browser")
            return
        
        # Step 2: Login
        print("Step 2: Logging into Rayyan...")
        if not self.login_to_rayyan():
            print("[ERROR] Failed to login to Rayyan")
            return
        
        # Step 3: Process articles
        try:
            results = self.process_articles()
            
            print("\n" + "="*80)
            print("WORKFLOW COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("[RESULTS] Processing Results:")
            print(f"   Total processed: {results['total_processed']}")
            print(f"   Include: {results['include_count']}")
            print(f"   Exclude: {results['exclude_count']}")
            print(f"   Maybe: {results['maybe_count']}")
            print(f"   Skipped (unprocessable): {results['skipped_count']}")
            
        except KeyboardInterrupt:
            print("\n[WARNING] Process interrupted by user")
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
        
        finally:
            input("\nPress Enter to close browser...")
            if self.driver:
                self.driver.quit()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pure LLM Cardiac Rayyan Runner')
    parser.add_argument('--max-articles', type=int, default=1000, help='Maximum articles to process')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold')
    
    args = parser.parse_args()
    
    runner = PureLLMCardiacRayyanRunner(
        max_articles=args.max_articles,
        confidence_threshold=args.confidence
    )
    
    runner.run()

if __name__ == "__main__":
    main()
