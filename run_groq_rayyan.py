#!/usr/bin/env python3
"""
Groq LLM Cardiac Rayyan Runner
Uses Groq LLM instead of Bio-ClinicalBERT - EXACT SAME CODE STRUCTURE
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

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except (ImportError, UnicodeDecodeError, Exception):
    # If python-dotenv is not installed or .env file is corrupted, skip it
    # Environment variables can still be set manually
    pass

# Our imports
from llm_article_screener import Article, ScreeningDecision, GroqProvider
from cardiac_llm_screener import CardiacBioClinicalBERTProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GroqLLMCardiacRayyanRunner:
    """Run Rayyan screening using Groq LLM - EXACT SAME AS Bio-ClinicalBERT VERSION"""
    
    def __init__(self, max_articles: int = 1000, confidence_threshold: float = 0.7):
        self.max_articles = max_articles
        self.confidence_threshold = confidence_threshold
        self.driver = None
        self.rayyan_url = "https://new.rayyan.ai/reviews/1643911/screening"
        
        # Login credentials - Get from environment variables for security
        self.username = os.getenv('RAYYAN_EMAIL', "vedantkasmalkarmd@gmail.com")
        self.password = os.getenv('RAYYAN_PASSWORD', "Masvadi123")
        
        # Warn if using default credentials
        if self.username == "vedantkasmalkarmd@gmail.com":
            logger.warning("Using default Rayyan credentials. Set RAYYAN_EMAIL and RAYYAN_PASSWORD in .env file for your own credentials.")
        
        # Initialize Groq provider only
        self.provider = GroqProvider(model_name="llama-3.3-70b-versatile")
        
        # Rate limiting settings for Rayyan integration
        self.api_call_count = 0
        self.start_time = time.time()
        self.calls_per_minute_limit = 15  # Conservative limit for free tier
        
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
                except NoSuchElementException:
                    continue
            
            if not email_field:
                logger.error("Could not find email field")
                return False
            
            email_field.clear()
            email_field.send_keys(self.username)
            logger.info("Filled email field")
            
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
                except NoSuchElementException:
                    continue
            
            if not password_field:
                logger.error("Could not find password field")
                return False
            
            password_field.clear()
            password_field.send_keys(self.password)
            logger.info("Filled password field")
            
            # Find and click login button
            login_selectors = [
                "button[type='submit']",
                "input[type='submit']",
                "button:contains('Log in')",
                "button:contains('Sign in')",
                "[data-testid*='login']",
                "[data-testid*='submit']"
            ]
            
            login_button = None
            for selector in login_selectors:
                try:
                    login_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    break
                except NoSuchElementException:
                    continue
            
            if not login_button:
                logger.error("Could not find login button")
                return False
            
            login_button.click()
            logger.info("Clicked login button")
            
            # Wait for login to complete
            time.sleep(5)
            
            # Navigate to screening URL
            logger.info(f"Navigating to screening URL: {self.rayyan_url}")
            self.driver.get(self.rayyan_url)
            
            # Wait for page to load
            time.sleep(5)
            
            # Check if we're logged in and on the screening page
            current_url = self.driver.current_url
            if "screening" in current_url or "rayyan" in current_url:
                logger.info("Successfully navigated to Rayyan screening interface")
                
                # Additional wait for interface to fully load
                time.sleep(3)
                
                # Try to detect screening elements
                try:
                    screening_elements = self.driver.find_elements(By.XPATH, 
                        "//button[contains(@class, 'include') or contains(@class, 'exclude') or contains(@data-testid, 'include') or contains(@data-testid, 'exclude')]")
                    
                    if len(screening_elements) > 0:
                        logger.info("Screening interface elements detected")
                        print("Successfully logged into Rayyan!")
                        print("Screening interface loaded!")
                        return True
                    else:
                        logger.warning("No screening elements detected yet")
                        return True  # Still return True, might load later
                        
                except Exception as e:
                    logger.debug(f"Error checking screening elements: {str(e)}")
                    return True  # Still return True, manual verification will catch issues
            else:
                logger.warning(f"Unexpected URL after login: {current_url}")
                return False
                
        except Exception as e:
            logger.error(f"Error logging into Rayyan: {str(e)}")
            return False
    
    def wait_for_manual_setup(self):
        """Wait for user to manually complete setup if automatic login fails"""
        print("\n" + "="*60)
        print("MANUAL SETUP REQUIRED")
        print("="*60)
        print("Please complete the following steps manually:")
        print("1. Log in to Rayyan manually")
        print("2. Navigate to your screening interface")
        print("3. Make sure articles are visible and ready for screening")
        print("4. Press Enter when ready to start automated screening")
        
        input("\nPress Enter when Rayyan is ready for automated screening...")
        
        # Verify we can see screening elements
        try:
            screening_elements = self.driver.find_elements(By.XPATH, 
                "//button[contains(@class, 'include') or contains(@class, 'exclude') or contains(text(), 'Include') or contains(text(), 'Exclude')]")
            
            if len(screening_elements) > 0:
                logger.info("Rayyan screening interface detected")
                return True
            else:
                print("Cannot detect Rayyan screening interface.")
                print("Please make sure you're on the screening page with articles visible.")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying Rayyan interface: {str(e)}")
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
            
            # Log what we found
            logger.info(f"Extracted article - Title: {'Yes' if title else 'No'} ({len(title)} chars), Abstract: {'Yes' if abstract else 'No'} ({len(abstract)} chars)")
            
            if title or abstract:
                return {
                    'title': title,
                    'abstract': abstract,
                    'authors': '',
                    'journal': '',
                    'year': '',
                    'doi': '',
                    'pmid': ''
                }
            else:
                logger.warning("Could not extract any article content")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting current article: {str(e)}")
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
            logger.error(f"Error clicking {decision.value} button: {str(e)}")
            return False
    
    def _wait_for_rate_limit(self):
        """Wait if we've hit the rate limit"""
        elapsed_time = time.time() - self.start_time
        
        # If we've made 15 calls and it's been less than 60 seconds, wait
        if self.api_call_count >= self.calls_per_minute_limit and elapsed_time < 60:
            wait_time = 60 - elapsed_time
            print(f"\nRate limit reached ({self.calls_per_minute_limit} calls/minute). Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            # Reset counter after waiting
            self.api_call_count = 0
            self.start_time = time.time()
            print("Rate limit reset. Continuing...")
        
        # Reset counter if a full minute has passed
        elif elapsed_time >= 60:
            self.api_call_count = 0
            self.start_time = time.time()
    
    def process_articles(self):
        """Process articles and apply screening decisions"""
        print(f"\n{'='*60}")
        print("GROQ LLM CARDIAC RAYYAN SCREENING")
        print(f"{'='*60}")
        print(f"Max articles: {self.max_articles}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        
        processed = 0
        include_count = 0
        exclude_count = 0
        maybe_count = 0
        
        for i in range(self.max_articles):
            try:
                print(f"\n--- Processing Article {i + 1} ---")
                
                # Extract current article
                article_data = self.extract_current_article()
                if not article_data:
                    logger.warning(f"Could not extract article {i + 1}, skipping...")
                    time.sleep(2)
                    continue
                
                # Create Article object
                article = Article(
                    title=article_data['title'],
                    abstract=article_data['abstract'],
                    authors=article_data.get('authors', ''),
                    journal=article_data.get('journal', ''),
                    year=article_data.get('year', ''),
                    doi=article_data.get('doi', ''),
                    pmid=article_data.get('pmid', '')
                )
                
                # Screen with Groq - wait if rate limit reached
                try:
                    # Check if we need to wait for rate limit reset
                    self._wait_for_rate_limit()
                    
                    decision, confidence, reasoning = self.provider.screen_article(article, self.base_criteria)
                    self.api_call_count += 1
                    
                    # Display result
                    print(f"\nArticle {i + 1}: {article.title[:60]}...")
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
                        logger.warning(f"Failed to apply decision for article {i + 1}")
                    
                except Exception as e:
                    logger.error(f"Error screening article {i + 1}: {str(e)}")
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
                
                # Rate limiting is handled in _wait_for_rate_limit() before each API call
                
                # Wait before next article (clicking should auto-advance)
                time.sleep(1.5)
                
                # Progress update every 50 articles
                if (i + 1) % 50 == 0:
                    print(f"\n--- Progress Update ---")
                    print(f"Processed: {processed}/{i + 1}")
                    print(f"Include: {include_count}, Exclude: {exclude_count}, Maybe: {maybe_count}")
                
            except KeyboardInterrupt:
                print(f"\nScreening interrupted by user at article {i + 1}")
                break
            except Exception as e:
                logger.error(f"Unexpected error processing article {i + 1}: {str(e)}")
                time.sleep(2)
        
        # Final summary
        print(f"\n{'='*60}")
        print("GROQ SCREENING COMPLETED")
        print(f"{'='*60}")
        print(f"Total processed: {processed}")
        print(f"Include: {include_count}")
        print(f"Exclude: {exclude_count}")
        print(f"Maybe: {maybe_count}")
    
    def run(self):
        """Run the complete Groq Rayyan workflow"""
        try:
            print("="*60)
            print("GROQ LLM CARDIAC RAYYAN RUNNER")
            print("="*60)
            
            # Step 1: Setup driver
            print("Step 1: Setting up Chrome driver...")
            if not self.setup_driver():
                print("Failed to setup Chrome driver")
                return False
            
            # Step 2: Try automatic login, fallback to manual
            print("Step 2: Logging into Rayyan...")
            if not self.login_to_rayyan():
                print("Automatic login failed, switching to manual setup...")
                if not self.wait_for_manual_setup():
                    print("Manual setup failed")
                    return False
            
            # Step 3: Process articles
            print("Step 3: Starting article processing...")
            self.process_articles()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in Groq Rayyan workflow: {str(e)}")
            return False
        finally:
            if self.driver:
                input("\nPress Enter to close browser...")
                self.driver.quit()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Groq LLM Cardiac Rayyan Runner')
    parser.add_argument('--max-articles', '-n', type=int, default=50, help='Maximum articles to process')
    parser.add_argument('--confidence', '-c', type=float, default=0.7, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Initialize and run
    runner = GroqLLMCardiacRayyanRunner(args.max_articles, args.confidence)
    success = runner.run()
    
    if success:
        print("\nGroq Cardiac Rayyan screening completed successfully!")
    else:
        print("\nGroq Cardiac Rayyan screening failed. Check logs for details.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScreening interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        logger.error(f"Unexpected error in main: {str(e)}")