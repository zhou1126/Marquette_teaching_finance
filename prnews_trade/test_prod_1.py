import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import dspy
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import os
from typing import List, Dict, Optional
import asyncio
from threading import Lock
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.prnewswire.com/news-releases/all-public-company-news/"

# Load environment variables from .env (install python-dotenv if needed)
load_dotenv()
# Support common variable names for flexibility
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_KEY') or os.getenv('OPENAI')
if not OPENAI_API_KEY:
    raise RuntimeError('OpenAI API key not found. Set OPENAI_API_KEY in your .env file.')


class PricePredictionSignature(dspy.Signature):
    """
    You are a senior finance analyst. Analyze the news content and predict stock price impact.
    
    IMPORTANT: You must return ONLY a valid JSON array. No explanations, no additional text.
    
    For each company mentioned in the news, provide a prediction object with these exact fields:
    - company: string (company name, never leave empty)
    - ticker: string (stock ticker symbol like "AAPL", "MSFT", never leave empty) 
    - ai_comments: string (brief analysis of expected impact)
    - short_run_days: number (integer, days for short-term prediction, typically 1-30)
    - short_run_range_low_percent: number (float, lowest expected percentage change)
    - short_run_range_high_percent: number (float, highest expected percentage change)  
    - long_run_range_percent: number (float, long-term percentage change prediction)
    
    Example format:
    [
      {
        "company": "Apple Inc",
        "ticker": "AAPL",
        "ai_comments": "Positive earnings report likely to drive stock up",
        "short_run_days": 5,
        "short_run_range_low_percent": 2.5,
        "short_run_range_high_percent": 7.8,
        "long_run_range_percent": 12.5
      }
    ]
    
    If no companies with clear stock impact are mentioned, return: []
    
    CRITICAL: Return ONLY the JSON array, nothing else.
    """
    text: str = dspy.InputField(desc="news content to analyze")
    records_json: str = dspy.OutputField(desc="JSON array of predictions")


class AsyncPRNewsStockPredictor:
    def __init__(self, openai_api_key: str, data_file: str = "stock_predictions.csv", 
                 max_concurrent_predictions: int = 10):
        """Initialize the predictor with OpenAI API key and data file path."""
        self.data_file = data_file
        self.df_base = self._load_data()
        self.max_concurrent_predictions = max_concurrent_predictions
        self.data_lock = Lock()  # Thread-safe data operations
        self.api_key = str(openai_api_key)
        
        # Configure DSPY - ensure api_key is a string
        dspy.configure(
            lm=dspy.LM(
                model='openai/gpt-5-mini',
                api_key=self.api_key,
                temperature=1.0,  # Lower temperature for more consistent JSON output
                max_tokens=16000   # Reduced to ensure we don't hit limits
            )
        )
        
        # Define prediction signature
        self.predictor = dspy.ChainOfThought(PricePredictionSignature)
    
    def _load_data(self) -> pd.DataFrame:
        """Load existing data or create empty DataFrame."""
        if os.path.exists(self.data_file):
            return pd.read_csv(self.data_file)
        
        columns = [
            'date', 'time', 'title', 'link', 'content', 'has_exchange',
            'company', 'ticker', 'short_run_days', 'short_run_range_low_percent',
            'short_run_range_high_percent', 'long_run_range_percent', 'ai_comments'
        ]
        return pd.DataFrame(columns=columns)
    
    def _save_data(self):
        """Save data to CSV file (thread-safe)."""
        with self.data_lock:
            self.df_base.to_csv(self.data_file, index=False)
    
    def try_request(self, url: str, timeout: int = 10) -> Optional[requests.Response]:
        """Make HTTP request with error handling."""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            return r if r.status_code == 200 else None
        except Exception as e:
            logger.warning(f"Request failed for {url}: {e}")
            return None
    
    def extract_time_and_clean_title(self, raw_title: str) -> tuple:
        """Extract timestamp and clean title from raw title."""
        m = re.match(r"^(\d{1,2}:\d{2}\s*(?:[AP]M\s*)?ET)\s*(.*)", raw_title, re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return None, raw_title.strip()
    
    def get_article_content(self, article_url: str) -> str:
        """Fetch and extract article content."""
        resp = self.try_request(article_url)
        if not resp:
            return ""
        
        soup = BeautifulSoup(resp.content, 'html.parser')
        paragraphs = [
            p.get_text(strip=True)
            for p in soup.find_all('p')
            if len(p.get_text(strip=True)) > 50
        ]
        content = "\n\n".join(paragraphs[:])
        
        # Remove disclaimer section
        cut = content.upper().find("DISCLAIMER")
        if cut != -1:
            content = content[:cut].strip()
        
        return content
    
    def has_exchange_info(self, text: str) -> int:
        """Check if text contains exchange information."""
        pattern = re.compile(r'(NYSE|DOW|NASDAQ)', re.IGNORECASE)
        return 1 if pattern.search(text or "") else 0
    
    def fetch_new_articles(self, max_pages: int = 1, sleep_time: float = 0.2) -> pd.DataFrame:
        """Fetch new articles for today (sequential - no parallelization needed)."""
        now_ct = datetime.now(ZoneInfo("America/Chicago"))
        year, month, day = now_ct.year, now_ct.month, now_ct.day
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        
        logger.info(f"Fetching articles for {date_str}...")
        
        # Get existing links for today
        if not self.df_base.empty and "link" in self.df_base.columns and "date" in self.df_base.columns:
            seen_links = set(self.df_base.loc[self.df_base["date"] == date_str, "link"].astype(str))
        else:
            seen_links = set()
        
        collected = []
        seen_this_run = set()
        
        for page in range(1, max_pages + 1):
            list_url = f"{BASE_URL}?month={month:02d}&day={day:02d}&year={year}&page={page}&pagesize=100"
            resp = self.try_request(list_url)
            if not resp:
                break
            
            soup = BeautifulSoup(resp.content, "html.parser")
            anchors = soup.find_all("a", href=re.compile(r"^/news-releases/.*\.html$"))
            if not anchors:
                break
            
            for a in anchors:
                href = a.get("href")
                if not href:
                    continue
                
                full_url = "https://www.prnewswire.com" + href
                
                # Skip if already seen
                if full_url in seen_links or full_url in seen_this_run:
                    continue
                
                seen_this_run.add(full_url)
                
                raw_title = a.get_text(strip=True)
                news_time, title = self.extract_time_and_clean_title(raw_title)
                
                content = self.get_article_content(full_url)
                has_exch = self.has_exchange_info(content)
                
                collected.append({
                    "date": date_str,
                    "time": news_time,
                    "title": title,
                    "link": full_url,
                    "content": content,
                    "has_exchange": has_exch,
                })
                
                time.sleep(sleep_time)
                logger.info(f"Fetched: {title[:50]}...")
        
        logger.info(f"Successfully fetched {len(collected)} articles")
        return pd.DataFrame(collected)
    
    def test_sync_prediction(self, content: str = None):
        """
        Test synchronous prediction to debug DSPY issues.
        """
        if content is None:
            content = """
            Apple Inc. (NASDAQ: AAPL) reported strong quarterly earnings today, beating analyst expectations by 15%. 
            The company announced record iPhone sales and expanded its services revenue. 
            CEO Tim Cook expressed optimism about future growth prospects in artificial intelligence and augmented reality.
            The stock is expected to benefit from these positive developments in both the short and long term.
            """
        
        logger.info("=== TESTING SYNC PREDICTION ===")
        logger.info(f"Content preview: {content[:100]}...")
        
        try:
            # Test the predictor directly
            pred = self.predictor(text=content)
            
            logger.info(f"Prediction successful!")
            logger.info(f"Prediction type: {type(pred)}")
            logger.info(f"Available attributes: {[attr for attr in dir(pred) if not attr.startswith('_')]}")
            
            raw_response = getattr(pred, "records_json", None)
            logger.info(f"Raw response: {raw_response}")
            
            if raw_response:
                try:
                    import json
                    parsed = json.loads(raw_response)
                    logger.info(f"Successfully parsed JSON with {len(parsed)} predictions")
                    return parsed
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing failed: {e}")
                    return None
            else:
                logger.warning("No records_json attribute found")
                return None
                
        except Exception as e:
            logger.error(f"Sync prediction failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    async def async_predict_stock_impact(self, content: str, article_idx: int = None) -> List[Dict]:
        """
        Async version of stock impact prediction.
        Uses asyncio to handle the I/O-bound LLM API calls efficiently.
        """
        try:
            # Truncate content if too long to avoid token limits
            max_content_length = 16000  # Reasonable limit for analysis
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
                logger.info(f"Truncated content for article {article_idx} to {max_content_length} characters")
            
            # Run the synchronous DSPY prediction in an executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Create a wrapper function for better error handling
            def run_prediction():
                try:
                    # DSPY modules expect keyword arguments
                    return self.predictor(text=content)
                except Exception as e:
                    logger.error(f"DSPY prediction failed for article {article_idx}: {e}")
                    return None
            
            pred = await loop.run_in_executor(None, run_prediction)
            
            if pred is None:
                logger.warning(f"Prediction returned None for article {article_idx}")
                return []
            
            raw = getattr(pred, "records_json", None)
            
            if not raw:
                logger.warning(f"Empty or None response from DSPY for article {article_idx}")
                logger.info(f"Available attributes: {[attr for attr in dir(pred) if not attr.startswith('_')]}")
                return []
            
            # Log the raw response for debugging
            logger.info(f"Raw DSPY response for article {article_idx}: {raw[:300]}...")
            
            # Clean the response if it contains extra text
            raw = raw.strip()
            if not raw.startswith('['):
                # Try to extract JSON from the response
                json_match = re.search(r'\[.*\]', raw, re.DOTALL)
                if json_match:
                    raw = json_match.group(0)
                else:
                    logger.warning(f"No valid JSON array found in response for article {article_idx}")
                    logger.warning(f"Full response: {raw}")
                    return []
            
            # Parse JSON
            result = json.loads(raw)
            
            # Validate the result structure
            if not isinstance(result, list):
                logger.warning(f"Expected list but got {type(result)} for article {article_idx}")
                return []
            
            # Validate each prediction in the result
            valid_predictions = []
            for i, pred_dict in enumerate(result):
                if isinstance(pred_dict, dict):
                    # Ensure required fields exist
                    cleaned_pred = {
                        'company': pred_dict.get('company', 'N/A'),
                        'ticker': pred_dict.get('ticker', 'N/A'),
                        'ai_comments': pred_dict.get('ai_comments', ''),
                        'short_run_days': pred_dict.get('short_run_days'),
                        'short_run_range_low_percent': pred_dict.get('short_run_range_low_percent'),
                        'short_run_range_high_percent': pred_dict.get('short_run_range_high_percent'),
                        'long_run_range_percent': pred_dict.get('long_run_range_percent')
                    }
                    
                    # Convert numeric fields
                    for field in ['short_run_days', 'short_run_range_low_percent', 
                                'short_run_range_high_percent', 'long_run_range_percent']:
                        try:
                            if cleaned_pred[field] is not None:
                                cleaned_pred[field] = float(cleaned_pred[field])
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid {field} value: {cleaned_pred[field]} for article {article_idx}")
                            cleaned_pred[field] = None
                    
                    valid_predictions.append(cleaned_pred)
                else:
                    logger.warning(f"Invalid prediction format at index {i} for article {article_idx}: {pred_dict}")
            
            logger.info(f"Successfully generated {len(valid_predictions)} valid predictions for article {article_idx}")
            return valid_predictions
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for article {article_idx}: {e}")
            logger.error(f"Raw response: {raw[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Async prediction error for article {article_idx}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    async def debug_single_prediction(self, content: str) -> Dict:
        """
        Debug method to test a single prediction and see detailed output.
        """
        logger.info("=== DEBUG: Testing single prediction ===")
        logger.info(f"Content length: {len(content)}")
        logger.info(f"Content preview: {content[:200]}...")
        
        try:
            loop = asyncio.get_event_loop()
            
            def run_prediction():
                try:
                    # DSPY modules expect keyword arguments
                    return self.predictor(text=content)
                except Exception as e:
                    logger.error(f"Debug prediction failed in wrapper: {e}")
                    return None
            
            pred = await loop.run_in_executor(None, run_prediction)
            
            logger.info(f"Prediction object type: {type(pred)}")
            logger.info(f"Prediction attributes: {[attr for attr in dir(pred) if not attr.startswith('_')]}")
            
            raw = getattr(pred, "records_json", None)
            logger.info(f"Raw records_json: {raw}")
            
            if hasattr(pred, 'completions'):
                logger.info(f"Completions: {pred.completions}")
            
            return {
                'raw_response': raw,
                'pred_type': str(type(pred)),
                'attributes': [attr for attr in dir(pred) if not attr.startswith('_')]
            }
            
        except Exception as e:
            logger.error(f"Debug prediction failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'error': str(e)}
    
    async def test_prediction_with_simple_content(self):
        """
        Test prediction with a simple, clear example.
        """
        test_content = """
        Apple Inc. (NASDAQ: AAPL) reported strong quarterly earnings today, beating analyst expectations by 15%. 
        The company announced record iPhone sales and expanded its services revenue. 
        CEO Tim Cook expressed optimism about future growth prospects in artificial intelligence and augmented reality.
        The stock is expected to benefit from these positive developments in both the short and long term.
        """
        
        logger.info("=== TESTING WITH SIMPLE CONTENT (ASYNC) ===")
        result = await self.debug_single_prediction(test_content)
        return result
    
    def test_prediction_with_simple_content_sync(self):
        """
        Sync version of simple content test.
        """
        test_content = """
        Apple Inc. (NASDAQ: AAPL) reported strong quarterly earnings today, beating analyst expectations by 15%. 
        The company announced record iPhone sales and expanded its services revenue. 
        CEO Tim Cook expressed optimism about future growth prospects in artificial intelligence and augmented reality.
        The stock is expected to benefit from these positive developments in both the short and long term.
        """
        
        logger.info("=== TESTING WITH SIMPLE CONTENT (SYNC) ===")
        return self.test_sync_prediction(test_content)
    
    async def async_process_article_predictions(self, row_data: tuple) -> List[dict]:
        """
        Async version of processing predictions for a single article.
        """
        idx, row = row_data
        logger.info(f"Processing predictions for article {idx}: {row['title'][:50]}...")
        
        predictions = await self.async_predict_stock_impact(row['content'], idx)
        
        all_records = []
        if predictions:
            for pred in predictions:
                record = {
                    'date': row['date'],
                    'time': row['time'],
                    'title': row['title'],
                    'link': row['link'],
                    'content': row['content'],
                    'has_exchange': row['has_exchange'],
                    'company': pred.get('company', 'N/A'),
                    'ticker': pred.get('ticker', np.nan),
                    'short_run_days': pred.get('short_run_days', np.nan),
                    'short_run_range_low_percent': pred.get('short_run_range_low_percent', np.nan),
                    'short_run_range_high_percent': pred.get('short_run_range_high_percent', np.nan),
                    'long_run_range_percent': pred.get('long_run_range_percent', np.nan),
                    'ai_comments': pred.get('ai_comments', '')
                }
                all_records.append(record)
        else:
            # Add record without predictions
            record = {
                'date': row['date'],
                'time': row['time'],
                'title': row['title'],
                'link': row['link'],
                'content': row['content'],
                'has_exchange': row['has_exchange'],
                'company': 'N/A',
                'ticker': np.nan,
                'short_run_days': np.nan,
                'short_run_range_low_percent': np.nan,
                'short_run_range_high_percent': np.nan,
                'long_run_range_percent': np.nan,
                'ai_comments': ''
            }
            all_records.append(record)
        
        return all_records
    
    async def async_process_predictions_batch(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all article predictions asynchronously with concurrency control.
        """
        # Create semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(self.max_concurrent_predictions)
        
        async def process_with_semaphore(row_data):
            async with semaphore:
                return await self.async_process_article_predictions(row_data)
        
        # Create tasks for all articles
        tasks = [
            process_with_semaphore(row_data) 
            for row_data in articles_df.iterrows()
        ]
        
        logger.info(f"Starting async processing of {len(tasks)} articles with max {self.max_concurrent_predictions} concurrent predictions...")
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_predictions = []
        successful_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed with exception: {result}")
            elif result:
                all_predictions.extend(result)
                successful_count += 1
        
        logger.info(f"Successfully processed {successful_count}/{len(tasks)} articles")
        return pd.DataFrame(all_predictions)
    
    async def async_process_new_articles(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Async version: Process new articles and generate predictions using async LLM calls.
        """
        # Fetch new articles (sequential)
        df_new = self.fetch_new_articles()
        
        if df_new.empty:
            logger.info("No new articles found.")
            return pd.DataFrame()
        
        # Filter articles with exchange information
        df_focus = df_new[df_new['has_exchange'] == 1]
        logger.info(f"Found {len(df_focus)} articles with exchange information.")
        
        if limit:
            df_focus = df_focus.head(limit)
        
        # Run async prediction processing
        df_predictions = await self.async_process_predictions_batch(df_focus)
        
        # Add to base data (thread-safe)
        with self.data_lock:
            self.df_base = pd.concat([self.df_base, df_predictions], ignore_index=True)
        
        self._save_data()
        
        logger.info(f"Added {len(df_predictions)} predictions. Total records: {len(self.df_base)}")
        return df_predictions
    
    def process_new_articles(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Synchronous wrapper for processing new articles.
        This creates a new event loop if not already in an async context.
        """
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            logger.warning("Already in an event loop. Use async_process_new_articles() instead.")
            # If we're in a loop, we can't use asyncio.run()
            # Return empty DataFrame and suggest using the async version
            return pd.DataFrame()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self.async_process_new_articles(limit))
    
    async def process_existing_articles_async(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Async method to process predictions for an existing DataFrame of articles.
        Useful when you want to process a specific set of articles.
        """
        return await self.async_process_predictions_batch(articles_df)
    
    def process_existing_articles(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Synchronous wrapper for async processing of existing articles.
        """
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            logger.warning("Already in an event loop. Use process_existing_articles_async() instead.")
            return pd.DataFrame()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self.process_existing_articles_async(articles_df))
    
    async def process_predictions_in_chunks(self, articles_df: pd.DataFrame, chunk_size: int = 20) -> pd.DataFrame:
        """
        Process predictions in chunks to better manage API rate limits and memory usage.
        """
        all_predictions = []
        
        for i in range(0, len(articles_df), chunk_size):
            chunk = articles_df.iloc[i:i + chunk_size]
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(articles_df) + chunk_size - 1)//chunk_size} ({len(chunk)} articles)")
            
            chunk_predictions = await self.async_process_predictions_batch(chunk)
            all_predictions.append(chunk_predictions)
            
            # Small delay between chunks to be respectful to the API
            if i + chunk_size < len(articles_df):
                await asyncio.sleep(1)
        
        return pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    
    def get_predictions_by_ticker(self, ticker: str) -> pd.DataFrame:
        """Get all predictions for a specific ticker."""
        with self.data_lock:
            return self.df_base[self.df_base['ticker'].str.upper() == ticker.upper()].copy()
    
    def get_recent_predictions(self, days: int = 1) -> pd.DataFrame:
        """Get predictions from recent days."""
        recent_date = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        with self.data_lock:
            return self.df_base[self.df_base['date'] >= recent_date].copy()
    
    def export_predictions(self, filename: str):
        """Export predictions to CSV."""
        with self.data_lock:
            self.df_base.to_csv(filename, index=False)
        logger.info(f"Exported {len(self.df_base)} records to {filename}")


# Usage functions
async def async_main():
    """Async main function demonstrating async LLM processing."""
    # Initialize predictor with API key read from .env
    api_key = OPENAI_API_KEY
    
    # Configure concurrent predictions based on your API rate limits
    # Start with 3-5 concurrent requests for debugging
    predictor = AsyncPRNewsStockPredictor(
        api_key, 
        max_concurrent_predictions=3  # Reduced for debugging
    )
    
    # DEBUG: Test synchronous prediction first
    logger.info("=== DEBUGGING SYNC PREDICTION ===")
    sync_result = predictor.test_prediction_with_simple_content_sync()
    logger.info(f"Sync test result: {sync_result}")
    
    if sync_result:
        logger.info("✅ Sync prediction working! Now testing async...")
        # DEBUG: Test with simple content first
        logger.info("=== DEBUGGING ASYNC PREDICTIONS ===")
        debug_result = await predictor.test_prediction_with_simple_content()
        logger.info(f"Async debug result: {debug_result}")
    else:
        logger.error("❌ Sync prediction failed. Check DSPY configuration.")
        return pd.DataFrame()
    
    # Only proceed if basic prediction works
    if sync_result:
        # Method 1: Process new articles using async version
        predictions = await predictor.async_process_new_articles()  # Very limited for debugging
        
        # Show results with better debugging
        if not predictions.empty:
            print("\n=== Recent Predictions (Async LLM Processing) ===")
            
            # Check for all NaN records
            nan_records = predictions[(predictions['company'] == 'N/A') & (predictions['ticker'].isna())]
            valid_records = predictions[~((predictions['company'] == 'N/A') & (predictions['ticker'].isna()))]
            
            print(f"Total predictions: {len(predictions)}")
            print(f"Valid predictions: {len(valid_records)}")
            print(f"NaN/empty predictions: {len(nan_records)}")
            
            if len(valid_records) > 0:
                print("\n--- Valid Predictions ---")
                for _, row in valid_records.iterrows():
                    print(f"Company: {row['company']}")
                    print(f"Ticker: {row['ticker']}")
                    print(f"Short-term ({row['short_run_days']} days): {row['short_run_range_low_percent']}% to {row['short_run_range_high_percent']}%")
                    print(f"Long-term: {row['long_run_range_percent']}%")
                    print(f"Analysis: {row['ai_comments']}")
                    print("-" * 50)
            else:
                print("\n❌ No valid predictions found!")
                print("This suggests the LLM is not returning properly formatted responses.")
                
                # Show a sample of the problematic data
                if len(predictions) > 0:
                    print("\n--- Sample of Raw Data ---")
                    sample_row = predictions.iloc[0]
                    print(f"Sample title: {sample_row['title']}")
                    print(f"Sample content length: {len(sample_row['content'])}")
                    print(f"Sample content preview: {sample_row['content'][:200]}...")
        else:
            print("❌ No predictions generated at all!")
        
        # Export results
        predictor.export_predictions("debug_stock_predictions_export.csv")
        return predictions
    else:
        return pd.DataFrame()


def main():
    """Synchronous wrapper for the async main function."""
    return asyncio.run(async_main())


def sync_main():
    """Synchronous main function for non-async environments."""
    # Initialize predictor with API key read from .env
    api_key = OPENAI_API_KEY

    predictor = AsyncPRNewsStockPredictor(
        api_key,
        max_concurrent_predictions=5
    )
    
    # Use the synchronous wrapper (creates its own event loop)
    predictions = predictor.process_new_articles()
    
    # Show results
    if not predictions.empty:
        print("\n=== Recent Predictions (Sync Wrapper) ===")
        for _, row in predictions.iterrows():
            print(f"Company: {row['company']}")
            print(f"Ticker: {row['ticker']}")
            print(f"Short-term ({row['short_run_days']} days): {row['short_run_range_low_percent']}% to {row['short_run_range_high_percent']}%")
            print(f"Long-term: {row['long_run_range_percent']}%")
            print(f"Analysis: {row['ai_comments']}")
            print("-" * 50)
    
    predictor.export_predictions("sync_stock_predictions_export.csv")
    return predictions


if __name__ == "__main__":
    results_out = main()