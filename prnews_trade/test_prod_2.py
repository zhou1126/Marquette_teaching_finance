import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import dspy
import json
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import os
from typing import List, Dict, Optional
import asyncio
from threading import Lock
import logging
import sqlite3
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.prnewswire.com/news-releases/all-public-company-news/"

# Load environment variables from .env (install python-dotenv if needed)
load_dotenv()
# Flexible env names
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_KEY') or os.getenv('OPENAI')
FMP_API_KEY = os.getenv('FMP_API_KEY') or os.getenv('FMP_KEY') or os.getenv('API_KEY')
if not OPENAI_API_KEY:
    raise RuntimeError('OpenAI API key not found. Set OPENAI_API_KEY in your .env file.')


class EnhancedPricePredictionSignature(dspy.Signature):
    """
    You are a senior finance analyst. Analyze the news content and predict stock price impact.
    
    IMPORTANT: You must return ONLY a valid JSON array. No explanations, no additional text.
    
    For each company mentioned in the news, provide a prediction object with these exact fields:
    - company: string (company name, never leave empty)
    - ticker: string (stock ticker symbol like "AAPL", "MSFT", never leave empty)
    - industry: string (e.g., "Technology", "Healthcare", "Finance", "Energy", "Consumer Goods", "Other")
    - industry_subcategory: string (specific sector within the industry)
    - sentiment: string ("Positive", "Negative", or "Neutral")
    - ai_comments: string (brief analysis of expected impact)
    - short_run_days: number (integer, days for short-term prediction, typically 1-30)
    - short_run_range_low_percent: number (float, lowest expected percentage change, can be negative)
    - short_run_range_high_percent: number (float, highest expected percentage change)  
    - long_run_range_percent: number (float, long-term percentage change prediction for 6-12 months)
    - confidence_score: number (0-100, your confidence in this prediction)
    
    Example format:
    [
      {
        "company": "Apple Inc",
        "ticker": "AAPL",
        "industry": "Technology",
        "industry_subcategory": "Consumer Electronics",
        "sentiment": "Positive",
        "ai_comments": "Positive earnings report likely to drive stock up 5-8% in next 2 weeks",
        "short_run_days": 10,
        "short_run_range_low_percent": 2.5,
        "short_run_range_high_percent": 7.8,
        "long_run_range_percent": 12.5,
        "confidence_score": 85
      }
    ]
    
    If no companies with clear stock impact are mentioned, return: []
    
    CRITICAL: Return ONLY the JSON array, nothing else. All numeric fields must have values.
    """
    text: str = dspy.InputField(desc="news content to analyze")
    records_json: str = dspy.OutputField(desc="JSON array of predictions with all required fields")


class StockPriceAPI:
    """Handler for fetching historical stock prices."""
    
    def __init__(self, api_key: str, api_base_url: str = "https://financialmodelingprep.com/api/v3"):
        self.api_key = api_key
        self.api_base_url = api_base_url
    
    def get_stock_price_data(self, ticker: str, days: int = 100) -> pd.DataFrame:
        """Fetch historical daily stock price data."""
        try:
            url = f"{self.api_base_url}/historical-price-full/{ticker}"
            params = {"timeseries": days, "apikey": self.api_key}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Check for various error responses
            if isinstance(data, dict) and "Error Message" in data:
                logger.warning(f"API error for {ticker}: {data['Error Message']}")
                return pd.DataFrame()
            
            if "historical" not in data or not data["historical"]:
                logger.warning(f"No historical data found for {ticker} (may be OTC, delisted, or invalid ticker)")
                return pd.DataFrame()

            df = pd.DataFrame(data["historical"])
            df['ticker'] = ticker
            df['date'] = pd.to_datetime(df['date'])
            return df[['date', 'open', 'close', 'high', 'low', 'volume', 'ticker']].copy()

        except requests.exceptions.RequestException as error:
            logger.warning(f"Network error fetching data for {ticker}: {error}")
            return pd.DataFrame()
        except Exception as error:
            logger.warning(f"Error fetching data for {ticker}: {error}")
            return pd.DataFrame()
    
    def calculate_price_change(self, df: pd.DataFrame, start_date: str, end_date: str = None) -> float:
        """Calculate percentage price change between dates."""
        if df.empty:
            return None
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        start_date = pd.to_datetime(start_date)
        
        if end_date:
            end_date = pd.to_datetime(end_date)
        else:
            end_date = df['date'].max()
        
        start_row = df.iloc[(df['date'] - start_date).abs().argsort()[:1]]
        end_row = df.iloc[(df['date'] - end_date).abs().argsort()[:1]]
        
        if start_row.empty or end_row.empty:
            return None
        
        start_price = start_row['close'].values[0]
        end_price = end_row['close'].values[0]
        
        return ((end_price - start_price) / start_price) * 100


class DatabaseManager:
    """Manages SQLite database for historical news storage."""
    
    def __init__(self, db_path: str = "stock_news_history.db"):
        self.db_path = db_path
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database with proper schema."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                time TEXT,
                title TEXT NOT NULL,
                link TEXT UNIQUE NOT NULL,
                content TEXT,
                has_exchange INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER,
                company TEXT,
                ticker TEXT,
                industry TEXT,
                industry_subcategory TEXT,
                sentiment TEXT,
                short_run_days REAL,
                short_run_range_low_percent REAL,
                short_run_range_high_percent REAL,
                long_run_range_percent REAL,
                confidence_score REAL,
                ai_comments TEXT,
                prediction_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (article_id) REFERENCES news_articles (id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS correlation_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                analysis_date TEXT NOT NULL,
                lookback_days INTEGER,
                total_news_items INTEGER,
                avg_predicted_growth REAL,
                actual_growth REAL,
                correlation_coefficient REAL,
                prediction_accuracy REAL,
                analysis_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON news_articles(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON stock_predictions(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_industry ON stock_predictions(industry)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_long_run ON stock_predictions(long_run_range_percent)")
        
        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    def insert_article(self, article_data: dict) -> int:
        """Insert article and return article_id."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM news_articles WHERE link = ?", (article_data['link'],))
        existing = cursor.fetchone()
        if existing:
            return existing[0]
        
        cursor.execute("""
            INSERT INTO news_articles (date, time, title, link, content, has_exchange)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            article_data['date'], article_data.get('time'), article_data['title'],
            article_data['link'], article_data.get('content'), article_data.get('has_exchange', 0)
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_prediction(self, article_id: int, prediction: dict):
        """Insert stock prediction."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO stock_predictions (
                article_id, company, ticker, industry, industry_subcategory,
                sentiment, short_run_days, short_run_range_low_percent,
                short_run_range_high_percent, long_run_range_percent,
                confidence_score, ai_comments, prediction_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            article_id, prediction.get('company'), prediction.get('ticker'),
            prediction.get('industry'), prediction.get('industry_subcategory'),
            prediction.get('sentiment'), prediction.get('short_run_days'),
            prediction.get('short_run_range_low_percent'), prediction.get('short_run_range_high_percent'),
            prediction.get('long_run_range_percent'), prediction.get('confidence_score'),
            prediction.get('ai_comments'), datetime.now().strftime('%Y-%m-%d')
        ))
        self.conn.commit()
    
    def insert_correlation_analysis(self, analysis: dict):
        """Insert correlation analysis results."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO correlation_analysis (
                ticker, analysis_date, lookback_days, total_news_items,
                avg_predicted_growth, actual_growth, correlation_coefficient,
                prediction_accuracy, analysis_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis['ticker'], analysis['analysis_date'], analysis['lookback_days'],
            analysis['total_news_items'], analysis['avg_predicted_growth'],
            analysis['actual_growth'], analysis['correlation_coefficient'],
            analysis['prediction_accuracy'], analysis['analysis_summary']
        ))
        self.conn.commit()
    
    def get_high_growth_predictions(self, threshold: float = 10.0) -> pd.DataFrame:
        """Get predictions with long-term growth > threshold%."""
        query = """
            SELECT DISTINCT p.ticker, p.company, 
                   COUNT(*) as prediction_count,
                   AVG(p.long_run_range_percent) as avg_predicted_growth,
                   MAX(a.date) as latest_news_date
            FROM stock_predictions p
            JOIN news_articles a ON p.article_id = a.id
            WHERE p.long_run_range_percent > ?
            GROUP BY p.ticker, p.company
            ORDER BY avg_predicted_growth DESC
        """
        return pd.read_sql_query(query, self.conn, params=(threshold,))
    
    def get_ticker_predictions_history(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """Get all historical predictions for a ticker."""
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        query = """
            SELECT a.date, a.time, a.title, a.link,
                   p.company, p.ticker, p.industry, p.sentiment,
                   p.long_run_range_percent, p.confidence_score, p.ai_comments
            FROM stock_predictions p
            JOIN news_articles a ON p.article_id = a.id
            WHERE UPPER(p.ticker) = UPPER(?) AND a.date >= ?
            ORDER BY a.date ASC
        """
        return pd.read_sql_query(query, self.conn, params=(ticker, cutoff_date))
    
    def get_today_articles(self) -> pd.DataFrame:
        """Get all articles from today."""
        today = datetime.now().strftime('%Y-%m-%d')
        query = """
            SELECT a.*, p.company, p.ticker, p.industry, p.industry_subcategory,
                   p.sentiment, p.short_run_days, p.short_run_range_low_percent,
                   p.short_run_range_high_percent, p.long_run_range_percent,
                   p.confidence_score, p.ai_comments
            FROM news_articles a
            LEFT JOIN stock_predictions p ON a.id = p.article_id
            WHERE a.date = ?
            ORDER BY a.time DESC
        """
        return pd.read_sql_query(query, self.conn, params=(today,))
    
    def get_industry_summary(self, date: Optional[str] = None) -> pd.DataFrame:
        """Get summary statistics by industry."""
        if date:
            query = """
                SELECT p.industry, COUNT(*) as article_count,
                       AVG(p.confidence_score) as avg_confidence,
                       AVG(p.long_run_range_percent) as avg_long_term_prediction
                FROM stock_predictions p
                JOIN news_articles a ON p.article_id = a.id
                WHERE a.date = ?
                GROUP BY p.industry
                ORDER BY article_count DESC
            """
            return pd.read_sql_query(query, self.conn, params=(date,))
        else:
            query = """
                SELECT p.industry, COUNT(*) as article_count,
                       AVG(p.confidence_score) as avg_confidence,
                       AVG(p.long_run_range_percent) as avg_long_term_prediction
                FROM stock_predictions p
                GROUP BY p.industry
                ORDER BY article_count DESC
            """
            return pd.read_sql_query(query, self.conn)
    
    def get_correlation_history(self, ticker: str = None) -> pd.DataFrame:
        """Get correlation analysis history."""
        if ticker:
            query = """
                SELECT * FROM correlation_analysis 
                WHERE UPPER(ticker) = UPPER(?)
                ORDER BY analysis_date DESC
            """
            return pd.read_sql_query(query, self.conn, params=(ticker,))
        else:
            query = """
                SELECT * FROM correlation_analysis 
                ORDER BY analysis_date DESC
            """
            return pd.read_sql_query(query, self.conn)
    
    def get_by_ticker(self, ticker: str) -> pd.DataFrame:
        """Get all predictions for a specific ticker."""
        query = """
            SELECT a.date, a.time, a.title, a.link,
                   p.company, p.ticker, p.industry, p.sentiment,
                   p.short_run_days, p.short_run_range_low_percent,
                   p.short_run_range_high_percent, p.long_run_range_percent,
                   p.confidence_score, p.ai_comments
            FROM stock_predictions p
            JOIN news_articles a ON p.article_id = a.id
            WHERE UPPER(p.ticker) = UPPER(?)
            ORDER BY a.date DESC, a.time DESC
        """
        return pd.read_sql_query(query, self.conn, params=(ticker,))
    
    def get_by_industry(self, industry: str) -> pd.DataFrame:
        """Get all predictions for a specific industry."""
        query = """
            SELECT a.date, a.time, a.title, a.link,
                   p.company, p.ticker, p.industry, p.industry_subcategory,
                   p.sentiment, p.confidence_score, p.ai_comments
            FROM stock_predictions p
            JOIN news_articles a ON p.article_id = a.id
            WHERE UPPER(p.industry) = UPPER(?)
            ORDER BY a.date DESC, a.time DESC
        """
        return pd.read_sql_query(query, self.conn, params=(industry,))
    
    def get_articles_by_date(self, date: str) -> pd.DataFrame:
        """Get all articles from a specific date."""
        query = """
            SELECT a.*, p.company, p.ticker, p.industry, p.industry_subcategory,
                   p.sentiment, p.short_run_days, p.short_run_range_low_percent,
                   p.short_run_range_high_percent, p.long_run_range_percent,
                   p.confidence_score, p.ai_comments
            FROM news_articles a
            LEFT JOIN stock_predictions p ON a.id = p.article_id
            WHERE a.date = ?
            ORDER BY a.time DESC
        """
        return pd.read_sql_query(query, self.conn, params=(date,))
    
    def get_recent_predictions(self, days: int = 7) -> pd.DataFrame:
        """Get predictions from recent days."""
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        query = """
            SELECT a.date, a.time, a.title, p.company, p.ticker, p.industry,
                   p.sentiment, p.long_run_range_percent, p.confidence_score
            FROM stock_predictions p
            JOIN news_articles a ON p.article_id = a.id
            WHERE a.date >= ?
            ORDER BY a.date DESC, p.long_run_range_percent DESC
        """
        return pd.read_sql_query(query, self.conn, params=(cutoff_date,))
    
    def get_top_predictions_today(self, limit: int = 10) -> pd.DataFrame:
        """Get top predictions by confidence score for today."""
        today = datetime.now().strftime('%Y-%m-%d')
        query = """
            SELECT a.title, p.company, p.ticker, p.industry, p.sentiment,
                   p.long_run_range_percent, p.confidence_score, p.ai_comments
            FROM stock_predictions p
            JOIN news_articles a ON p.article_id = a.id
            WHERE a.date = ? AND p.ticker != 'N/A'
            ORDER BY p.confidence_score DESC, p.long_run_range_percent DESC
            LIMIT ?
        """
        return pd.read_sql_query(query, self.conn, params=(today, limit))
    
    def get_sentiment_distribution(self, days: int = 30) -> pd.DataFrame:
        """Get sentiment distribution over recent days."""
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        query = """
            SELECT p.sentiment, COUNT(*) as count,
                   AVG(p.long_run_range_percent) as avg_prediction
            FROM stock_predictions p
            JOIN news_articles a ON p.article_id = a.id
            WHERE a.date >= ?
            GROUP BY p.sentiment
            ORDER BY count DESC
        """
        return pd.read_sql_query(query, self.conn, params=(cutoff_date,))
    
    def close(self):
        if self.conn:
            self.conn.close()


class DeepDiveAnalyzer:
    """Analyzes correlation between news predictions and actual stock performance."""
    
    def __init__(self, db: DatabaseManager, price_api: StockPriceAPI):
        self.db = db
        self.price_api = price_api
    
    def analyze_ticker_correlation(self, ticker: str, lookback_days: int = 180) -> Dict:
        """Deep dive: correlate news predictions with actual price movement."""
        logger.info(f"Starting deep dive analysis for {ticker}...")
        
        predictions_df = self.db.get_ticker_predictions_history(ticker, lookback_days)
        
        if predictions_df.empty:
            logger.warning(f"No predictions found for {ticker}")
            return None
        
        logger.info(f"Found {len(predictions_df)} news items for {ticker}")
        
        price_df = self.price_api.get_stock_price_data(ticker, days=lookback_days + 30)
        
        if price_df.empty:
            logger.warning(f"Skipping {ticker} - no price data available (may be OTC, delisted, or invalid ticker)")
            return None
        
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        predictions_df = predictions_df.sort_values('date')
        
        actual_growths = []
        prediction_dates = []
        predicted_growths = []
        
        for _, row in predictions_df.iterrows():
            news_date = row['date']
            predicted_growth = row['long_run_range_percent']
            
            if pd.isna(predicted_growth):
                continue
            
            future_date = news_date + timedelta(days=30)
            actual_growth = self.price_api.calculate_price_change(
                price_df, 
                news_date.strftime('%Y-%m-%d'),
                future_date.strftime('%Y-%m-%d')
            )
            
            if actual_growth is not None:
                actual_growths.append(actual_growth)
                prediction_dates.append(news_date)
                predicted_growths.append(predicted_growth)
        
        if len(actual_growths) < 2:
            logger.warning(f"Insufficient data points for correlation analysis on {ticker}")
            return None
        
        from scipy.stats import pearsonr
        correlation, p_value = pearsonr(predicted_growths, actual_growths)
        
        mae = np.mean(np.abs(np.array(predicted_growths) - np.array(actual_growths)))
        accuracy = max(0, 100 - mae)
        
        avg_predicted = np.mean(predicted_growths)
        avg_actual = np.mean(actual_growths)
        
        summary = f"""
                    Deep Dive Analysis for {ticker}:
                    - Total news items: {len(predictions_df)}
                    - Valid pairs: {len(actual_growths)}
                    - Avg predicted growth: {avg_predicted:.2f}%
                    - Avg actual growth (30d): {avg_actual:.2f}%
                    - Correlation: {correlation:.3f} (p={p_value:.4f})
                    - Prediction accuracy: {accuracy:.1f}%
                    - Bias: {avg_predicted - avg_actual:.2f}%
                    """
        
        analysis_result = {
            'ticker': ticker,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'lookback_days': lookback_days,
            'total_news_items': len(predictions_df),
            'avg_predicted_growth': float(avg_predicted),
            'actual_growth': float(avg_actual),
            'correlation_coefficient': float(correlation),
            'prediction_accuracy': float(accuracy),
            'analysis_summary': summary.strip()
        }
        
        self.db.insert_correlation_analysis(analysis_result)
        logger.info(summary)
        
        return analysis_result
    
    def batch_analyze_high_growth(self, threshold: float = 10.0, top_n: int = 10):
        """Analyze all stocks with predicted growth > threshold."""
        high_growth_df = self.db.get_high_growth_predictions(threshold)
        
        if high_growth_df.empty:
            logger.info(f"No stocks with predicted growth > {threshold}%")
            return []
        
        logger.info(f"Found {len(high_growth_df)} high-growth stocks")
        
        results = []
        skipped_tickers = []
        
        for _, row in high_growth_df.head(top_n).iterrows():
            ticker = row['ticker']
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing {ticker} - Predicted: {row['avg_predicted_growth']:.2f}%")
            logger.info(f"{'='*60}")
            
            analysis = self.analyze_ticker_correlation(ticker)
            if analysis:
                results.append(analysis)
            else:
                skipped_tickers.append(ticker)
                logger.info(f"Skipped {ticker} (no price data available)")
            
            time.sleep(1)
        
        # Summary of skipped tickers
        if skipped_tickers:
            logger.info(f"\n{'='*60}")
            logger.info(f"SKIPPED TICKERS (no price data):")
            for ticker in skipped_tickers:
                logger.info(f"  - {ticker}")
            logger.info(f"{'='*60}")
        
        logger.info(f"\nSuccessfully analyzed {len(results)} out of {len(high_growth_df.head(top_n))} stocks")
        
        return results


class EnhancedPRNewsStockPredictor:
    def __init__(self, openai_api_key: str, fmp_api_key: str = None,
                 db_path: str = "stock_news_history.db", max_concurrent_predictions: int = 10):
        self.db = DatabaseManager(db_path)
        self.max_concurrent_predictions = max_concurrent_predictions
        self.api_key = str(openai_api_key)
        
        self.price_api = None
        if fmp_api_key:
            self.price_api = StockPriceAPI(fmp_api_key)
            self.analyzer = DeepDiveAnalyzer(self.db, self.price_api)
        
        dspy.configure(
            lm=dspy.LM(
                model='openai/gpt-4o-mini',
                api_key=self.api_key,
                temperature=0.7,
                max_tokens=16000
            )
        )
        
        self.predictor = dspy.ChainOfThought(EnhancedPricePredictionSignature)
    
    def get_today_summary(self) -> Dict:
        """Get summary of today's news."""
        today_df = self.db.get_today_articles()
        
        if today_df.empty:
            return {"message": "No articles processed for today"}
        
        industry_summary = self.db.get_industry_summary(datetime.now().strftime('%Y-%m-%d'))
        
        summary = {
            "total_articles": len(today_df),
            "total_predictions": len(today_df[today_df['ticker'].notna()]),
            "industries_covered": industry_summary.to_dict('records') if not industry_summary.empty else [],
            "top_sentiment": today_df['sentiment'].value_counts().to_dict() if 'sentiment' in today_df.columns else {},
            "avg_confidence": today_df['confidence_score'].mean() if 'confidence_score' in today_df.columns else None
        }
        
        return summary
    
    def export_today_to_csv(self, filename: str = None):
        """Export today's data to CSV."""
        if filename is None:
            today_str = datetime.now().strftime('%Y%m%d')
            filename = f"news_predictions_{today_str}.csv"
        
        today_df = self.db.get_today_articles()
        today_df.to_csv(filename, index=False)
        logger.info(f"Exported {len(today_df)} records to {filename}")
        return filename
    
    def run_deep_dive_analysis(self, growth_threshold: float = 10.0, top_n: int = 10):
        """
        Run deep dive correlation analysis on high-growth predictions.
        This is the main method to call for the correlation analysis.
        """
        if not self.price_api:
            logger.error("Price API not initialized. Please provide FMP API key.")
            return None
        
        logger.info(f"\n{'='*80}")
        logger.info(f"DEEP DIVE ANALYSIS: Stocks with predicted growth > {growth_threshold}%")
        logger.info(f"{'='*80}\n")
        
        results = self.analyzer.batch_analyze_high_growth(growth_threshold, top_n)
        
        # Create summary report
        if results:
            self._generate_summary_report(results)
        else:
            logger.warning("No stocks could be analyzed. Check if tickers are valid or have available price data.")
        
        return results
    
    def _generate_summary_report(self, analyses: List[Dict]):
        """Generate a summary report of all analyses."""
        report_filename = f"deep_dive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DEEP DIVE ANALYSIS SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for analysis in analyses:
                f.write(f"\n{'-'*80}\n")
                f.write(analysis['analysis_summary'])
                f.write(f"\n{'-'*80}\n")
            
            # Overall statistics
            f.write("\n\n" + "="*80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("="*80 + "\n")
            
            avg_correlation = np.mean([a['correlation_coefficient'] for a in analyses])
            avg_accuracy = np.mean([a['prediction_accuracy'] for a in analyses])
            
            f.write(f"Total stocks analyzed: {len(analyses)}\n")
            f.write(f"Average correlation: {avg_correlation:.3f}\n")
            f.write(f"Average prediction accuracy: {avg_accuracy:.1f}%\n")
            
            # Best and worst performers
            if len(analyses) > 0:
                best = max(analyses, key=lambda x: x['correlation_coefficient'])
                worst = min(analyses, key=lambda x: x['correlation_coefficient'])
                
                f.write(f"\nBest correlation: {best['ticker']} ({best['correlation_coefficient']:.3f})\n")
                f.write(f"Worst correlation: {worst['ticker']} ({worst['correlation_coefficient']:.3f})\n")
        
        logger.info(f"Summary report saved to {report_filename}")
    
    def try_request(self, url: str, timeout: int = 10) -> Optional[requests.Response]:
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            return r if r.status_code == 200 else None
        except Exception as e:
            logger.warning(f"Request failed: {e}")
            return None
    
    def extract_time_and_clean_title(self, raw_title: str) -> tuple:
        m = re.match(r"^(\d{1,2}:\d{2}\s*(?:[AP]M\s*)?ET)\s*(.*)", raw_title, re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return None, raw_title.strip()
    
    def get_article_content(self, article_url: str) -> str:
        resp = self.try_request(article_url)
        if not resp:
            return ""
        
        soup = BeautifulSoup(resp.content, 'html.parser')
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 50]
        content = "\n\n".join(paragraphs)
        
        cut = content.upper().find("DISCLAIMER")
        if cut != -1:
            content = content[:cut].strip()
        
        return content
    
    def has_exchange_info(self, text: str) -> int:
        pattern = re.compile(r'(NYSE|DOW|NASDAQ)', re.IGNORECASE)
        return 1 if pattern.search(text or "") else 0
    
    def fetch_today_articles(self, max_pages: int = 1) -> pd.DataFrame:
        now_ct = datetime.now(ZoneInfo("America/Chicago"))
        year, month, day = now_ct.year, now_ct.month, now_ct.day
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        
        logger.info(f"Fetching articles for {date_str}...")
        
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
                if full_url in seen_this_run:
                    continue
                
                seen_this_run.add(full_url)
                
                raw_title = a.get_text(strip=True)
                news_time, title = self.extract_time_and_clean_title(raw_title)
                
                content = self.get_article_content(full_url)
                has_exch = self.has_exchange_info(content)
                
                collected.append({
                    "date": date_str, "time": news_time, "title": title,
                    "link": full_url, "content": content, "has_exchange": has_exch
                })
                
                time.sleep(0.2)
                logger.info(f"Fetched: {title[:50]}...")
        
        logger.info(f"Successfully fetched {len(collected)} articles")
        return pd.DataFrame(collected)
    
    async def async_predict_stock_impact(self, content: str, article_idx: int = None) -> List[Dict]:
        try:
            max_content_length = 16000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            loop = asyncio.get_event_loop()
            
            def run_prediction():
                try:
                    return self.predictor(text=content)
                except Exception as e:
                    logger.error(f"Prediction failed: {e}")
                    return None
            
            pred = await loop.run_in_executor(None, run_prediction)
            
            if pred is None:
                return []
            
            raw = getattr(pred, "records_json", None)
            if not raw:
                return []
            
            raw = raw.strip()
            if not raw.startswith('['):
                json_match = re.search(r'\[.*\]', raw, re.DOTALL)
                if json_match:
                    raw = json_match.group(0)
                else:
                    return []
            
            result = json.loads(raw)
            if not isinstance(result, list):
                return []
            
            valid_predictions = []
            for pred_dict in result:
                if isinstance(pred_dict, dict):
                    cleaned_pred = {
                        'company': pred_dict.get('company', 'N/A'),
                        'ticker': pred_dict.get('ticker', 'N/A'),
                        'industry': pred_dict.get('industry', 'Other'),
                        'industry_subcategory': pred_dict.get('industry_subcategory', ''),
                        'sentiment': pred_dict.get('sentiment', 'Neutral'),
                        'ai_comments': pred_dict.get('ai_comments', ''),
                        'short_run_days': pred_dict.get('short_run_days'),
                        'short_run_range_low_percent': pred_dict.get('short_run_range_low_percent'),
                        'short_run_range_high_percent': pred_dict.get('short_run_range_high_percent'),
                        'long_run_range_percent': pred_dict.get('long_run_range_percent'),
                        'confidence_score': pred_dict.get('confidence_score')
                    }
                    
                    for field in ['short_run_days', 'short_run_range_low_percent', 
                                'short_run_range_high_percent', 'long_run_range_percent', 'confidence_score']:
                        try:
                            if cleaned_pred[field] is not None:
                                cleaned_pred[field] = float(cleaned_pred[field])
                        except (ValueError, TypeError):
                            cleaned_pred[field] = None
                    
                    valid_predictions.append(cleaned_pred)
            
            return valid_predictions
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return []
    
    async def async_process_article(self, row_data: tuple) -> List[dict]:
        idx, row = row_data
        logger.info(f"Processing article {idx}: {row['title'][:50]}...")
        
        article_id = self.db.insert_article(row.to_dict())
        predictions = await self.async_predict_stock_impact(row['content'], idx)
        
        for pred in predictions:
            self.db.insert_prediction(article_id, pred)
        
        all_records = []
        if predictions:
            for pred in predictions:
                record = row.to_dict()
                record.update(pred)
                all_records.append(record)
        else:
            record = row.to_dict()
            record.update({
                'company': 'N/A', 'ticker': np.nan, 'industry': 'Other',
                'industry_subcategory': '', 'sentiment': 'Neutral',
                'short_run_days': np.nan, 'short_run_range_low_percent': np.nan,
                'short_run_range_high_percent': np.nan, 'long_run_range_percent': np.nan,
                'confidence_score': np.nan, 'ai_comments': ''
            })
            all_records.append(record)
        
        return all_records
    
    async def async_process_articles_batch(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        semaphore = asyncio.Semaphore(self.max_concurrent_predictions)
        
        async def process_with_semaphore(row_data):
            async with semaphore:
                return await self.async_process_article(row_data)
        
        tasks = [process_with_semaphore(row_data) for row_data in articles_df.iterrows()]
        
        logger.info(f"Processing {len(tasks)} articles...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_predictions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {result}")
            elif result:
                all_predictions.extend(result)
        
        return pd.DataFrame(all_predictions)
    
    async def async_process_today(self, limit: Optional[int] = None) -> pd.DataFrame:
        df_new = self.fetch_today_articles()
        
        if df_new.empty:
            logger.info("No articles found for today.")
            return pd.DataFrame()
        
        df_focus = df_new[df_new['has_exchange'] == 1]
        logger.info(f"Found {len(df_focus)} articles with exchange info.")
        
        if limit:
            df_focus = df_focus.head(limit)
        
        return await self.async_process_articles_batch(df_focus)
    
    def process_today(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Synchronous wrapper for processing today's articles."""
        try:
            loop = asyncio.get_running_loop()
            logger.warning("Already in event loop. Use async_process_today() instead.")
            return pd.DataFrame()
        except RuntimeError:
            return asyncio.run(self.async_process_today(limit))


# Convenient functions
async def run_daily_collection(openai_api_key: str, fmp_api_key: str = None, limit: Optional[int] = None):
    """Main daily runner."""
    predictor = EnhancedPRNewsStockPredictor(
        openai_api_key,
        fmp_api_key=fmp_api_key,
        max_concurrent_predictions=5
    )
    
    logger.info("=== Starting Daily News Collection ===")
    
    predictions = await predictor.async_process_today(limit)
    summary = predictor.get_today_summary()
    
    logger.info("\n=== Today's Summary ===")
    logger.info(f"Total articles: {summary.get('total_articles', 0)}")
    logger.info(f"Total predictions: {summary.get('total_predictions', 0)}")
    logger.info(f"Average confidence: {summary.get('avg_confidence', 'N/A')}")
    
    if summary.get('industries_covered'):
        logger.info("\nIndustries covered:")
        for ind in summary['industries_covered']:
            logger.info(f"  - {ind['industry']}: {ind['article_count']} articles")
    
    filename = predictor.export_today_to_csv()
    logger.info(f"\n=== Daily collection complete. Exported to {filename} ===")
    
    return predictor, predictions, summary


def daily_main(openai_api_key: str, fmp_api_key: str = None):
    """Simple daily runner - synchronous version."""
    return asyncio.run(run_daily_collection(openai_api_key, fmp_api_key))


def run_correlation_analysis(openai_api_key: str, fmp_api_key: str, 
                             growth_threshold: float = 10.0, top_n: int = 10):
    """
    Run correlation analysis on stocks with high predicted growth.
    This is the main function to call for deep dive analysis.
    """
    predictor = EnhancedPRNewsStockPredictor(
        openai_api_key,
        fmp_api_key=fmp_api_key
    )
    
    return predictor.run_deep_dive_analysis(growth_threshold, top_n)


# Example usage
if __name__ == "__main__":
    # Use environment-loaded keys
    print("="*80)
    print("STEP 1: DAILY NEWS COLLECTION")
    print("="*80)
    predictor, predictions, summary = daily_main(OPENAI_API_KEY, FMP_API_KEY)

    # ===== DEEP DIVE ANALYSIS =====
    print("\n" + "="*80)
    print("STEP 2: DEEP DIVE CORRELATION ANALYSIS")
    print("="*80)
    analysis_results = run_correlation_analysis(
        OPENAI_API_KEY,
        FMP_API_KEY,
        growth_threshold=10.0,
        top_n=10
    )

    # ===== QUERY EXAMPLES =====
    print("\n" + "="*80)
    print("STEP 3: QUERY EXAMPLES")
    print("="*80)

    high_growth = predictor.db.get_high_growth_predictions(threshold=10.0)
    print(f"\nStocks with >10% predicted growth: {len(high_growth)}")
    if not high_growth.empty:
        print(high_growth[['ticker', 'company', 'avg_predicted_growth', 'prediction_count']].head())

    correlation_history = predictor.db.get_correlation_history()
    print(f"\nPrevious correlation analyses: {len(correlation_history)}")

    print("\n=== Today's Industry Breakdown ===")
    industry_summary = predictor.db.get_industry_summary(datetime.now().strftime('%Y-%m-%d'))
    if not industry_summary.empty:
        print(industry_summary)

    print("\n=== Example: Query Specific Ticker ===")
    sample_ticker = "AAPL"
    ticker_predictions = predictor.db.get_ticker_predictions_history(sample_ticker, days=30)
    print(f"Recent predictions for {sample_ticker}: {len(ticker_predictions)} articles")
    if not ticker_predictions.empty:
        print(ticker_predictions[['date', 'sentiment', 'long_run_range_percent', 'confidence_score']].head())

    predictor.db.close()

    print("\n" + "="*80)
    print("COMPLETE! All data saved to database and CSV files.")
    print("="*80)