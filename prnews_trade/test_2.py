import requests
from bs4 import BeautifulSoup
import feedparser
import time
from datetime import datetime
import re
from typing import List, Dict
import json

class PublicCompanyNewsScraper:
    """
    Scraper for Business Wire, GlobeNewswire, and AccessWire
    Focused on public company press releases
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    # ============ BUSINESS WIRE ============
    
    def scrape_businesswire(self, company_name: str = None, ticker: str = None, limit: int = 20) -> List[Dict]:
        """
        Scrape Business Wire for company news
        """
        print(f"Scraping Business Wire for: {company_name or ticker}")
        results = []
        
        try:
            # Business Wire search URL
            if ticker:
                search_query = ticker
            elif company_name:
                search_query = company_name
            else:
                search_query = ""
            
            # Try multiple approaches
            
            # Approach 1: Direct search
            url = f"https://www.businesswire.com/portal/site/home/search/?searchType=news&searchTerm={search_query}&searchPage=1"
            
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news items (Business Wire structure)
            news_items = soup.find_all('div', class_='bwNewsList')
            
            if not news_items:
                # Try alternative structure
                news_items = soup.find_all('div', class_='bwRelease')
            
            for item in news_items[:limit]:
                try:
                    title_elem = item.find('a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '')
                    
                    if link and not link.startswith('http'):
                        link = 'https://www.businesswire.com' + link
                    
                    date_elem = item.find('time')
                    date = date_elem.get_text(strip=True) if date_elem else 'N/A'
                    
                    # Extract summary if available
                    summary_elem = item.find('div', class_='bwAbstract')
                    summary = summary_elem.get_text(strip=True) if summary_elem else ''
                    
                    results.append({
                        'source': 'Business Wire',
                        'title': title,
                        'link': link,
                        'date': date,
                        'summary': summary,
                        'company': company_name or ticker
                    })
                except Exception as e:
                    print(f"Error parsing Business Wire item: {e}")
                    continue
            
            # Approach 2: Try RSS feed if direct scraping didn't work
            if len(results) == 0:
                print("Trying Business Wire RSS feed...")
                rss_url = "https://www.businesswire.com/portal/site/home/news/rss"
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:limit]:
                    # Filter by company name/ticker if provided
                    if search_query and search_query.lower() not in entry.title.lower():
                        continue
                    
                    results.append({
                        'source': 'Business Wire',
                        'title': entry.title,
                        'link': entry.link,
                        'date': entry.get('published', 'N/A'),
                        'summary': entry.get('summary', ''),
                        'company': company_name or ticker
                    })
        
        except Exception as e:
            print(f"Error scraping Business Wire: {e}")
        
        time.sleep(1)  # Be polite
        return results
    
    # ============ GLOBENEWSWIRE ============
    
    def scrape_globenewswire(self, company_name: str = None, ticker: str = None, limit: int = 20) -> List[Dict]:
        """
        Scrape GlobeNewswire for company news
        """
        print(f"Scraping GlobeNewswire for: {company_name or ticker}")
        results = []
        
        try:
            search_query = ticker if ticker else company_name
            
            # GlobeNewswire search URL
            url = f"https://www.globenewswire.com/search/keyword/{search_query}"
            
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news items
            news_items = soup.find_all('div', class_='results-group')
            
            if not news_items:
                # Try alternative structure
                news_items = soup.find_all('article')
            
            for item in news_items[:limit]:
                try:
                    title_elem = item.find('a', class_='link')
                    if not title_elem:
                        title_elem = item.find('h3')
                        if title_elem:
                            title_elem = title_elem.find('a')
                    
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '')
                    
                    if link and not link.startswith('http'):
                        link = 'https://www.globenewswire.com' + link
                    
                    # Extract date
                    date_elem = item.find('time')
                    if not date_elem:
                        date_elem = item.find('span', class_='date')
                    date = date_elem.get_text(strip=True) if date_elem else 'N/A'
                    
                    # Extract summary
                    summary_elem = item.find('p')
                    summary = summary_elem.get_text(strip=True) if summary_elem else ''
                    
                    results.append({
                        'source': 'GlobeNewswire',
                        'title': title,
                        'link': link,
                        'date': date,
                        'summary': summary,
                        'company': company_name or ticker
                    })
                except Exception as e:
                    print(f"Error parsing GlobeNewswire item: {e}")
                    continue
            
            # Try RSS feed approach if no results
            if len(results) == 0:
                print("Trying GlobeNewswire RSS feed...")
                # Try company-specific RSS
                rss_url = f"https://www.globenewswire.com/RssFeed/keyword/{search_query}"
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:limit]:
                    results.append({
                        'source': 'GlobeNewswire',
                        'title': entry.title,
                        'link': entry.link,
                        'date': entry.get('published', 'N/A'),
                        'summary': entry.get('summary', ''),
                        'company': company_name or ticker
                    })
        
        except Exception as e:
            print(f"Error scraping GlobeNewswire: {e}")
        
        time.sleep(1)  # Be polite
        return results
    
    # ============ ACCESSWIRE ============
    
    def scrape_accesswire(self, company_name: str = None, ticker: str = None, limit: int = 20) -> List[Dict]:
        """
        Scrape AccessWire for company news
        """
        print(f"Scraping AccessWire for: {company_name or ticker}")
        results = []
        
        try:
            search_query = ticker if ticker else company_name
            
            # AccessWire search URL
            url = f"https://www.accesswire.com/search?q={search_query}"
            
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news items
            news_items = soup.find_all('div', class_='article-item')
            
            if not news_items:
                # Try alternative structure
                news_items = soup.find_all('article')
            
            if not news_items:
                # Try finding by class pattern
                news_items = soup.find_all('div', class_=re.compile('card|item'))
            
            for item in news_items[:limit]:
                try:
                    title_elem = item.find('h2') or item.find('h3') or item.find('a', class_='title')
                    
                    if title_elem:
                        if title_elem.name == 'a':
                            title = title_elem.get_text(strip=True)
                            link = title_elem.get('href', '')
                        else:
                            link_elem = title_elem.find('a')
                            if link_elem:
                                title = link_elem.get_text(strip=True)
                                link = link_elem.get('href', '')
                            else:
                                continue
                    else:
                        continue
                    
                    if link and not link.startswith('http'):
                        link = 'https://www.accesswire.com' + link
                    
                    # Extract date
                    date_elem = item.find('time') or item.find('span', class_='date')
                    date = date_elem.get_text(strip=True) if date_elem else 'N/A'
                    
                    # Extract summary
                    summary_elem = item.find('p', class_='summary') or item.find('div', class_='excerpt')
                    summary = summary_elem.get_text(strip=True) if summary_elem else ''
                    
                    results.append({
                        'source': 'AccessWire',
                        'title': title,
                        'link': link,
                        'date': date,
                        'summary': summary,
                        'company': company_name or ticker
                    })
                except Exception as e:
                    print(f"Error parsing AccessWire item: {e}")
                    continue
            
            # Try RSS feed if no results
            if len(results) == 0:
                print("Trying AccessWire RSS feed...")
                rss_url = "https://www.accesswire.com/rss/company"
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries:
                    # Filter by search query
                    if search_query and search_query.lower() not in entry.title.lower():
                        continue
                    
                    results.append({
                        'source': 'AccessWire',
                        'title': entry.title,
                        'link': entry.link,
                        'date': entry.get('published', 'N/A'),
                        'summary': entry.get('summary', ''),
                        'company': company_name or ticker
                    })
                    
                    if len(results) >= limit:
                        break
        
        except Exception as e:
            print(f"Error scraping AccessWire: {e}")
        
        time.sleep(1)  # Be polite
        return results
    
    # ============ MAIN SCRAPER ============
    
    def scrape_all_sources(self, company_name: str = None, ticker: str = None, limit: int = 20) -> Dict:
        """
        Scrape all three sources for a company
        """
        all_results = {
            'company': company_name or ticker,
            'scrape_date': datetime.now().isoformat(),
            'results': []
        }
        
        # Scrape each source
        bw_results = self.scrape_businesswire(company_name, ticker, limit)
        gn_results = self.scrape_globenewswire(company_name, ticker, limit)
        aw_results = self.scrape_accesswire(company_name, ticker, limit)
        
        # Combine results
        all_results['results'] = bw_results + gn_results + aw_results
        
        # Sort by date (newest first) - basic sorting
        all_results['results'].sort(key=lambda x: x['date'], reverse=True)
        
        return all_results
    
    def scrape_multiple_companies(self, companies: List[Dict], limit: int = 20) -> List[Dict]:
        """
        Scrape news for multiple companies
        companies: List of dicts with 'name' and/or 'ticker' keys
        Example: [{'name': 'Apple Inc', 'ticker': 'AAPL'}, {'name': 'Tesla', 'ticker': 'TSLA'}]
        """
        all_company_results = []
        
        for company in companies:
            print(f"\n{'='*60}")
            print(f"Processing: {company.get('name', company.get('ticker'))}")
            print(f"{'='*60}\n")
            
            results = self.scrape_all_sources(
                company_name=company.get('name'),
                ticker=company.get('ticker'),
                limit=limit
            )
            
            all_company_results.append(results)
            
            time.sleep(2)  # Be extra polite between companies
        
        return all_company_results
    
    def save_to_json(self, data, filename: str):
        """Save results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {filename}")
    
    def print_results(self, results: Dict):
        """Pretty print results"""
        print(f"\n{'='*80}")
        print(f"Company: {results['company']}")
        print(f"Total articles found: {len(results['results'])}")
        print(f"{'='*80}\n")
        
        for i, article in enumerate(results['results'], 1):
            print(f"{i}. [{article['source']}] {article['title']}")
            print(f"   Date: {article['date']}")
            print(f"   Link: {article['link']}")
            if article['summary']:
                print(f"   Summary: {article['summary'][:150]}...")
            print()


# ============ USAGE EXAMPLES ============

if __name__ == "__main__":
    
    scraper = PublicCompanyNewsScraper()
    
    # Example 1: Single company by ticker
    print("\n--- Example 1: Single Company (Apple) ---")
    results = scraper.scrape_all_sources(ticker="AAPL", limit=10)
    scraper.print_results(results)
    scraper.save_to_json(results, "apple_news.json")
    
    # Example 2: Single company by name
    print("\n--- Example 2: Single Company by Name (Tesla) ---")
    results = scraper.scrape_all_sources(company_name="Tesla", limit=10)
    scraper.print_results(results)
    
    # Example 3: Multiple companies
    print("\n--- Example 3: Multiple Companies ---")
    companies = [
        {'name': 'Apple Inc', 'ticker': 'AAPL'},
        {'name': 'Tesla Inc', 'ticker': 'TSLA'},
        {'name': 'Microsoft', 'ticker': 'MSFT'},
        {'name': 'NVIDIA', 'ticker': 'NVDA'}
    ]
    
    all_results = scraper.scrape_multiple_companies(companies, limit=5)
    scraper.save_to_json(all_results, "multiple_companies_news.json")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for company_data in all_results:
        print(f"{company_data['company']}: {len(company_data['results'])} articles found")
    
    # Example 4: Get only from specific source
    print("\n--- Example 4: Only GlobeNewswire ---")
    gn_results = scraper.scrape_globenewswire(ticker="AAPL", limit=5)
    for article in gn_results:
        print(f"- {article['title']}")
        print(f"  {article['link']}\n")