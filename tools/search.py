# tools/search.py
"""
Web search tools integration for the Research Brief Generator.
Uses SERP API for web search functionality.
"""

import os
import requests
from typing import List, Dict, Any, Optional, Type
import logging
from dataclasses import dataclass
from enum import Enum
import time

# Load environment variables at module level
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    # dotenv not installed, that's okay
    pass

# LangChain imports
from langchain_core.tools import BaseTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SearchProvider(str, Enum):
    """Supported search providers."""
    SERPER = "serper"


@dataclass
class SearchResult:
    """Structured search result."""
    title: str
    url: str
    snippet: str
    source: str
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "rank": self.rank
        }


class WebSearchManager:
    """Manages SERP API search functionality."""
    
    def __init__(self):
        self.serper_api = None
        self._initialize_serper()
    
    def _initialize_serper(self):
        """Initialize SERP API provider."""
        api_key = os.getenv("SERPER_API_KEY")  # Fixed typo here
        
        if not api_key:
            error_msg = (
                "SERPER_API_KEY environment variable is required. "
                "Please set it in your environment or .env file. "
                "Get your API key from https://serper.dev/"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate API key format (should be a reasonable length)
        if len(api_key.strip()) < 10:
            error_msg = "SERPER_API_KEY appears to be invalid (too short)"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Initialize with explicit API key
            self.serper_api = GoogleSerperAPIWrapper(serper_api_key=api_key.strip())
            logger.info("Successfully initialized Serper search provider")
            
            # Test the API with a simple query
            self._test_api_connection()
            
        except Exception as e:
            error_msg = f"Failed to initialize Serper API: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _test_api_connection(self):
        """Test API connection with a simple query."""
        try:
            test_result = self.serper_api.results("test")
            logger.info("API connection test successful")
        except Exception as e:
            logger.warning(f"API connection test failed: {e}")
            # Don't raise here, let it fail on actual usage with better error handling
    
    def search(self, 
               query: str, 
               max_results: int = 5) -> List[SearchResult]:
        """
        Search for information using SERP API.
        """
        if not self.serper_api:
            logger.error("SERP API not initialized")
            raise ValueError("SERP API not initialized")
        
        try:
            logger.info(f"Searching with SERP API: '{query}'")
            return self._search_serper(query, max_results)
        except Exception as e:
            error_msg = f"Search failed with SERP API: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _search_serper(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Google Serper API."""
        try:
            # Serper returns structured results
            raw_results = self.serper_api.results(query)
            results = []
            
            if not raw_results:
                logger.warning(f"No results returned for query: {query}")
                return results
            
            organic_results = raw_results.get('organic', [])
            
            if not organic_results:
                logger.warning(f"No organic results found for query: {query}")
                return results
            
            for i, result in enumerate(organic_results[:max_results]):
                search_result = SearchResult(
                    title=result.get('title', 'No Title'),
                    url=result.get('link', ''),
                    snippet=result.get('snippet', 'No description available'),
                    source=SearchProvider.SERPER.value,
                    rank=i + 1
                )
                results.append(search_result)
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                error_msg = (
                    "SERP API returned 403 Forbidden. Possible causes:\n"
                    "1. Invalid API key - check your SERPER_API_KEY\n"
                    "2. API quota exceeded - check your usage at https://serper.dev/\n"
                    "3. Account not activated - verify your Serper account\n"
                    f"Response: {e.response.text if hasattr(e, 'response') else 'No response details'}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            elif e.response.status_code == 429:
                error_msg = "SERP API rate limit exceeded. Please wait before making more requests."
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                error_msg = f"SERP API HTTP error {e.response.status_code}: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Error in _search_serper: {e}")
            raise


class ContentFetcher:
    """Fetches and processes content from URLs."""
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def fetch_content(self, url: str) -> Optional[str]:
        """
        Fetch content from a URL with error handling and retries.
        """
        if not url or not url.startswith(('http://', 'https://')):
            logger.warning(f"Invalid URL: {url}")
            return None
            
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Fetching content from {url} (attempt {attempt + 1})")
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # Try to extract text content
                content = self._extract_text_content(response.text, url)
                return content
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to fetch {url} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        logger.error(f"Failed to fetch content from {url} after {self.max_retries} attempts")
        return None
    
    def _extract_text_content(self, html: str, url: str) -> str:
        """Extract text content from HTML."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content length
            max_length = 10000  # Adjust as needed
            if len(text) > max_length:
                text = text[:max_length] + "... [Content truncated]"
            
            return text
            
        except ImportError:
            logger.warning("BeautifulSoup not available, returning raw HTML snippet")
            # Fallback: return first 1000 characters of HTML
            return html[:1000] + "... [Install beautifulsoup4 for better content extraction]"
        except Exception as e:
            logger.error(f"Failed to extract text from {url}: {e}")
            return f"Failed to extract content from {url}"


# Input schema for the search tool (Pydantic v2 compatible)
class ResearchSearchInput(BaseModel):
    """Input schema for research search tool."""
    query: str = Field(description="The search query to research")
    max_results: int = Field(default=5, description="Maximum number of results to return")


# LangChain Tool Wrapper (Pydantic v2 compatible)
class ResearchSearchTool(BaseTool):
    """LangChain tool wrapper for research search."""
    
    name: str = Field(default="research_search", description="Name of the tool")
    description: str = Field(
        default="Search for information on the web using SERP API. Input should be a search query string.",
        description="Description of what the tool does"
    )
    args_schema: Type[BaseModel] = Field(default=ResearchSearchInput, description="Schema for tool arguments")
    search_manager: Optional[WebSearchManager] = Field(default=None, description="Search manager instance")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, 'search_manager') or self.search_manager is None:
            try:
                self.search_manager = WebSearchManager()
            except Exception as e:
                logger.error(f"Failed to initialize search manager in tool: {e}")
                raise
    
    def _run(self, query: str, max_results: int = 5) -> str:
        """Execute the search and return formatted results."""
        try:
            if not self.search_manager:
                raise ValueError("Search manager not initialized")
                
            results = self.search_manager.search(query, max_results)
            
            if not results:
                return f"No results found for query: {query}"
            
            formatted_results = []
            for result in results:
                formatted_results.append(
                    f"Title: {result.title}\n"
                    f"URL: {result.url}\n"
                    f"Snippet: {result.snippet}\n"
                    f"---"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            error_msg = f"Search tool execution failed: {e}"
            logger.error(error_msg)
            return error_msg
    
    async def _arun(self, query: str, max_results: int = 5) -> str:
        """Async version of the search."""
        return self._run(query, max_results)


def create_research_search_tool() -> ResearchSearchTool:
    """Create a research search tool instance."""
    try:
        return ResearchSearchTool()
    except Exception as e:
        logger.error(f"Failed to create research search tool: {e}")
        raise


# Global instances
_search_manager = None
_content_fetcher = None
_search_tool = None

def get_search_manager() -> WebSearchManager:
    """Get the global search manager instance."""
    global _search_manager
    if _search_manager is None:
        try:
            _search_manager = WebSearchManager()
        except Exception as e:
            logger.error(f"Failed to create global search manager: {e}")
            raise
    return _search_manager

def get_content_fetcher() -> ContentFetcher:
    """Get the global content fetcher instance."""
    global _content_fetcher
    if _content_fetcher is None:
        _content_fetcher = ContentFetcher()
    return _content_fetcher

def get_search_tool():
    """Get the global search tool instance."""
    global _search_tool
    if _search_tool is None:
        try:
            _search_tool = create_research_search_tool()
        except Exception as e:
            logger.error(f"Failed to create global search tool: {e}")
            raise
    return _search_tool

def search_and_fetch(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search for content and fetch full text from results.
    Returns a list of dictionaries with search results and content.
    """
    try:
        search_manager = get_search_manager()
        content_fetcher = get_content_fetcher()
        
        # Get search results
        search_results = search_manager.search(query, max_results)
        
        if not search_results:
            logger.warning(f"No search results found for query: {query}")
            return []
        
        # Fetch content for each result
        enriched_results = []
        for result in search_results:
            content = content_fetcher.fetch_content(result.url)
            
            enriched_result = result.to_dict()
            enriched_result['content'] = content
            enriched_result['content_length'] = len(content) if content else 0
            
            enriched_results.append(enriched_result)
        
        return enriched_results
        
    except Exception as e:
        logger.error(f"Error in search_and_fetch: {e}")
        return []


def search_for_topic(topic: str, depth: int) -> List[Dict[str, Any]]:
    """Search for information about a topic with depth-appropriate number of results."""
    try:
        # Adjust number of results based on depth
        max_results = min(2 + depth, 8)  # 3-8 results depending on depth
        
        # Create search queries
        queries = [
            topic,
            f"{topic} research analysis",
            f"{topic} latest developments"
        ]
        
        # Add more specific queries for higher depth
        if depth >= 4:
            queries.append(f"{topic} comprehensive review")
            queries.append(f"{topic} future trends")
        
        all_results = []
        seen_urls = set()
        
        for query in queries:
            try:
                results = search_and_fetch(query, max_results // len(queries) + 1)
                
                # Deduplicate by URL
                for result in results:
                    if result['url'] not in seen_urls:
                        seen_urls.add(result['url'])
                        all_results.append(result)
                        
                        if len(all_results) >= max_results:
                            break
                
                if len(all_results) >= max_results:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to search for query '{query}': {e}")
                continue
        
        logger.info(f"Found {len(all_results)} total results for topic: {topic}")
        return all_results[:max_results]
        
    except Exception as e:
        logger.error(f"Error in search_for_topic: {e}")
        return []


# Test function to verify API key and connection
def test_serper_connection():
    """Test the SERP API connection."""
    try:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return False, "SERPER_API_KEY environment variable not found"
        
        # Print API key info (first 8 and last 4 characters for security)
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "Key too short"
        print(f"Using API key: {masked_key}")
        
        search_manager = WebSearchManager()
        results = search_manager.search("test query", max_results=1)
        
        if results:
            return True, f"Connection successful. Found {len(results)} results."
        else:
            return False, "Connection established but no results returned"
            
    except Exception as e:
        return False, f"Connection failed: {e}"


def debug_serper_api():
    """Debug function to check SERP API status."""
    import requests
    
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        print("❌ SERPER_API_KEY not found in environment variables")
        return
    
    print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else 'too_short'}")
    
    # Test direct API call
    url = "https://google.serper.dev/search"
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    data = {
        'q': 'test query',
        'num': 1
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        print(f"Direct API Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Direct API call successful")
            result = response.json()
            print(f"Results count: {len(result.get('organic', []))}")
        elif response.status_code == 403:
            print("❌ 403 Forbidden - Check your API key and account status")
            print("   • Visit https://serper.dev/ to verify your account")
            print("   • Check if your account is active and has credits")
            print(f"   • Response: {response.text}")
        elif response.status_code == 429:
            print("❌ 429 Rate Limited - Too many requests")
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")


# Create a mock search function as fallback
def create_mock_search_results(query: str, max_results: int) -> List[SearchResult]:
    """Create mock search results for testing when API is unavailable."""
    mock_results = []
    
    # Generate some realistic mock results
    topics = {
        'AI': ['artificial intelligence applications', 'machine learning advances', 'AI in business'],
        'Healthcare': ['medical technology', 'digital health solutions', 'telemedicine trends'],
        'Technology': ['tech innovations', 'software development', 'digital transformation'],
        'default': ['research findings', 'latest developments', 'comprehensive analysis']
    }
    
    # Find relevant topic or use default
    topic_key = 'default'
    for key in topics.keys():
        if key.lower() in query.lower():
            topic_key = key
            break
    
    relevant_topics = topics[topic_key]
    
    for i in range(min(max_results, 3)):
        topic = relevant_topics[i % len(relevant_topics)]
        mock_result = SearchResult(
            title=f"{topic.title()}: {query}",
            url=f"https://example.com/article-{i+1}",
            snippet=f"This article discusses {topic} related to {query}. "
                   f"It provides comprehensive insights and analysis on the topic, "
                   f"covering recent developments and future implications.",
            source="mock",
            rank=i + 1
        )
        mock_results.append(mock_result)
    
    return mock_results


# Usage example and debugging
if __name__ == "__main__":
    print("=== SERP API Debug Tool ===\n")
    
    # Debug API setup
    debug_serper_api()
    print("\n" + "="*50 + "\n")
    
    # Test the connection
    success, message = test_serper_connection()
    print(f"SERP API Test: {'SUCCESS' if success else 'FAILED'}")
    print(f"Message: {message}")
    
    if success:
        # Test search functionality
        try:
            print("\n=== Testing Search Functionality ===")
            results = search_for_topic("artificial intelligence", 3)
            print(f"\nFound {len(results)} results for 'artificial intelligence'")
            for i, result in enumerate(results[:2], 1):
                print(f"\nResult {i}:")
                print(f"Title: {result['title']}")
                print(f"URL: {result['url']}")
                print(f"Snippet: {result['snippet'][:100]}...")
        except Exception as e:
            print(f"Search test failed: {e}")
    else:
        print("\n=== API Issues Detected ===")
        print("Common solutions:")
        print("1. Check your API key at https://serper.dev/")
        print("2. Verify your account is active and has credits")
        print("3. Check if you've exceeded your quota")
        print("4. Try regenerating your API key")
        
        print("\n=== Testing with Mock Data ===")
        mock_results = create_mock_search_results("test query", 3)
        print(f"Created {len(mock_results)} mock results:")
        for i, result in enumerate(mock_results, 1):
            print(f"{i}. {result.title}")
            print(f"   {result.snippet[:80]}...")
            print()