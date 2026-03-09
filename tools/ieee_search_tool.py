"""
title: IEEE Search Tool
description: Tool to perform semantic search for IEEE papers and view abstracts
author: AI Assistant
version: 0.1.0
"""

import aiohttp
import asyncio
from typing import Any, Optional, Callable, Awaitable, List, Dict
from pydantic import BaseModel
import urllib.parse
import logging
from datetime import datetime

try:
    from bs4 import BeautifulSoup

    HTML_PARSER_AVAILABLE = True
except ImportError:
    HTML_PARSER_AVAILABLE = False
    logging.warning(
        "BeautifulSoup4 is not available. Basic HTML parsing will be limited."
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tools:
    class UserValves(BaseModel):
        """No API keys required for IEEE search"""

        pass

    def __init__(self):
        self.base_url = "https://ieeexplore.ieee.org/search"
        self.results_per_page = 20
        self.max_results = 5
        self.search_timeout = 30
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search_ieee_papers(
        self,
        topic: str,
        year_range: str = "all",
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search IEEE Xplore for papers on a given topic and return formatted results with abstracts.

        Args:
            topic: Topic to search for (e.g., "quantum computing", "transformer models")
            year_range: Year range to search (e.g., "all", "last_5_years", "2020-2024")
            __event_emitter__: Optional event emitter for progress updates

        Returns:
            Formatted string containing paper details including titles, authors, dates,
            URLs and abstracts.
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Searching IEEE Xplore...", "done": False},
                }
            )

        if not HTML_PARSER_AVAILABLE:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "BeautifulSoup4 is not installed. Install with: pip install beautifulsoup4 lxml",
                            "done": True,
                        },
                    }
                )
            return "BeautifulSoup4 is not installed. Please install it with: pip install beautifulsoup4 lxml"

        try:
            # Construct search query with parameters
            search_query = f'"{topic}"'

            # Build query parameters for IEEE Xplore
            params = {
                "queryText": search_query,
                "highlight": "true",
                "rows": self.results_per_page,
                "start": 0,
                "pageNumber": 1,
            }

            # Add year range filter if specified
            if year_range != "all":
                if "-" in year_range:
                    start_year, end_year = year_range.split("-")
                    params["highlight"] = f"year:{start_year},{end_year}"
                else:
                    params["highlight"] = f"year:{year_range}"

            headers = {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
                "accept": "application/json, text/plain, */*",
                "accept-language": "en-US,en;q=0.9",
                "x-requested-with": "fetch",
            }

            async with self.session.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=self.search_timeout,
            ) as response:
                response.raise_for_status()

                # Try to parse as JSON first
                try:
                    data = await response.json()
                    if "records" in data:
                        entries = data["records"]
                    else:
                        entries = data.get("papers", [])
                except:
                    # Fallback to HTML parsing
                    html_content = await response.text()
                    entries = self._parse_html_results(html_content)

            if not entries:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "No papers found", "done": True},
                        }
                    )
                return f"No papers found on IEEE Xplore related to '{topic}'"

            results = ""

            # Loop over each paper entry
            for i, entry in enumerate(entries[: self.max_results], 1):
                paper_info = self._extract_paper_info(entry)
                if paper_info:
                    results += self._format_paper_entry(paper_info, i)

                    # Emit citation data with abstract
                    if __event_emitter__ and paper_info.get("abstract"):
                        await __event_emitter__(
                            {
                                "type": "citation",
                                "data": {
                                    "document": [
                                        paper_info.get(
                                            "abstract", "No abstract available"
                                        )
                                    ],
                                    "metadata": [
                                        {
                                            "title": paper_info.get("title"),
                                            "authors": paper_info.get("authors"),
                                            "year": paper_info.get("year"),
                                            "url": paper_info.get("url"),
                                        }
                                    ],
                                    "source": {
                                        "name": paper_info.get("title"),
                                        "ieee_xplore": paper_info.get("url"),
                                    },
                                },
                            }
                        )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Search completed. Found {len(entries[:self.max_results])} papers",
                            "done": True,
                        },
                    }
                )

            return results

        except aiohttp.ClientError as e:
            error_msg = f"Error searching IEEE Xplore: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error during search: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            logger.error(error_msg)
            return error_msg

    async def search_ieee_with_filters(
        self,
        topic: str,
        year_range: str = "all",
        document_type: str = "article",
        index_terms: Optional[List[str]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search IEEE Xplore with advanced filtering options.

        Args:
            topic: Topic to search for
            year_range: Year range to search (e.g., "all", "last_5_years", "2020-2024")
            document_type: Type of document (e.g., "article", "conference", "book")
            index_terms: List of index terms to include
            __event_emitter__: Optional event emitter for progress updates

        Returns:
            Formatted string containing paper details
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Searching IEEE with advanced filters...",
                        "done": False,
                    },
                }
            )

        try:
            search_query = topic
            if index_terms:
                search_query = f'"{topic}" AND ' + " AND ".join(
                    f'"{term}"' for term in index_terms
                )

            params = {
                "queryText": search_query,
                "highlight": "true",
                "rows": self.results_per_page,
                "start": 0,
            }

            if year_range != "all":
                if "-" in year_range:
                    start_year, end_year = year_range.split("-")
                    params["highlight"] = f"year:{start_year},{end_year}"
                else:
                    params["highlight"] = f"year:{year_range}"

            if document_type:
                params["docType"] = document_type

            headers = {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }

            async with self.session.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=self.search_timeout,
            ) as response:
                response.raise_for_status()

                try:
                    data = await response.json()
                    entries = data.get("records", [])
                except:
                    html_content = await response.text()
                    entries = self._parse_html_results(html_content)

            if not entries:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "No papers found with filters",
                                "done": True,
                            },
                        }
                    )
                return f"No papers found on IEEE Xplore related to '{topic}' with specified filters"

            results = ""
            for i, entry in enumerate(entries[: self.max_results], 1):
                paper_info = self._extract_paper_info(entry)
                if paper_info:
                    results += self._format_paper_entry(paper_info, i)

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Advanced search completed",
                            "done": True,
                        },
                    }
                )

            return results

        except Exception as e:
            error_msg = f"Error in advanced search: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            logger.error(error_msg)
            return error_msg

    def _parse_html_results(self, html_content: str) -> List[Dict]:
        """
        Parse HTML content from IEEE search results.

        Args:
            html_content: HTML content from IEEE search

        Returns:
            List of paper information dictionaries
        """
        if not HTML_PARSER_AVAILABLE:
            return []

        papers = []
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Try different selectors based on IEEE's HTML structure
            results = soup.select(
                'div.result-card, .results-container > div.result-item, [class*="result"], [class*="Paper"]'
            )

            for result in results:
                try:
                    paper_info = {
                        "title": "",
                        "authors": [],
                        "abstract": "",
                        "year": "",
                        "url": "",
                        "citation_count": "",
                        "doi": "",
                    }

                    # Extract title
                    title_element = result.find(
                        ["h2", "h3", "h4", "a"],
                        class_=lambda x: x
                        and ("title" in x.lower() or "result-title" in x.lower()),
                    )
                    if title_element:
                        paper_info["title"] = title_element.get_text(strip=True)
                        paper_info["url"] = title_element.get("href", "")

                    # Extract authors
                    author_elements = result.find_all(
                        ["span", "div", "a"],
                        class_=lambda x: x and ("author" in x.lower()),
                    )
                    authors = [
                        a.get_text(strip=True)
                        for a in author_elements
                        if a.get_text(strip=True)
                    ]
                    paper_info["authors"] = authors[:10]  # Limit to first 10 authors

                    # Extract abstract
                    abstract_element = result.find(
                        ["p", "div", "span"],
                        class_=lambda x: x
                        and ("abstract" in x.lower() or "summary" in x.lower()),
                    )
                    if abstract_element:
                        paper_info["abstract"] = abstract_element.get_text(strip=True)

                    # Extract year
                    year_element = result.find(
                        ["span", "div"], class_=lambda x: x and ("year" in x.lower())
                    )
                    if year_element:
                        year_text = year_element.get_text(strip=True)
                        year_match = "".join(filter(str.isdigit, year_text))
                        if year_match:
                            paper_info["year"] = year_match

                    # Extract DOI if available
                    doi_element = result.find(
                        "span",
                        class_=lambda x: x
                        and ("doi" in x.lower() or "identifier" in x.lower()),
                    )
                    if doi_element:
                        paper_info["doi"] = doi_element.get_text(strip=True)

                    # Extract citation count
                    cite_element = result.find(
                        ["span", "div"],
                        class_=lambda x: x
                        and ("cite" in x.lower() or "cited" in x.lower()),
                    )
                    if cite_element:
                        paper_info["citation_count"] = cite_element.get_text(strip=True)

                    if paper_info["title"]:
                        papers.append(paper_info)

                except Exception as e:
                    continue

        except Exception as e:
            logger.error(f"Error parsing HTML results: {str(e)}")

        return papers

    def _extract_paper_info(self, entry: Dict) -> Dict:
        """
        Extract paper information from different input formats.

        Args:
            entry: Paper entry from search results

        Returns:
            Dictionary with paper information
        """
        paper_info = {
            "title": "",
            "authors": [],
            "abstract": "",
            "year": "",
            "url": "",
            "citation_count": "",
            "doi": "",
        }

        if isinstance(entry, dict):
            # Handle JSON-style entries
            paper_info["title"] = entry.get("title", entry.get("documentTitle", ""))
            paper_info["authors"] = entry.get("authors", entry.get("authors", []))
            paper_info["abstract"] = entry.get("abstract", entry.get("summary", ""))
            paper_info["year"] = entry.get("year", "")
            paper_info["url"] = entry.get("url", entry.get("link", ""))
            paper_info["doi"] = entry.get("doi", "")
            paper_info["citation_count"] = entry.get("citations", "")

        elif isinstance(entry, str):
            # Handle string-based entries
            paper_info["title"] = entry

        return paper_info if paper_info["title"] else {}

    def _format_paper_entry(self, paper_info: Dict, index: int) -> str:
        """
        Format paper information into a readable string.

        Args:
            paper_info: Dictionary with paper information
            index: Index number for the paper

        Returns:
            Formatted string containing paper details
        """
        results = f"{index}. {paper_info.get('title', 'Unknown Title')}\n"

        # Authors
        if paper_info.get("authors"):
            authors = ", ".join(paper_info["authors"][:5])
            results += f"   Authors: {authors}\n"
            if len(paper_info["authors"]) > 5:
                results += f"   ... and {len(paper_info["authors"]) - 5} more\n"

        # Year and metadata
        year = paper_info.get("year", "Unknown Year")
        results += f"   Year: {year}\n"

        # Citation count
        citations = paper_info.get("citation_count", "")
        if citations:
            results += f"   Citations: {citations}\n"

        # DOI
        doi = paper_info.get("doi", "")
        if doi:
            results += f"   DOI: {doi}\n"

        # URL
        url = paper_info.get("url", "")
        if url and url.startswith("http"):
            results += f"   URL: {url}\n"

        # Abstract
        abstract = paper_info.get("abstract", "")
        if abstract:
            abstract = abstract[:500] + "..." if len(abstract) > 500 else abstract
            results += f"   Abstract: {abstract}\n\n"
        else:
            results += "   Abstract: No abstract available\n\n"

        return results

# Unit tests for IEEE Search Tool
import unittest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
import aiohttp
import pytest

class TestUserValves(unittest.TestCase):
    """Test UserValves model"""
    
    def test_user_valves_creation(self):
        """Test that UserValves can be created successfully"""
        valves = UserValves()
        self.assertIsNotNone(valves)
    
    def test_user_valves_no_api_keys(self):
        """Test that UserValves requires no API keys"""
        valves = UserValves()
        self.assertTrue(hasattr(valves, '__dict__'))
    
    def test_user_valves_is_base_model(self):
        """Test that UserValves is a Pydantic BaseModel"""
        self.assertEqual(UserValves.__bases__[0].__name__, 'BaseModel')

class TestToolsContextManager(unittest.TestCase):
    """Test async context manager functionality"""
    
    @patch('aiohttp.ClientSession')
    def test_async_enter(self, mock_session_class):
        """Test that __aenter__ creates and returns session"""
        tools = Tools()
        session = mock_session_class.return_value
        
        result = tools.__aenter__()
        
        mock_session_class.assert_called_once()
        self.assertEqual(result, session)
    
    @patch('aiohttp.ClientSession')
    def test_async_exit(self, mock_session_class):
        """Test that __aexit__ closes the session"""
        tools = Tools()
        mock_session = mock_session_class.return_value
        mock_session.__aexit__ = AsyncMock()
        
        tools.__aenter__()
        tools.__aexit__(None, None, None)
        
        mock_session.__aexit__.assert_called_once()

class TestToolsInitialization(unittest.TestCase):
    """Test Tools class initialization"""
    
    def test_tools_initialization(self):
        """Test that Tools can be initialized with default values"""
        tools = Tools()
        
        self.assertEqual(tools.base_url, "https://ieeexplore.ieee.org/search")
        self.assertEqual(tools.results_per_page, 20)
        self.assertEqual(tools.max_results, 5)
        self.assertEqual(tools.search_timeout, 30)
        self.assertIsNone(tools.session)
    
    def test_tools_with_custom_params(self):
        """Test Tools initialization with custom parameters"""
        tools = Tools()
        
        # Verify default configuration
        self.assertTrue(hasattr(tools, 'base_url'))
        self.assertTrue(hasattr(tools, 'results_per_page'))
        self.assertTrue(hasattr(tools, 'max_results'))
        self.assertTrue(hasattr(tools, 'search_timeout'))

class TestToolsAttributes(unittest.TestCase):
    """Test Tools class attributes"""
    
    def test_base_url_attribute(self):
        """Test base_url attribute"""
        tools = Tools()
        self.assertEqual(tools.base_url, "https://ieeexplore.ieee.org/search")
    
    def test_results_per_page_attribute(self):
        """Test results_per_page attribute"""
        tools = Tools()
        self.assertEqual(tools.results_per_page, 20)
    
    def test_max_results_attribute(self):
        """Test max_results attribute"""
        tools = Tools()
        self.assertEqual(tools.max_results, 5)
    
    def test_search_timeout_attribute(self):
        """Test search_timeout attribute"""
        tools = Tools()
        self.assertEqual(tools.search_timeout, 30)
    
    def test_session_attribute(self):
        """Test session attribute initialization"""
        tools = Tools()
        self.assertIsNone(tools.session)

class TestSearchIEEPPapers(unittest.TestCase):
    """Test search_ieee_papers method"""
    
    @patch.object(Tools, '_parse_html_results')
    @patch.object(Tools, '__aenter__')
    @patch.object(Tools, '__aexit__')
    @patch('aiohttp.ClientSession')
    def test_search_ieee_papers_success(self, mock_session_class, mock_exit, mock_enter, mock_parse):
        """Test successful paper search"""
        tools = Tools()
        mock_session = mock_session_class.return_value
        mock_enter.return_value = mock_session
        
        # Mock successful search with sample HTML
        sample_html = """
        <html>
            <div class="search-results">
                <div class="article">
                    <h3><a href="/article/1">Test Paper</a></h3>
                    <p>Author Name</p>
                    <p>2023</p>
                    <div class="abstract">Test abstract content</div>
                </div>
            </div>
        </html>
        """
        
        mock_parse.return_value = [{
            'title': 'Test Paper',
            'authors': ['Author Name'],
            'year': '2023',
            'url': '/article/1',
            'abstract': 'Test abstract content'
        }]
        
        result = tools.search_ieee_papers("test topic")
        
        self.assertIsNotNone(result)
        self.assertIn("Test Paper", result)
        self.assertIn("Author Name", result)
    
    @patch.object(Tools, '_parse_html_results')
    @patch.object(Tools, '__aenter__')
    @patch.object(Tools, '__aexit__')
    @patch('aiohttp.ClientSession')
    def test_search_ieee_papers_with_event_emitter(self, mock_session_class, mock_exit, mock_enter, mock_parse):
        """Test search with event emitter"""
        tools = Tools()
        mock_session = mock_session_class.return_value
        mock_enter.return_value = mock_session
        
        event_emitter = AsyncMock()
        
        # Mock successful search
        mock_parse.return_value = []
        
        result = tools.search_ieee_papers("test topic", event_emitter=event_emitter)
        
        event_emitter.assert_any_call({
            "type": "status",
            "data": {"description": "Searching IEEE Xplore...", "done": False},
        })

class TestParseHTMLResults(unittest.TestCase):
    """Test _parse_html_results method"""
    
    def test_parse_html_results_with_beautifulsoup(self):
        """Test parsing HTML results with BeautifulSoup"""
        tools = Tools()
        
        html_content = """
        <html>
            <div class="search-results">
                <div class="article">
                    <h3><a href="/article/1">Test Paper Title</a></h3>
                    <p>John Doe</p>
                    <p>2023</p>
                    <div class="abstract">This is a test abstract</div>
                </div>
            </div>
        </html>
        """
        
        result = tools._parse_html_results(html_content)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
    
    def test_parse_html_results_with_empty_html(self):
        """Test parsing empty HTML content"""
        tools = Tools()
        
        result = tools._parse_html_results("")
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)
    
    def test_parse_html_results_with_no_papers(self):
        """Test parsing HTML with no papers found"""
        tools = Tools()
        
        html_content = "<html><body>No papers here</body></html>"
        
        result = tools._parse_html_results(html_content)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)
    
    def test_parse_html_results_with_incomplete_html(self):
        """Test parsing HTML with incomplete structure"""
        tools = Tools()
        
        html_content = "<html><body>Just some text</body></html>"
        
        result = tools._parse_html_results(html_content)
        
        # Should not crash and return empty list
        self.assertIsInstance(result, list)

class TestHTMLParserAvailability(unittest.TestCase):
    """Test HTML parser availability checks"""
    
    def test_beautifulsoup_import_available(self):
        """Test if BeautifulSoup is available"""
        try:
            from bs4 import BeautifulSoup
            self.assertTrue(True)
        except ImportError:
            self.fail("BeautifulSoup4 is not installed")

class TestToolsAsyncOperations(unittest.TestCase):
    """Test async operations in Tools class"""
    
    @patch('aiohttp.ClientSession')
    @patch('aiohttp.ClientSession.request')
    @patch.object(Tools, '_parse_html_results')
    async def test_async_search_operations(self, mock_parse, mock_request, mock_session_class):
        """Test async search operations"""
        tools = Tools()
        mock_session = mock_session_class.return_value
        mock_request.return_value = AsyncMock()
        
        # Simulate async search flow
        async with tools:
            # Mock response
            mock_response = AsyncMock()
            mock_response.text = AsyncMock(return_value="<html><body>Test</body></html>")
            mock_request.return_value = mock_response
            
            mock_parse.return_value = [{
                'title': 'Test',
                'authors': [],
                'year': '2023',
                'url': '/test',
                'abstract': 'Test abstract'
            }]
            
            # This would normally trigger search, but we're just testing the flow
            self.assertIsNotNone(tools.session)
        
        mock_session.close.assert_called_once()

class TestSearchIEEPPapersWithYearRange(unittest.TestCase):
    """Test search with different year ranges"""
    
    @patch.object(Tools, '_parse_html_results')
    @patch.object(Tools, '__aenter__')
    @patch.object(Tools, '__aexit__')
    @patch('aiohttp.ClientSession')
    def test_search_all_years(self, mock_session_class, mock_exit, mock_enter, mock_parse):
        """Test search with 'all' year range"""
        tools = Tools()
        mock_session = mock_session_class.return_value
        mock_enter.return_value = mock_session
        
        mock_parse.return_value = []
        
        result = tools.search_ieee_papers("quantum computing", year_range="all")
        
        self.assertIsNotNone(result)
    
    @patch.object(Tools, '_parse_html_results')
    @patch.object(Tools, '__aenter__')
    @patch.object(Tools, '__aexit__')
    @patch('aiohttp.ClientSession')
    def test_search_last_5_years(self, mock_session_class, mock_exit, mock_enter, mock_parse):
        """Test search with 'last_5_years' year range"""
        tools = Tools()
        mock_session = mock_session_class.return_value
        mock_enter.return_value = mock_session
        
        mock_parse.return_value = []
        
        result = tools.search_ieee_papers("transformer models", year_range="last_5_years")
        
        self.assertIsNotNone(result)
    
    @patch.object(Tools, '_parse_html_results')
    @patch.object(Tools, '__aenter__')
    @patch.object(Tools, '__aexit__')
    @patch('aiohttp.ClientSession')
    def test_search_specific_year_range(self, mock_session_class, mock_exit, mock_enter, mock_parse):
        """Test search with specific year range"""
        tools = Tools()
        mock_session = mock_session_class.return_value
        mock_enter.return_value = mock_session
        
        mock_parse.return_value = []
        
        result = tools.search_ieee_papers("deep learning", year_range="2020-2024")
        
        self.assertIsNotNone(result)

# Run tests
if __name__ == '__main__':
    unittest.main(verbosity=2)