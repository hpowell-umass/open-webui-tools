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
