"""
title: Grokipedia Interface Tool
author: Grok (built by xAI)
version: 1.0
license: MIT
description: Interfaces with Grokipedia (https://grokipedia.com) using BeautifulSoup for web scraping. Provides search results (so the model can select specific pages), full page content retrieval, and internal link listing for navigation between connected Grokipedia pages.
requirements: requests beautifulsoup4
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from urllib.parse import quote
from pydantic import BaseModel, Field


class Tools:
    def __init__(self):
        """Initialize the Grokipedia tool."""
        self.citation = True
        self.base_url = "https://grokipedia.com"

    def grokipedia_search(self, query: str) -> str:
        """Search Grokipedia for a term/phrase and return structured search results.
        The model can then select one or more specific results (by title or URL) and call get_grokipedia_page to fetch the full page(s).

        :param query: The search term or phrase to look up on Grokipedia.
        :return: A formatted string listing the top results with title, full URL, and snippet. Returns an error message if the request fails.
        """
        encoded_query = quote(query)
        search_url = f"{self.base_url}/search?q={encoded_query}"
        try:
            response = requests.get(search_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            results: List[Dict[str, str]] = []
            # Parse results by finding links to /page/ (core Grokipedia article pattern)
            # Snippets are extracted from nearby text (robust fallback since exact classes may vary)
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.startswith("/page/"):
                    title = link.get_text(strip=True)
                    full_url = f"{self.base_url}{href}"
                    # Grab nearby text as snippet (parent or siblings)
                    snippet = ""
                    parent = link.find_parent(["h1", "h2", "h3", "div", "li", "p"])
                    if parent:
                        snippet = parent.get_text(strip=True)[:300]
                    # Deduplicate by URL
                    if title and full_url and not any(r["url"] == full_url for r in results):
                        results.append({"title": title, "url": full_url, "snippet": snippet or "No snippet available"})

            # Limit to top 10 for brevity
            results = results[:10]

            if not results:
                return f"No results found for '{query}' (or parsing failed)."

            output = f"Grokipedia Search Results for '{query}' ({len(results)} shown):\n\n"
            for i, res in enumerate(results, 1):
                output += f"{i}. **{res['title']}**\n   URL: {res['url']}\n   Snippet: {res['snippet']}\n\n"
            return output
        except requests.RequestException as e:
            return f"Error searching Grokipedia: {str(e)}"
        except Exception as e:
            return f"Unexpected error during search: {str(e)}"

    def get_grokipedia_page(self, page_title_or_url: str) -> str:
        """Fetch the full content of a specific Grokipedia page (after selecting from search results).

        :param page_title_or_url: Either the exact page title (e.g. 'Elon Musk') or the full page URL.
        :return: Cleaned main content of the page (title + body text). Returns an error message if the request fails.
        """
        if page_title_or_url.startswith("http"):
            url = page_title_or_url
        else:
            # Basic slugification (Grokipedia uses Title_With_Underscores)
            slug = page_title_or_url.replace(" ", "_").replace("/", "_").replace(":", "")
            url = f"{self.base_url}/page/{quote(slug)}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Clean unwanted elements
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()

            # Extract main content (try common containers first)
            main_content = (
                soup.find("main")
                or soup.find("article")
                or soup.find("div", class_=lambda x: x and ("content" in x.lower() or "body" in x.lower()))
                or soup.body
            )

            if main_content:
                # Convert to readable text with line breaks for sections
                text = main_content.get_text(separator="\n\n", strip=True)
                # Prepend title if available
                title_tag = soup.find("h1") or soup.find("title")
                page_title = title_tag.get_text(strip=True) if title_tag else url.split("/")[-1]
                return f"# {page_title}\n\n{text[:15000]}"  # Truncate very long pages
            else:
                return f"Page content could not be extracted from {url}."
        except requests.RequestException as e:
            return f"Error retrieving page {url}: {str(e)}"
        except Exception as e:
            return f"Unexpected error retrieving page: {str(e)}"

    def list_grokipedia_page_links(self, page_title_or_url: str) -> str:
        """List internal links to other Grokipedia pages within the given page.
        Useful for the model to discover and navigate to connected articles.

        :param page_title_or_url: Either the exact page title (e.g. 'Elon Musk') or the full page URL.
        :return: A formatted list of internal Grokipedia links (title → URL). Returns an error message if the request fails.
        """
        if page_title_or_url.startswith("http"):
            url = page_title_or_url
        else:
            slug = page_title_or_url.replace(" ", "_").replace("/", "_").replace(":", "")
            url = f"{self.base_url}/page/{quote(slug)}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            internal_links: List[Dict[str, str]] = []
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.startswith("/page/") and "/page/Main_Page" not in href:
                    title = link.get_text(strip=True)
                    if title and len(title) > 2:  # Filter out very short or noisy links
                        full_url = f"{self.base_url}{href}"
                        internal_links.append({"title": title, "url": full_url})

            # Deduplicate by URL
            unique_links = {l["url"]: l for l in internal_links}.values()

            if not unique_links:
                return f"No internal Grokipedia links found on {url}."

            output = f"Internal Grokipedia links found on {url} ({len(unique_links)} shown):\n\n"
            for i, link in enumerate(list(unique_links)[:25], 1):  # Reasonable limit
                output += f"{i}. {link['title']} → {link['url']}\n"
            return output
        except requests.RequestException as e:
            return f"Error listing links for {url}: {str(e)}"
        except Exception as e:
            return f"Unexpected error listing links: {str(e)}"


# =============================================================================
# UNIT TESTS (run with `python this_file.py`)
# =============================================================================
import unittest
from unittest.mock import patch, MagicMock


class TestGrokipediaTools(unittest.TestCase):
    def setUp(self):
        self.tools = Tools()

    @patch("requests.get")
    def test_grokipedia_search(self, mock_get):
        """Test search returns parsed results."""
        mock_html = """
        <html>
        <a href="/page/Elon_Musk">Elon Musk</a>
        <p>Elon Musk is an engineer and entrepreneur...</p>
        <a href="/page/SpaceX">SpaceX</a>
        <p>SpaceX is a space exploration company...</p>
        </html>
        """
        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.tools.grokipedia_search("Elon Musk")
        self.assertIn("Elon Musk", result)
        self.assertIn("/page/Elon_Musk", result)
        self.assertIn("SpaceX", result)
        self.assertIn("Grokipedia Search Results", result)

    @patch("requests.get")
    def test_get_grokipedia_page(self, mock_get):
        """Test page retrieval and content cleaning."""
        mock_html = """
        <html><head><title>Test Page</title></head>
        <main><h1>Test Page</h1><p>This is the main content.</p><script>ignore me</script></main>
        </html>
        """
        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.tools.get_grokipedia_page("Test Page")
        self.assertIn("# Test Page", result)
        self.assertIn("This is the main content", result)
        self.assertNotIn("ignore me", result)

    @patch("requests.get")
    def test_list_grokipedia_page_links(self, mock_get):
        """Test internal link extraction."""
        mock_html = """
        <html>
        <a href="/page/Linked_Page_1">Linked Page 1</a>
        <a href="/page/Linked_Page_2">Linked Page 2</a>
        <a href="/page/Main_Page">Skip Main</a>
        </html>
        """
        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.tools.list_grokipedia_page_links("Test Page")
        self.assertIn("Linked Page 1", result)
        self.assertIn("Linked Page 2", result)
        self.assertIn("→ https://grokipedia.com/page/Linked_Page_1", result)
        self.assertNotIn("Main_Page", result)

    @patch("requests.get")
    def test_error_handling(self, mock_get):
        """Test graceful error handling."""
        mock_get.side_effect = requests.RequestException("Connection failed")
        result = self.tools.grokipedia_search("bad query")
        self.assertIn("Error searching Grokipedia", result)


if __name__ == "__main__":
    unittest.main()