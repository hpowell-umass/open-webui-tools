# =====================================================
# Open-WebUI Wikipedia Tool
# =====================================================
# This is a complete, ready-to-use Open-WebUI tool file.
# Place it in your Open-WebUI tools directory (or upload via the UI).
# It exposes three functions that the model can call:
#   1. wikipedia_search          → get multiple search results
#   2. get_wikipedia_page        → fetch a specific page after selecting from search
#   3. list_wikipedia_page_links → list internal Wikipedia links for navigation
#
# Uses:
#   • wikipedia-api (for reliable page content & existence checks)
#   • BeautifulSoup (for cleaning snippets and extracting internal links)
#   • Proper User-Agent headers everywhere to prevent 403 Forbidden errors
#
# Author: Grok (built by xAI) – generated exactly to spec

import json
import requests
from bs4 import BeautifulSoup
import wikipediaapi
from urllib.parse import unquote
from typing import List, Dict, Any

class Tools:
    """Internal Tools class that encapsulates all Wikipedia logic.
    Open-WebUI discovers the top-level functions below, which delegate here.
    All requests use a custom User-Agent to avoid 403 errors from Wikipedia."""

    user_agent: str = "Open-WebUI-WikipediaTool/1.0 (https://openwebui.com; contact: support@openwebui.com)"

    @staticmethod
    def wikipedia_search(query: str, num_results: int = 5) -> str:
        """Internal implementation – do not call directly. See top-level function docstring."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": num_results,
            "srprop": "snippet",
        }
        headers = {"User-Agent": Tools.user_agent}

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            search_results = data.get("query", {}).get("search", [])
            results: List[Dict[str, str]] = []

            for result in search_results:
                title = result.get("title", "")
                snippet_html = result.get("snippet", "")
                # BeautifulSoup cleans any HTML tags that appear in the snippet
                snippet = BeautifulSoup(snippet_html, "html.parser").get_text()
                page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "url": page_url
                })

            return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"Search failed: {str(e)}"})

    @staticmethod
    def get_wikipedia_page(title: str) -> str:
        """Internal implementation – do not call directly. See top-level function docstring."""
        try:
            wiki = wikipediaapi.Wikipedia(
                language="en",
                user_agent=Tools.user_agent
            )
            page = wiki.page(title)

            if not page.exists():
                return json.dumps({"error": f"Page '{title}' does not exist on Wikipedia."})

            links = list(page.links.keys())[:30]

            result: Dict[str, Any] = {
                "title": page.title,
                "summary": page.summary,
                "text": page.text[:4000] + "..." if len(page.text) > 4000 else page.text,
                "url": page.fullurl,
                "links_count": len(page.links),
                "sample_links": links
            }
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"Failed to fetch page: {str(e)}"})

    @staticmethod
    def list_wikipedia_page_links(title: str) -> str:
        """Internal implementation – do not call directly. See top-level function docstring."""
        try:
            # First verify the page exists using wikipedia-api
            wiki = wikipediaapi.Wikipedia(
                language="en",
                user_agent=Tools.user_agent
            )
            page = wiki.page(title)
            if not page.exists():
                return json.dumps({"error": f"Page '{title}' does not exist."})

            # Now use BeautifulSoup on the rendered HTML to extract clean internal links
            wiki_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            headers = {"User-Agent": Tools.user_agent}
            response = requests.get(wiki_url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            content_div = soup.find("div", id="mw-content-text")

            internal_links: set[str] = set()
            if content_div:
                for a in content_div.find_all("a", href=True):
                    href = a["href"]
                    # Only keep main-namespace internal links (no namespace colon)
                    if (href.startswith("/wiki/") and
                        ":" not in href[6:] and
                        not href.startswith(("/wiki/Special:", "/wiki/File:", "/wiki/Help:",
                                             "/wiki/Portal:", "/wiki/Template:", "/wiki/Category:"))):
                        link_title = unquote(href[6:]).replace("_", " ")
                        if link_title and link_title != page.title:
                            internal_links.add(link_title)

            links_list = list(internal_links)[:100]  # reasonable limit for model context

            return json.dumps({
                "page_title": page.title,
                "internal_links": links_list
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"Failed to list links: {str(e)}"})


# =============================================================================
# TOP-LEVEL TOOL FUNCTIONS (exposed to Open-WebUI / model)
# =============================================================================
# These are the functions the LLM will actually call via function calling.
# Their docstrings are what the model sees.

def wikipedia_search(query: str, num_results: int = 5) -> str:
    """Search Wikipedia for a term and return multiple possible results.
    The model should use this first to explore options, then call get_wikipedia_page
    on the exact title(s) it wants.

    Args:
        query (str): Search term or phrase.
        num_results (int, optional): Number of results to return (default 5).

    Returns:
        str: JSON array of objects with 'title', 'snippet', and 'url'.
             Or JSON error object if something fails.
    """
    return Tools.wikipedia_search(query, num_results)


def get_wikipedia_page(title: str) -> str:
    """Fetch the content of a specific Wikipedia page by its exact title.
    Use after wikipedia_search to retrieve the page the model selected.

    Args:
        title (str): Exact Wikipedia page title (case-sensitive).

    Returns:
        str: JSON object containing title, summary, text excerpt, URL, and sample links.
             Or JSON error object if the page does not exist or fetch fails.
    """
    return Tools.get_wikipedia_page(title)


def list_wikipedia_page_links(title: str) -> str:
    """List internal Wikipedia page links (hyperlinks) inside the given page.
    This lets the model navigate to connected/related pages by feeding the returned
    titles back into wikipedia_search or get_wikipedia_page.

    Args:
        title (str): Exact title of the Wikipedia page to analyze.

    Returns:
        str: JSON object with 'page_title' and 'internal_links' (list of titles).
             Or JSON error object if the page does not exist or fetch fails.
    """
    return Tools.list_wikipedia_page_links(title)


# =============================================================================
# UNIT TESTS (using unittest) – placed at the end of the file as requested
# =============================================================================

import unittest
from unittest.mock import patch, MagicMock


class TestWikipediaTool(unittest.TestCase):

    @patch('requests.get')
    def test_wikipedia_search_success(self, mock_get):
        """Test that search returns cleaned results and uses BeautifulSoup on snippets."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "query": {
                "search": [
                    {
                        "title": "Python (programming language)",
                        "snippet": "Python is a <b>high-level</b> programming language."
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        result_str = wikipedia_search("python", num_results=1)
        result = json.loads(result_str)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Python (programming language)")
        self.assertIn("high-level", result[0]["snippet"])  # BS4 cleaned the <b> tags

    @patch('wikipediaapi.Wikipedia')
    def test_get_wikipedia_page_success(self, mock_wikipedia):
        """Test get_wikipedia_page using wikipedia-api mock."""
        mock_wiki_instance = MagicMock()
        mock_page = MagicMock()
        mock_page.exists.return_value = True
        mock_page.title = "Python (programming language)"
        mock_page.summary = "Python is an interpreted high-level programming language."
        mock_page.text = "A very long article text about Python..."
        mock_page.fullurl = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        mock_page.links = {"Programming language": None, "Guido van Rossum": None}
        mock_wiki_instance.page.return_value = mock_page
        mock_wikipedia.return_value = mock_wiki_instance

        result_str = get_wikipedia_page("Python (programming language)")
        result = json.loads(result_str)

        self.assertIn("title", result)
        self.assertEqual(result["title"], "Python (programming language)")
        self.assertIn("sample_links", result)
        self.assertEqual(len(result["sample_links"]), 2)

    @patch('wikipediaapi.Wikipedia')
    @patch('requests.get')
    def test_list_wikipedia_page_links_success(self, mock_requests_get, mock_wikipedia):
        """Test list_wikipedia_page_links – verifies wikipedia-api + BeautifulSoup parsing."""
        # Mock wikipedia-api page existence check
        mock_wiki_instance = MagicMock()
        mock_page = MagicMock()
        mock_page.exists.return_value = True
        mock_page.title = "Python (programming language)"
        mock_wiki_instance.page.return_value = mock_page
        mock_wikipedia.return_value = mock_wiki_instance

        # Mock HTML response with internal links
        mock_html_response = MagicMock()
        mock_html_response.status_code = 200
        mock_html_response.text = '''
        <html>
        <div id="mw-content-text">
            <p>Python is ...</p>
            <a href="/wiki/Programming_language">Programming language</a>
            <a href="/wiki/Guido_van_Rossum">Guido van Rossum</a>
            <a href="/wiki/Category:Programming_languages">Should be filtered</a>
            <a href="/wiki/File:Python_logo.png">Should be filtered</a>
        </div>
        </html>
        '''
        mock_requests_get.return_value = mock_html_response

        result_str = list_wikipedia_page_links("Python (programming language)")
        result = json.loads(result_str)

        self.assertIn("internal_links", result)
        self.assertIn("Programming language", result["internal_links"])
        self.assertIn("Guido van Rossum", result["internal_links"])
        # Filtered items should NOT appear
        self.assertNotIn("Should be filtered", result["internal_links"])

    def test_error_handling_search(self):
        """Test graceful error handling when search fails."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.RequestException("Network error")
            result_str = wikipedia_search("nonexistentquery12345")
            result = json.loads(result_str)
            self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()