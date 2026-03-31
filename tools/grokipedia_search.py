"""
title: Grokipedia Search Tool
description: Tool to search Grokipedia (xAI's AI-generated encyclopedia) and retrieve comprehensive article information using BeautifulSoup
author: Grok (adapted from Haervwe Wikipedia tool)
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
requirements: aiohttp beautifulsoup4
version: 0.1.0
"""

import aiohttp
from typing import Optional, Any, Callable, Awaitable
from pydantic import BaseModel, Field
import logging
from bs4 import BeautifulSoup
from urllib.parse import quote
import unittest
from unittest.mock import AsyncMock, patch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tools:
    class Valves(BaseModel):
        """Configuration for Grokipedia tool"""

        user_agent: str = Field(
            default="OpenWebUI-GrokipediaTool/1.0 (https://github.com/Haervwe/open-webui-tools; contact@example.com)",
            description="User agent for Grokipedia requests - REQUIRED to avoid 403 errors",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.base_url = "https://grokipedia.com"
        self.search_url = f"{self.base_url}/search"

    async def _fetch_page(self, url: str) -> str:
        """Helper to fetch a page with proper headers."""
        headers = {
            "User-Agent": self.valves.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: status {response.status}")
                    return ""
                return await response.text()

    async def search_grokipedia(
        self,
        query: str,
        max_results: int = 5,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search Grokipedia and return a list of matching articles so the model can select one or more.
        
        Returns:
            Markdown-formatted list of search results with titles, URLs, and short snippets.
        """
        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Searching Grokipedia for: {query}",
                            "done": False,
                        },
                    }
                )

            search_query = quote(query)
            url = f"{self.search_url}?q={query}"
            html = await self._fetch_page(url)

            if not html:
                if __event_emitter__:
                    await __event_emitter__(
                        {"type": "status", "data": {"description": "Search failed", "done": True}}
                    )
                return "No results found or search failed."

            soup = BeautifulSoup(html, "html.parser")

            # Parse search results - Grokipedia uses h2 for titles and p for snippets
            results = []
            # Look for result blocks (h2 followed by descriptive p or div)
            headings = soup.find_all("h2")
            for heading in headings[:max_results]:
                title_tag = heading.find("a") or heading
                title = title_tag.get_text(strip=True)
                link = title_tag.get("href", "")
                if not link.startswith("/page/") and not link.startswith("http"):
                    continue
                if not link.startswith("http"):
                    link = self.base_url + link

                # Find nearest snippet (next sibling p or div)
                snippet = ""
                next_p = heading.find_next_sibling("p")
                if next_p:
                    snippet = next_p.get_text(strip=True)[:300]

                results.append({"title": title, "url": link, "snippet": snippet})

            if not results:
                return f"No search results found for '{query}'."

            # Build markdown output
            result_md = f"# Grokipedia Search Results for \"{query}\"\n\n"
            for i, r in enumerate(results, 1):
                result_md += f"{i}. **[{r['title']}]({r['url']})**\n"
                if r['snippet']:
                    result_md += f"   {r['snippet']}\n\n"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Found {len(results)} Grokipedia results for: {query}",
                            "done": True,
                        },
                    }
                )

            return result_md

        except Exception as e:
            error_msg = f"Error during Grokipedia search: {str(e)}"
            logger.error(error_msg)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return error_msg

    async def get_grokipedia_page(
        self,
        title: str,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Retrieve the full Grokipedia article for a specific title (use exact title from search results).
        
        Returns:
            Formatted markdown with full article text, sections, images, and references.
        """
        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Fetching Grokipedia page: {title}",
                            "done": False,
                        },
                    }
                )

            # Convert title to Grokipedia slug (spaces → underscores)
            slug = title.replace(" ", "_").replace("/", "_")
            url = f"{self.base_url}/page/{slug}"
            html = await self._fetch_page(url)

            if not html:
                if __event_emitter__:
                    await __event_emitter__(
                        {"type": "status", "data": {"description": "Page not found", "done": True}}
                    )
                return f"No Grokipedia article found for '{title}'."

            soup = BeautifulSoup(html, "html.parser")

            # Extract title
            h1 = soup.find("h1")
            page_title = h1.get_text(strip=True) if h1 else title

            # Main content container (fallback to body if no specific class)
            content = soup.find("article") or soup.find("main") or soup.body

            # Extract images
            images = []
            for img in content.find_all("img") if content else []:
                src = img.get("src")
                alt = img.get("alt", "")
                if src and not src.lower().endswith((".svg", ".gif")) and "http" in src:
                    if not src.startswith("http"):
                        src = self.base_url + src
                    images.append({"title": alt or page_title, "url": src})

            # Build full result
            result = f"# {page_title}\n\n"

            # Main image if available
            if images:
                result += f"![{images[0]['title']}]({images[0]['url']})\n\n"

            # Full text extraction + section parsing
            sections = []
            summary = ""
            current_section = {"title": "Introduction", "text": ""}

            if content:
                for tag in content.find_all(["h1", "h2", "h3", "p"]):
                    if tag.name in ["h2", "h3"]:
                        if current_section["text"].strip():
                            sections.append(current_section)
                        current_section = {"title": tag.get_text(strip=True), "text": ""}
                    elif tag.name == "p" and not summary:
                        summary = tag.get_text(strip=True)
                        current_section["text"] += tag.get_text(strip=True) + "\n\n"
                    elif tag.name == "p":
                        current_section["text"] += tag.get_text(strip=True) + "\n\n"

                if current_section["text"].strip():
                    sections.append(current_section)

            # Add summary
            if summary:
                result += f"{summary}\n\n"

            # Add sections
            if sections:
                result += "## Article Contents\n\n"
                for section in sections:
                    result += f"### {section['title']}\n{section['text']}\n\n"

            # Gallery
            if len(images) > 1:
                result += "## Gallery\n\n"
                for img in images[1:]:
                    result += f"![{img['title']}]({img['url']})\n\n"

            # References & links
            result += "## References & Links\n"
            result += f"- Grokipedia Article: [{page_title}]({url})\n"

            # Emit citation with full text
            if __event_emitter__:
                full_text = summary + "\n" + "\n".join([s["text"] for s in sections])
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "document": [full_text],
                            "metadata": [{"source": url}],
                            "source": {"name": page_title},
                        },
                    }
                )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Retrieved Grokipedia article: {page_title}",
                            "done": True,
                        },
                    }
                )

            return result

        except Exception as e:
            error_msg = f"Error fetching Grokipedia page: {str(e)}"
            logger.error(error_msg)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return error_msg


# ========================= UNIT TESTS =========================

class TestGrokipediaTool(unittest.IsolatedAsyncioTestCase):
    """Unit tests for the Grokipedia Search Tool (search + full page parsing)."""

    def setUp(self):
        self.tools = Tools()
        # Strong User-Agent to prevent 403 errors
        self.tools.valves.user_agent = "GrokipediaTool-Test/1.0 (https://github.com/Haervwe/open-webui-tools; test@example.com)"
        self.event_emitter = AsyncMock()

    async def test_search_grokipedia_returns_results(self):
        """Test the new search function returns usable results for selection."""
        result = await self.tools.search_grokipedia(
            query="George Washington",
            max_results=3,
            __event_emitter__=self.event_emitter
        )

        self.assertIsInstance(result, str)
        self.assertIn("# Grokipedia Search Results for \"George Washington\"", result)
        self.assertIn("George Washington", result)
        self.assertIn("/page/George_Washington", result)  # at least one valid link

        # Verify event emitter
        self.event_emitter.assert_called()
        calls = self.event_emitter.call_args_list
        status_calls = [c for c in calls if c[0][0]["type"] == "status"]
        self.assertGreater(len(status_calls), 1)

    async def test_get_grokipedia_page_george_washington_full_text(self):
        """Test that fetching a full Grokipedia page returns complete article text (requested test)."""
        result = await self.tools.get_grokipedia_page(
            title="George Washington",
            __event_emitter__=self.event_emitter
        )

        self.assertIsInstance(result, str)
        self.assertIn("# George Washington", result)
        self.assertIn("https://grokipedia.com/page/George_Washington", result)
        self.assertGreater(len(result), 1500, "Result should contain substantial full article text")

        # Verify key content from George Washington article
        lower_result = result.lower()
        self.assertIn("continental army", lower_result)
        self.assertIn("first president", lower_result)
        self.assertIn("mount vernon", lower_result)

        # Check citation full text was emitted
        calls = self.event_emitter.call_args_list
        citation_calls = [c for c in calls if c[0][0].get("type") == "citation"]
        if citation_calls:
            doc = citation_calls[0][0][0]["data"]["document"][0]
            self.assertGreater(len(doc), 2000, "Citation should include comprehensive full text")

    async def test_get_grokipedia_page_nonexistent(self):
        """Test graceful handling of non-existent pages."""
        result = await self.tools.get_grokipedia_page(
            title="ThisPageDoesNotExistXYZ12345",
            __event_emitter__=self.event_emitter
        )
        self.assertIn("No Grokipedia article found", result)

    async def test_search_and_get_page_workflow(self):
        """Test full workflow: search → select → get full page."""
        # Step 1: Search
        search_result = await self.tools.search_grokipedia(query="Python programming", max_results=1)
        self.assertIn("Python", search_result)

        # Step 2: Get page (using a known title)
        page_result = await self.tools.get_grokipedia_page(
            title="Python (programming language)",
            __event_emitter__=self.event_emitter
        )
        self.assertIn("# Python (programming language)", page_result)
        self.assertGreater(len(page_result), 1000)


if __name__ == '__main__':
    unittest.main(verbosity=2)