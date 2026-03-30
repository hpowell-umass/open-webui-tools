"""
title: Grokipedia Search Tool
description: Tool to search and retrieve comprehensive articles from Grokipedia (xAI's AI-powered encyclopedia) including content, images, and references
author: Grok (adapted from Haervwe Wikipedia tool)
author_url: https://x.ai
funding_url: https://github.com/Haervwe/open-webui-tools
requirements: aiohttp beautifulsoup4
version: 0.1.0
"""

import aiohttp
from typing import Optional, Any, Callable, Awaitable
from pydantic import BaseModel, Field
import logging
from bs4 import BeautifulSoup
import unittest
from unittest.mock import AsyncMock, patch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _get_grokipedia_page(title: str, user_agent: str) -> dict:
    """Fetch and parse a Grokipedia article page."""
    url = f"https://grokipedia.com/page/{title.replace(' ', '_')}"
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch page: HTTP {response.status}")
            html = await response.text()

    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    page_title = soup.find("h1")
    page_title = page_title.get_text().strip() if page_title else title.replace("_", " ")

    # Extract main content (Grokipedia uses markdown-like structure in HTML)
    content = []
    # Look for main article container (adjust selectors based on actual DOM if needed)
    main_content = soup.find("article") or soup.find("div", class_=lambda x: x and "content" in x.lower()) or soup.body

    if main_content:
        # Get all headings and paragraphs
        for element in main_content.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol"]):
            if element.name.startswith("h"):
                content.append(f"{'#' * int(element.name[1])} {element.get_text().strip()}")
            elif element.name == "p":
                text = element.get_text().strip()
                if text:
                    content.append(text)
            elif element.name in ["ul", "ol"]:
                for li in element.find_all("li", recursive=False):
                    content.append(f"- {li.get_text().strip()}")

    full_text = "\n\n".join(content) if content else "No content extracted."

    # Extract images (if any)
    images = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if src and not src.startswith("data:"):
            if not src.startswith("http"):
                src = "https://grokipedia.com" + src if src.startswith("/") else src
            alt = img.get("alt") or "Image"
            images.append({"title": alt, "url": src})

    return {
        "title": page_title,
        "url": url,
        "summary": full_text[:2000] + "..." if len(full_text) > 2000 else full_text,
        "full_text": full_text,
        "images": images[:6],  # limit images
        "html": html  # for advanced parsing if needed
    }


class Tools:
    class Valves(BaseModel):
        """Configuration for Grokipedia tool"""

        user_agent: str = Field(
            default="GrokipediaTool/1.0 (https://x.ai; Open-WebUI tool)",
            description="User agent for Grokipedia requests (required to avoid blocks)",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.base_url = "https://grokipedia.com"

    async def search_grokipedia(
        self,
        query: str,
        max_related: int = 3,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search Grokipedia and return comprehensive article information.

        Args:
            query: Search term / article title to retrieve
            max_related: Maximum number of related links to include

        Returns:
            Formatted markdown string with article content, images, and links
        """
        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Fetching from Grokipedia: {query}", "done": False},
                    }
                )

            data = await _get_grokipedia_page(query, self.valves.user_agent)

            # Build response
            result = f"# {data['title']}\n\n"

            # Main image if available
            if data["images"]:
                result += f"![{data['images'][0]['title']}]({data['images'][0]['url']})\n\n"

            result += f"{data['summary']}\n\n"

            # Full content (already cleaned)
            result += "## Article Content\n\n"
            result += data["full_text"] + "\n\n"

            # Gallery for additional images
            if len(data["images"]) > 1:
                result += "## Gallery\n\n"
                for img in data["images"][1:]:
                    result += f"![{img['title']}]({img['url']})\n\n"

            # References & Links
            result += "## References & Links\n"
            result += f"- Grokipedia Article: [{data['title']}]({data['url']})\n"

            # Note: Grokipedia has fewer "related articles" exposed; we can extend later with search

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "document": [data["full_text"]],
                            "metadata": [{"source": data["url"]}],
                            "source": {"name": data["title"]},
                        },
                    }
                )

                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Retrieved Grokipedia article: {data['title']}", "done": True},
                    }
                )

            return result

        except Exception as e:
            error_msg = f"Error fetching from Grokipedia: {str(e)}"
            logger.error(error_msg)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return error_msg


# ========================= UNIT TESTS =========================

class TestGrokipediaTool(unittest.IsolatedAsyncioTestCase):
    """Unit tests for the Grokipedia Search Tool."""

    def setUp(self):
        self.tools = Tools()
        self.tools.valves.user_agent = "GrokipediaTool-Test/1.0 (https://x.ai; test@example.com)"
        self.event_emitter = AsyncMock()

    async def test_search_grokipedia_george_washington_full_text(self):
        """Test that searching Grokipedia for George Washington returns full article text."""
        result = await self.tools.search_grokipedia(
            query="George Washington",
            max_related=2,
            __event_emitter__=self.event_emitter
        )

        self.assertIsInstance(result, str)
        self.assertIn("# George Washington", result)
        self.assertIn("https://grokipedia.com/page/George_Washington", result)
        self.assertGreater(len(result), 800, "Should contain substantial article content")

        lower_result = result.lower()
        self.assertIn("president", lower_result)
        self.assertIn("mount vernon", lower_result)
        self.assertIn("slavery", lower_result)  # Grokipedia often emphasizes this

        # Check citation emission
        calls = self.event_emitter.call_args_list
        citation_calls = [c for c in calls if c[0][0].get("type") == "citation"]
        if citation_calls:
            doc = citation_calls[0][0][0]["data"]["document"][0]
            self.assertGreater(len(doc), 1000, "Citation should include full extracted text")

    async def test_search_grokipedia_nonexistent(self):
        """Test graceful handling of non-existent pages."""
        result = await self.tools.search_grokipedia(
            query="ThisPageDefinitelyDoesNotExistOnGrokipedia12345XYZ",
            __event_emitter__=self.event_emitter
        )
        self.assertIn("Error fetching from Grokipedia", result)  # or adjust based on your error handling

    async def test_search_grokipedia_with_images(self):
        """Test image extraction path (mocked)."""
        with patch('__main__._get_grokipedia_page', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {
                "title": "George Washington",
                "url": "https://grokipedia.com/page/George_Washington",
                "summary": "George Washington was the first President...",
                "full_text": "Long text here...",
                "images": [
                    {"title": "Washington", "url": "https://example.com/gw.jpg"},
                    {"title": "Mount Rushmore", "url": "https://example.com/rushmore.jpg"}
                ]
            }

            result = await self.tools.search_grokipedia(
                query="George Washington",
                __event_emitter__=self.event_emitter
            )

            self.assertIn("![Washington](https://example.com/gw.jpg)", result)
            self.assertIn("## Gallery", result)
            mock_fetch.assert_called_once()

    async def test_full_text_extraction(self):
        """Ensure full_text is properly extracted and passed to citation."""
        result = await self.tools.search_grokipedia(
            query="Python (programming language)",
            __event_emitter__=self.event_emitter
        )

        self.assertIn("# Python", result)
        calls = self.event_emitter.call_args_list
        citation_calls = [c for c in calls if c[0][0].get("type") == "citation"]
        if citation_calls:
            doc = citation_calls[0][0][0]["data"]["document"][0]
            self.assertGreater(len(doc), 500)


if __name__ == '__main__':
    unittest.main(verbosity=2)