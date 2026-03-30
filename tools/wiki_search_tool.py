"""
title: Wikipedia Search Tool
description: Tool to search Wikipedia and retrieve comprehensive article information including images and references
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
requires: wikipedia-api aiohttp beautifulsoup4
version: 0.1.2
"""

import wikipediaapi
import aiohttp
from typing import Optional, Any, Callable, Awaitable
from pydantic import BaseModel, Field
import logging
from bs4 import BeautifulSoup
import unittest
import asyncio
from unittest.mock import AsyncMock, patch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _get_page_images(base_api_url: str, title: str, user_agent: str) -> list[dict]:
    """Fetch images for a Wikipedia page with proper User-Agent to avoid 403 errors."""
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "images|pageimages",
            "piprop": "original",
            "imlimit": "5",
        }

        async with session.get(base_api_url, params=params) as response:
            if response.status != 200:
                logger.warning(f"Image API returned status {response.status}")
                return []
            data = await response.json()

        page_id = list(data["query"]["pages"].keys())[0]
        page_data = data["query"]["pages"][page_id]

        images = []
        if "images" in page_data:
            image_titles = [
                img["title"]
                for img in page_data["images"]
                if not img["title"].lower().endswith((".svg", ".gif"))
            ]

            if image_titles:
                params = {
                    "action": "query",
                    "format": "json",
                    "titles": "|".join(image_titles[:5]),  # limit to avoid too long URL
                    "prop": "imageinfo",
                    "iiprop": "url",
                }

                async with session.get(base_api_url, params=params) as response:
                    if response.status != 200:
                        return images
                    img_data = await response.json()

                for page in img_data["query"]["pages"].values():
                    if "imageinfo" in page and page["imageinfo"]:
                        images.append(
                            {"title": page["title"], "url": page["imageinfo"][0]["url"]}
                        )

        return images


class Tools:
    class Valves(BaseModel):
        """Configuration for Wikipedia tool"""

        user_agent: str = Field(
            default="OpenWebUI-WikipediaTool/1.0 (https://github.com/Haervwe/open-webui-tools; contact@example.com)",
            description="User agent for Wikipedia API requests - REQUIRED to avoid 403 errors",
        )
        language: str = Field(
            default="en", description="Language for Wikipedia articles"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.wiki = None
        self.base_api_url = "https://en.wikipedia.org/w/api.php"
        self.image_prompt = """IMPORTANT INSTRUCTION FOR USING IMAGES:
1. The text after this contains Wikipedia article images in markdown format.
... (your original image_prompt remains the same)
"""

    async def search_wiki(
        self,
        query: str,
        max_results: int = 3,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search Wikipedia and return comprehensive article information including content, images, and references.
        """
        try:
            await _init_wiki(self, self.valves)

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Searching Wikipedia for: {query}",
                            "done": False,
                        },
                    }
                )

            # Get main page
            page = self.wiki.page(query)

            if not page.exists():
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "No Wikipedia article found",
                                "done": True,
                            },
                        }
                    )
                return "No Wikipedia article found for the given query."

            # Get images - FIXED: pass user_agent explicitly
            images = await _get_page_images(
                self.base_api_url, 
                page.title, 
                self.valves.user_agent
            )

            # Build response (rest of your original logic remains unchanged)
            result = f"# {page.title}\n\n"

            if images and len(images) > 0:
                result += f"![{page.title}]({images[0]['url']})\n\n"

            summary = BeautifulSoup(page.summary, "html.parser").get_text()
            result += f"{summary}\n\n"

            # Add sections
            sections = []
            for section in page.sections:
                if len(section.text.strip()) > 0:
                    clean_text = BeautifulSoup(section.text, "html.parser").get_text()
                    if len(clean_text) > 0:
                        sections.append({"title": section.title, "text": clean_text})

            if sections:
                result += "## Article Contents\n\n"
                for section in sections:
                    result += f"### {section['title']}\n{section['text']}\n\n"

            if len(images) > 1:
                result += "## Gallery\n\n"
                for img in images[1:]:
                    result += f"![{img['title']}]({img['url']})\n\n"

            result += f"## References & Links\n"
            result += f"- Wikipedia Article: [{page.title}]({page.fullurl})\n"

            result += "\n## Related Articles\n"
            for link_title, link_page in list(page.links.items())[:max_results]:
                if link_page.exists() and not link_title.startswith("File:"):
                    result += f"- [{link_title}]({link_page.fullurl})\n"

            # Emit citation
            if __event_emitter__:
                full_text = summary + "\n" + "\n".join([s["text"] for s in sections])
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "document": [full_text],
                            "metadata": [{"source": page.fullurl}],
                            "source": {"name": page.title},
                        },
                    }
                )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Found Wikipedia article: {page.title}",
                            "done": True,
                        },
                    }
                )

            return result

        except Exception as e:
            error_msg = f"Error during Wikipedia search: {str(e)}"
            logger.error(error_msg)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return error_msg


async def _init_wiki(wiki_instance, valves):
    """Initialize Wikipedia API client"""
    if not wiki_instance.wiki:
        wiki_instance.wiki = wikipediaapi.Wikipedia(
            user_agent=valves.user_agent,
            language=valves.language,
            extract_format=wikipediaapi.ExtractFormat.HTML,
        )


# ========================= UNIT TESTS (Updated) =========================

class TestWikipediaTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tools = Tools()
        # Good User-Agent to prevent 403
        self.tools.valves.user_agent = "WikipediaSearchTool-Test/1.0 (https://github.com/Haervwe/open-webui-tools; test@example.com)"
        self.event_emitter = AsyncMock()

    async def test_search_wiki_george_washington_full_text(self):
        """Test searching and parsing George Washington article (full text)."""
        result = await self.tools.search_wiki(
            query="George Washington",
            max_results=2,
            __event_emitter__=self.event_emitter
        )

        self.assertIsInstance(result, str)
        self.assertIn("# George Washington", result)
        self.assertIn("https://en.wikipedia.org/wiki/George_Washington", result)
        self.assertGreater(len(result), 1000)

        lower_result = result.lower()
        self.assertIn("first president", lower_result)
        self.assertIn("continental army", lower_result)

        # Check citation full text
        calls = self.event_emitter.call_args_list
        citation_calls = [c for c in calls if c[0][0].get("type") == "citation"]
        if citation_calls:
            doc = citation_calls[0][0][0]["data"]["document"][0]
            self.assertGreater(len(doc), 1500)

    # ... (you can keep the other tests from previous version)

if __name__ == '__main__':
    unittest.main(verbosity=2)