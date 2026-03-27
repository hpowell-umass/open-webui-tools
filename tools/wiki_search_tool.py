"""
title: Wikipedia Search Tool
description: Tool to search Wikipedia and retrieve comprehensive article information including images and references
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
requirements:wikipedia-api
version: 0.1.1
"""

import wikipediaapi
import aiohttp
from typing import Optional, Any, Callable, Awaitable
from pydantic import BaseModel, Field
import logging
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def _init_wiki(wiki_instance, valves):
    """Initialize Wikipedia API client if not already initialized"""
    if not wiki_instance.wiki:
        wiki_instance.wiki = wikipediaapi.Wikipedia(
            user_agent=valves.user_agent,
            language=valves.language,
            extract_format=wikipediaapi.ExtractFormat.HTML
        )

async def _get_page_images(base_api_url: str, title: str) -> list[dict]:
    """Fetch images for a Wikipedia page"""
    async with aiohttp.ClientSession() as session:
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "images|pageimages",
            "piprop": "original",
            "imlimit": "5"
        }
        
        async with session.get(base_api_url, params=params) as response:
            data = await response.json()
            
        page_id = list(data["query"]["pages"].keys())[0]
        page_data = data["query"]["pages"][page_id]
        
        images = []
        if "images" in page_data:
            image_titles = [img["title"] for img in page_data["images"]
                          if not img["title"].lower().endswith((".svg", ".gif"))]
            
            if image_titles:
                params = {
                    "action": "query",
                    "format": "json",
                    "titles": "|".join(image_titles),
                    "prop": "imageinfo",
                    "iiprop": "url"
                }
                
                async with session.get(base_api_url, params=params) as response:
                    img_data = await response.json()
                    
                for page in img_data["query"]["pages"].values():
                    if "imageinfo" in page:
                        images.append({
                            "title": page["title"],
                            "url": page["imageinfo"][0]["url"]
                        })
        
        return images

class Tools:
    class Valves(BaseModel):
        """Configuration for Wikipedia tool"""
        user_agent: str = Field(
            default="open-webui-tools/1.0",
            description="User agent for Wikipedia API requests"
        )
        language: str = Field(
            default="en",
            description="Language for Wikipedia articles"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.wiki = None
        self.citation = False
        self.base_api_url = "https://en.wikipedia.org/w/api.php"
        self.image_prompt = """IMPORTANT INSTRUCTION FOR USING IMAGES:
1. The text after this contains Wikipedia article images in markdown format.
2. Images are formatted like this example: ![Socrates](https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Socrate_du_Louvre.jpg/1024px-Socrate_du_Louvre.jpg)
3. You MUST use these images to:
   - Analyze visual details in the images
   - Reference specific images when discussing related content
   - Include visual descriptions in your responses
   - Help users better understand the topic through visual context
4. When you see an image, treat it as if you can fully see and analyze it
5. Images may show historical figures, objects, places, diagrams, or other relevant visuals
6. Make your responses more engaging by referring to what is shown in the images

Example of how to reference images:
"As we can see in the image, the marble bust shows Socrates with his characteristic snub nose and beard..."
"""

    async def search_wiki(
        self,
        query: str,
        max_results: int = 3,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> str:
        """
        Search Wikipedia and return comprehensive article information including content, images, and references.
        
        Args:
            query: Search term to find Wikipedia articles
            max_results: Maximum number of related articles to include (default: 3)
        
        Returns:
            Formatted string containing article information with images and links in markdown format
        """
        try:
            await _init_wiki(self, self.valves)
            
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Searching Wikipedia for: {query}", "done": False}
            })

            # Get main page
            page = self.wiki.page(query)
            
            if not page.exists():
                await __event_emitter__({
                    "type": "status", 
                    "data": {"description": "No Wikipedia article found", "done": True}
                })
                return "No Wikipedia article found for the given query."

            # Get images
            images = await _get_page_images(self.base_api_url, page.title)
            
            # Build response
            result = f"# {page.title}\n\n"

            # Add main image if available
            if images and len(images) > 0:
                result += f"![{page.title}]({images[0]['url']})\n\n"

            # Add full summary
            summary = BeautifulSoup(page.summary, "html.parser").get_text()
            result += f"{summary}\n\n"

            # Add complete sections without truncation
            sections = []
            for section in page.sections:
                if len(section.text.strip()) > 0:
                    clean_text = BeautifulSoup(section.text, "html.parser").get_text()
                    if len(clean_text) > 0:
                        sections.append({
                            "title": section.title,
                            "text": clean_text
                        })

            if sections:
                result += "## Article Contents\n\n"
                for section in sections:
                    result += f"### {section['title']}\n{section['text']}\n\n"

            # Add additional images in a gallery
            if len(images) > 1:
                result += "## Gallery\n\n"
                for img in images[1:]:
                    result += f"![{img['title']}]({img['url']})\n\n"

            # Add references and external links
            result += f"## References & Links\n"
            result += f"- Wikipedia Article: [{page.title}]({page.fullurl})\n"
            
            # Add related pages
            result += "\n## Related Articles\n"
            for link_title, link_page in list(page.links.items())[:max_results]:
                if link_page.exists() and not link_title.startswith("File:"):
                    result += f"- [{link_title}]({link_page.fullurl})\n"

            # Emit citation data
            if __event_emitter__:
                full_text = summary + "\n" + "\n".join([s["text"] for s in sections])
                await __event_emitter__({
                    "type": "citation",
                    "data": {
                        "document": [full_text],
                        "metadata": [{"source": page.fullurl}],
                        "source": {"name": page.title}
                    }
                })

            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"Found Wikipedia article: {page.title}",
                    "done": True
                }
            })

            return result

        except Exception as e:
            error_msg = f"Error during Wikipedia search: {str(e)}"
            logger.error(error_msg)
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg, "done": True}
                })
            return error_msg

import pytest
from unittest.mock import AsyncMock, Mock, patch
import json

# Assuming the Tools class is defined in the same file above
# We'll test the Tools class as defined in the original code

@pytest.mark.asyncio
async def test_tools_initialization():
    """Test that Tools initializes correctly"""
    tools = Tools()
    assert isinstance(tools.valves, Tools.Valves)
    assert tools.wiki is None
    assert tools.citation is False
    assert tools.base_api_url == "https://en.wikipedia.org/w/api.php"
    assert "IMPORTANT INSTRUCTION" in tools.image_prompt

@pytest.mark.asyncio
async def test_search_wiki_page_found():
    """Test search_wiki when a Wikipedia page is found"""
    tools = Tools()
    
    # Mock Wikipedia page
    mock_page = Mock()
    mock_page.exists.return_value = True
    mock_page.title = "Test Article"
    mock_page.summary = "<p>This is a test summary.</p>"
    mock_page.fullurl = "https://en.wikipedia.org/wiki/Test_Article"
    mock_page.sections = []
    mock_page.links = {}
    
    # Mock Wikipedia instance
    mock_wiki = Mock()
    mock_wiki.page.return_value = mock_page
    
    # Mock event emitter
    mock_event_emitter = AsyncMock()
    
    # Mock aiohttp session for image fetching
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.json = AsyncMock(return_value={
        "query": {
            "pages": {
                "123": {
                    "title": "Test Article",
                    "imageinfo": [{"url": "https://example.com/image.jpg"}]
                }
            }
        }
    })
    mock_session.get.return_value.__aenter__.return_value = mock_response
    mock_session.get.return_value.__aexit__.return_value = None
    
    with patch('wikipediaapi.Wikipedia') as mock_wiki_class, \
         patch('aiohttp.ClientSession') as mock_session_class:
        mock_wiki_class.return_value = mock_wiki
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        result = await tools.search_wiki(
            query="test",
            max_results=3,
            __event_emitter__=mock_event_emitter
        )
    
    # Verify result contains expected elements
    assert "# Test Article" in result
    assert "This is a test summary." in result
    assert "![Test Article](https://example.com/image.jpg)" in result
    assert "## References & Links" in result
    assert "- Wikipedia Article: [Test Article](https://en.wikipedia.org/wiki/Test_Article)" in result
    
    # Verify event emitter calls
    assert mock_event_emitter.call_count >= 3  # status, citation, status
    # Check first status call
    mock_event_emitter.assert_any_call({
        "type": "status",
        "data": {
            "description": "Searching Wikipedia for: test",
            "done": False
        }
    })
    # Check citation call
    mock_event_emitter.assert_any_call({
        "type": "citation",
        "data": {
            "document": [Mock()],  # We don't need to check exact content
            "metadata": [{"source": "https://en.wikipedia.org/wiki/Test_Article"}],
            "source": {"name": "Test Article"}
        }
    })
    # Check final status call
    mock_event_emitter.assert_any_call({
        "type": "status",
        "data": {
            "description": "Found Wikipedia article: Test Article",
            "done": True
        }
    })

@pytest.mark.asyncio
async def test_search_wiki_page_not_found():
    """Test search_wiki when no Wikipedia page is found"""
    tools = Tools()
    
    # Mock Wikipedia page that doesn't exist
    mock_page = Mock()
    mock_page.exists.return_value = False
    
    # Mock Wikipedia instance
    mock_wiki = Mock()
    mock_wiki.page.return_value = mock_page
    
    # Mock event emitter
    mock_event_emitter = AsyncMock()
    
    with patch('wikipediaapi.Wikipedia') as mock_wiki_class:
        mock_wiki_class.return_value = mock_wiki
        
        result = await tools.search_wiki(
            query="nonexistentarticle12345",
            __event_emitter__=mock_event_emitter
        )
    
    assert result == "No Wikipedia article found for the given query."
    
    # Verify event emitter calls
    assert mock_event_emitter.call_count == 2  # initial status and final status
    mock_event_emitter.assert_any_call({
        "type": "status",
        "data": {
            "description": "Searching Wikipedia for: nonexistentarticle12345",
            "done": False
        }
    })
    mock_event_emitter.assert_any_call({
        "type": "status",
        "data": {
            "description": "No Wikipedia article found",
            "done": True
        }
    })

@pytest.mark.asyncio
async def test_search_wiki_exception():
    """Test search_wiki when an exception occurs"""
    tools = Tools()
    
    # Mock event emitter
    mock_event_emitter = AsyncMock()
    
    with patch('wikipediaapi.Wikipedia') as mock_wiki_class:
        mock_wiki_class.side_effect = Exception("Test exception")
        
        result = await tools.search_wiki(
            query="test",
            __event_emitter__=mock_event_emitter
        )
    
    assert result == "Error during Wikipedia search: Test exception"
    
    # Verify event emitter calls for error
    assert mock_event_emitter.call_count == 2  # initial status and error status
    mock_event_emitter.assert_any_call({
        "type": "status",
        "data": {
            "description": "Searching Wikipedia for: test",
            "done": False
        }
    })
    mock_event_emitter.assert_any_call({
        "type": "status",
        "data": {
            "description": "Error during Wikipedia search: Test exception",
            "done": True
        }
    })

@pytest.mark.asyncio
async def test_search_wiki_with_sections_and_links():
    """Test search_wiki with sections and related articles"""
    tools = Tools()
    
    # Mock section
    mock_section = Mock()
    mock_section.title = "History"
    mock_section.text = "<p>This is the history section.</p>"
    
    # Mock link page
    mock_link_page = Mock()
    mock_link_page.exists.return_value = True
    mock_link_page.title = "Related Article"
    mock_link_page.fullurl = "https://en.wikipedia.org/wiki/Related_Article"
    
    # Mock Wikipedia page
    mock_page = Mock()
    mock_page.exists.return_value = True
    mock_page.title = "Test Article"
    mock_page.summary = "<p>Test summary.</p>"
    mock_page.fullurl = "https://en.wikipedia.org/wiki/Test_Article"
    mock_page.sections = [mock_section]
    mock_page.links = {
        "Related Article": mock_link_page,
        "Another Link": mock_link_page  # Duplicate for testing limit
    }
    
    # Mock Wikipedia instance
    mock_wiki = Mock()
    mock_wiki.page.return_value = mock_page
    
    # Mock event emitter
    mock_event_emitter = AsyncMock()
    
    # Mock aiohttp session (return no images for simplicity)
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.json = AsyncMock(return_value={"query": {"pages": {}}})
    mock_session.get.return_value.__aenter__.return_value = mock_response
    mock_session.get.return_value.__aexit__.return_value = None
    
    with patch('wikipediaapi.Wikipedia') as mock_wiki_class, \
         patch('aiohttp.ClientSession') as mock_session_class:
        mock_wiki_class.return_value = mock_wiki
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        result = await tools.search_wiki(
            query="test",
            max_results=1,  # Limit to 1 related article
            __event_emitter__=mock_event_emitter
        )
    
    # Verify sections are included
    assert "## Article Contents" in result
    assert "### History" in result
    assert "This is the history section." in result
    
    # Verify related articles (limited to max_results=1)
    assert "## Related Articles" in result
    # Count occurrences of related article links - should be exactly 1
    assert result.count("[Related Article]") == 1
    # Verify another link is not included due to limit
    assert result.count("[Another Link]") == 0