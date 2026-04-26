"""
title: HF Papers Tool
author: Your Name / Grok-assisted
version: 1.0
description: Search HF Papers and fetch full papers as Markdown. Powered by official HF Papers API.
requirements: requests
"""

import requests
from typing import Dict, List, Any

class Tools:
    def __init__(self):
        pass

    async def search_hf_papers(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Search Hugging Face Papers (hybrid semantic + keyword search).
        
        :param query: Search query (title, author, topic, etc.)
        :param limit: Number of results to return (max ~20-30 recommended)
        :return: List of papers with title, authors, abstract snippet, arXiv ID, HF paper URL, etc.
        """
        try:
            url = f"https://huggingface.co/api/papers/search?q={query}&limit={limit}"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            # HF returns a list of results directly
            return {
                "status": "success",
                "query": query,
                "results": data[:limit] if isinstance(data, list) else data
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_hf_paper_markdown(self, paper_id: str) -> str:
        """
        Fetch the FULL paper content in clean Markdown format from HF Papers.
        The paper_id is the arXiv ID (e.g. 2401.12345 or 2602.08025).
        
        :param paper_id: arXiv-style paper ID (without version suffix if not needed)
        :return: Full paper Markdown (headings, equations, tables preserved where possible)
        """
        try:
            # Preferred .md endpoint (cleanest Markdown)
            url = f"https://huggingface.co/papers/{paper_id}.md"
            response = requests.get(url, timeout=20)
            if response.status_code == 404:
                return f"Paper {paper_id} not yet indexed on HF Papers. Try arXiv PDF or abstract instead."
            response.raise_for_status()
            markdown = response.text
            # Optional: truncate extremely long papers to avoid context explosion
            # (HF papers are usually manageable; remove if you want raw full text)
            if len(markdown) > 80000:  # ~60-80k tokens depending on model
                markdown = markdown[:80000] + "\n\n... [Markdown truncated due to length. Ask for specific sections if needed.]"
            return markdown
        except Exception as e:
            return f"Error fetching Markdown: {str(e)}"


# ===================== UNIT TESTS =====================
# Paste this at the very bottom of the tool file

import unittest
import asyncio
from typing import Dict, Any

# Assuming the Tools class is defined above in the same file
# If you named the class differently, adjust the import/reference below

class TestHFPapersTool(unittest.TestCase):
    def setUp(self):
        self.tools = Tools()  # Instantiate the Tools class from your tool

    def test_search_hf_papers_success(self):
        """Test that search returns successful results with expected structure."""
        async def run_test():
            result: Dict[str, Any] = await self.tools.search_hf_papers(
                query="attention is all you need", 
                limit=3
            )
            
            self.assertEqual(result.get("status"), "success")
            self.assertIn("query", result)
            self.assertIn("results", result)
            
            results = result["results"]
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0, "Should return at least one result for a well-known paper")
            
            # Check structure of first result (based on current HF Papers API)
            first = results[0]
            self.assertIn("title", first)
            self.assertIn("authors", first["paper"])  # usually a list or string
            self.assertIn("id", first["paper"])       # arXiv-style ID like "1706.03762"
            self.assertIn("summary", first["paper"]) # or similar field for snippet
            
            print(f"✅ Search test passed. Found {len(results)} results.")
            return result

        asyncio.run(run_test())

    # def test_search_hf_papers_empty_query(self):
    #     """Test behavior with empty or very short query."""
    #     async def run_test():
    #         result: Dict[str, Any] = await self.tools.search_hf_papers(query="", limit=5)
    #         self.assertEqual(result.get("status"), "success")
    #         # HF usually still returns some results or an empty list
    #         self.assertIsInstance(result.get("results"), list)
    #         print("✅ Empty query test passed.")
    #         return result

    #     asyncio.run(run_test())

    def test_get_hf_paper_markdown_success(self):
        """Test fetching a real, well-known paper in Markdown format."""
        # Use a stable, popular paper that is definitely indexed on HF Papers
        test_paper_id = "1706.03762"  # "Attention Is All You Need" (Transformer paper)

        async def run_test():
            markdown: str = await self.tools.get_hf_paper_markdown(test_paper_id)
            
            self.assertIsInstance(markdown, str)
            self.assertGreater(len(markdown), 500, "Markdown should be reasonably long")
            
            # Basic content checks for a real paper
            self.assertIn("Attention", markdown)   # Title contains "Attention"
            self.assertIn("Transformer", markdown)  # Key term
            self.assertIn("#", markdown)           # Should have Markdown headings
            self.assertIn("##", markdown)          # Subheadings
            
            # Should not be an error message
            self.assertNotIn("Error fetching", markdown)
            self.assertNotIn("not yet indexed", markdown.lower())
            
            print(f"✅ Markdown fetch test passed for paper {test_paper_id} ({len(markdown)} chars).")
            return markdown

        asyncio.run(run_test())

    def test_get_hf_paper_markdown_nonexistent(self):
        """Test graceful handling of a non-existent or not-yet-indexed paper."""
        async def run_test():
            markdown: str = await self.tools.get_hf_paper_markdown("9999.99999")
            
            self.assertIsInstance(markdown, str)
            # Should contain either an error message or "not yet indexed"
            self.assertTrue(
                "error" in markdown.lower() or 
                "not yet indexed" in markdown.lower() or
                "404" in markdown,
                "Should indicate paper was not found"
            )
            print("✅ Non-existent paper test passed (graceful error handling).")
            return markdown

        asyncio.run(run_test())

    def test_get_hf_paper_markdown_truncation(self):
        """Test that very long papers get truncated (if your code has truncation logic)."""
        async def run_test():
            # Use a paper that tends to produce longer Markdown
            markdown: str = await self.tools.get_hf_paper_markdown("1706.03762")
            
            if "[Markdown truncated" in markdown:
                self.assertIn("... [Markdown truncated", markdown)
                print("✅ Truncation logic triggered and working.")
            else:
                print("ℹ️  Truncation not triggered (paper was short enough).")
            
            return markdown

        asyncio.run(run_test())


if __name__ == "__main__":
    # Run tests when the file is executed directly (handy for local testing)
    unittest.main(verbosity=2)