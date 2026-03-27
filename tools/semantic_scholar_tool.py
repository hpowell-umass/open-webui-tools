"""
title: Semantic Scholar Tool
description: Tool to perform search for relevant papers on Semantic Scholar (covering journals like IEEE, Nature, ION, and top ML conferences/journals), with PDF download and analysis capabilities where open access is available, including image extraction and recursive citation search.
author: Inspired by arXiv Tool
author_urls:
  - https://github.com/Haervwe/
  - https://github.com/tan-yong-sheng/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.1.0
"""

import aiohttp
import asyncio
from typing import Any, Optional, Callable, Awaitable, List, Dict
from pydantic import BaseModel, ConfigDict
import os
import tempfile
from pathlib import Path
import logging
import re
import unicodedata
import string
from datetime import datetime
import concurrent.futures

try:
    import fitz  # PyMuPDF
    PDF_READER_AVAILABLE = True
except ImportError:
    PDF_READER_AVAILABLE = False
    logging.warning("PyMuPDF not available. PDF reading functionality will be limited.")

try:
    from semanticscholar import SemanticScholar
    SEMANTIC_SCHOLAR_AVAILABLE = True
except ImportError:
    SEMANTIC_SCHOLAR_AVAILABLE = False
    logging.warning("semanticscholar package not available. Please install it: pip install semanticscholar")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tools:
    class UserValves(BaseModel):
        """No API keys required for Semantic Scholar search and PDF reading"""

        pass

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self):
        self.max_results = 5
        self.download_dir = os.path.join(tempfile.gettempdir(), "semanticscholar_downloads")
        if SEMANTIC_SCHOLAR_AVAILABLE:
            self.client = SemanticScholar()

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to make it safe for filesystem.
        """
        # Normalize unicode
        filename = unicodedata.normalize('NFKD', filename)
        # Remove invalid characters
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        filename = ''.join(c for c in filename if c in valid_chars)
        # Limit length
        filename = filename[:200]
        # Ensure it ends with .pdf
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        return filename

    async def download_pdf(
        self,
        pdf_url: str,
        title: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Optional[str]:
        """
        Download a PDF file from the given URL, then rename based on title.

        Args:
            pdf_url: The URL of the PDF
            title: Title to use for filename
            __event_emitter__: Optional event emitter for progress updates

        Returns:
            Path to the downloaded and renamed PDF file, or None if download fails
        """
        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Downloading PDF from: {pdf_url}...",
                            "done": False,
                        },
                    }
                )

            headers = {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/132.0.0.0 Safari/537.36",
                "x-requested-with": "XMLHttpRequest",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    pdf_url, headers=headers, timeout=60
                ) as response:
                    response.raise_for_status()
                    content = await response.read()

            # Create download directory if it doesn't exist
            os.makedirs(self.download_dir, exist_ok=True)

            # Temporary filename
            temp_filename = f"ss_{hash(pdf_url)}.pdf"
            temp_path = os.path.join(self.download_dir, temp_filename)

            with open(temp_path, "wb") as f:
                f.write(content)

            if not title or title == 'Unknown Title':
                title = f"ss_{hash(pdf_url)}"

            # Sanitize and create final filename
            safe_filename = self.sanitize_filename(title)
            pdf_path = os.path.join(self.download_dir, safe_filename)

            # Rename file
            os.rename(temp_path, pdf_path)

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"PDF downloaded and renamed successfully: {pdf_path}",
                            "done": True,
                        },
                    }
                )

            return pdf_path

        except aiohttp.ClientError as e:
            error_msg = f"Error downloading PDF: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            logger.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Unexpected error during PDF download: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            logger.error(error_msg)
            return None

    async def read_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Read and extract text content from a PDF file using PyMuPDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content from the PDF, or None if reading fails
        """
        try:
            if not PDF_READER_AVAILABLE:
                raise ImportError(
                    "PyMuPDF is not available. Please install it: pip install pymupdf"
                )

            doc = fitz.open(pdf_path)
            text_content = []

            for page in doc:
                text = page.get_text("text")
                if text:
                    text_content.append(text)

            full_text = "\n".join(text_content)
            doc.close()
            return full_text

        except ImportError:
            logger.error("PyMuPDF is not installed. Install it with: pip install pymupdf")
            return None
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            return None

    async def extract_images_from_pdf(self, pdf_path: str) -> List[str]:
        """
        Extract images from PDF and save them in the same directory.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of paths to extracted image files
        """
        try:
            if not PDF_READER_AVAILABLE:
                raise ImportError(
                    "PyMuPDF is not available. Please install it: pip install pymupdf"
                )

            doc = fitz.open(pdf_path)
            image_paths = []
            image_dir = os.path.dirname(pdf_path)

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)

                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        image_filename = f"figure_page{page_num+1}_img{img_index+1}.{image_ext}"
                        image_path = os.path.join(image_dir, image_filename)
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        image_paths.append(image_path)

            doc.close()
            return image_paths

        except ImportError:
            logger.error("PyMuPDF is not installed. Install it with: pip install pymupdf")
            return []
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {str(e)}")
            return []

    def describe_image(self, image_path: str) -> str:
        """
        Placeholder function to describe an image. 
        Since no VLM is integrated, provide a basic description.
        For real implementation, integrate with a vision model API.
        This gives qualitative info; quantitative for plots would require OCR/ML analysis.
        """
        # TODO: Integrate with a real VLM like GPT-4V or similar via API.
        # For now, dummy description.
        return f"Image at {image_path}: This appears to be a figure or plot. Qualitative: Visual representation of data. Quantitative: Unable to extract numbers without advanced analysis."

    async def extract_citations(self, pdf_text: str) -> List[str]:
        """
        Extract potential citations from the references section.

        Args:
            pdf_text: Full text of the PDF

        Returns:
            List of potential DOIs, arXiv IDs, or titles from citations
        """
        # Find references section
        lower_text = pdf_text.lower()
        ref_start = lower_text.find("references")
        if ref_start == -1:
            ref_start = lower_text.find("bibliography")
        if ref_start == -1:
            return []

        ref_text = pdf_text[ref_start:]
        
        # Extract potential DOIs
        dois = re.findall(r'doi:\s*([\w./-]+)', ref_text, re.IGNORECASE)
        # arXiv IDs
        arxiv_ids = re.findall(r'arXiv:(\d{4}\.\d{4,5})(v\d+)?', ref_text)
        arxiv_ids = [id[0] for id in arxiv_ids]
        
        # Extract titles (heuristic)
        citations = re.findall(r'\[\d+\]\s+.*?\.', ref_text, re.DOTALL)
        titles = [cit.split('.')[1].strip() for cit in citations if '.' in cit]

        return list(set(dois + arxiv_ids + titles))

    async def search_relevant_citations(
        self,
        citations: List[str],
        original_query: str,
        max_depth: int = 1,
        current_depth: int = 0,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Recursively search for relevant cited papers based on original query.

        Args:
            citations: List of citations (DOIs, IDs, titles)
            original_query: The original search topic
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth
            __event_emitter__: Optional event emitter

        Returns:
            Dictionary with recursive search results
        """
        if current_depth >= max_depth:
            return {}

        results = {}
        for cit in citations:
            # Search for the citation as a topic
            search_result = await self.search_papers(cit, __event_emitter__)
            if "No papers found" not in search_result:
                results[cit] = search_result

        return results

    async def analyze_pdf_content(
        self,
        pdf_path: str,
        original_query: Optional[str] = None,
        recursive_depth: int = 1,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a PDF file to extract key information, including images and citations.

        Args:
            pdf_path: Path to the PDF file
            original_query: Original search query for relevance in recursive search
            recursive_depth: Max depth for recursive citation search
            __event_emitter__: Optional event emitter for progress updates

        Returns:
            Dictionary containing analysis results
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Analyzing PDF content...", "done": False},
                }
            )

        try:
            if not PDF_READER_AVAILABLE:
                raise ImportError(
                    "PyMuPDF is not available. Please install it: pip install pymupdf"
                )

            # Read PDF content
            extracted_text = await self.read_pdf(pdf_path)

            if not extracted_text:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Failed to extract text from PDF",
                                "done": True,
                            },
                        }
                    )
                return {}

            doc = fitz.open(pdf_path)
            page_count = len(doc)

            # Basic text analysis
            word_count = len(extracted_text.split())
            char_count = len(extracted_text)
            line_count = len(extracted_text.splitlines())

            analysis_summary = {
                "total_pages": page_count,
                "estimated_words": word_count,
                "character_count": char_count,
                "line_count": line_count,
            }

            lower_text = extracted_text.lower()
            sections = {
                "has_abstract": "abstract" in lower_text,
                "has_introduction": "introduction" in lower_text,
                "has_methods": "method" in lower_text or "approach" in lower_text,
                "has_results": "result" in lower_text or "experiment" in lower_text,
                "has_conclusion": "conclusion" in lower_text,
                "has_references": "reference" in lower_text
                or "bibliography" in lower_text,
            }

            analysis_summary["sections"] = sections
            analysis_summary["avg_words_per_page"] = round(
                word_count / max(1, page_count), 2
            )

            # Extract images
            image_paths = await self.extract_images_from_pdf(pdf_path)
            image_descriptions = [self.describe_image(path) for path in image_paths]

            # Extract citations
            citations = await self.extract_citations(extracted_text)

            # Recursive search
            recursive_results = {}
            if original_query and citations:
                recursive_results = await self.search_relevant_citations(
                    citations, original_query, recursive_depth, 0, __event_emitter__
                )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "PDF analysis completed", "done": True},
                    }
                )

            doc.close()

            return {
                "extracted_text": extracted_text,
                **analysis_summary,
                "image_paths": image_paths,
                "image_descriptions": image_descriptions,
                "citations": citations,
                "recursive_results": recursive_results,
            }

        except ImportError:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "PyMuPDF is not installed. Install with: pip install pymupdf",
                            "done": True,
                        },
                    }
                )
            return {}
        except Exception as e:
            error_msg = f"Error analyzing PDF: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            logger.error(error_msg)
            return {}

    async def search_and_download_papers(
        self,
        topic: str,
        num_papers: int = 1,
        analyze_images: bool = True,
        recursive_depth: int = 1,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Search for papers on a topic and download available open access PDFs.

        Args:
            topic: Topic to search for
            num_papers: Number of papers to attempt download (default: 1)
            analyze_images: Whether to extract and analyze images
            recursive_depth: Depth for recursive citation search
            __event_emitter__: Optional event emitter for progress updates

        Returns:
            Dictionary with search results and downloaded PDF information
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Searching and downloading papers...",
                        "done": False,
                    },
                }
            )

        search_results = await self.search_papers(
            topic, __event_emitter__=__event_emitter__
        )

        if "No papers found" in search_results:
            return {"error": "No papers found", "search_results": search_results}

        downloaded_files = []
        papers_data = search_results.split("\n\n")
        papers_metadata = []

        for i, paper_info in enumerate(papers_data[:num_papers], 1):
            # Extract PDF URL if available
            if "Open Access PDF:" in paper_info:
                pdf_url_line = paper_info.split("Open Access PDF:")[1].split("\n")[0].strip()
                if pdf_url_line != "None":
                    # Extract title
                    title_line = paper_info.splitlines()[0].strip()
                    if ". " in title_line:
                        title = title_line.split(". ", 1)[1]
                    else:
                        title = title_line

                    # Download PDF
                    pdf_path = await self.download_pdf(
                        pdf_url_line, title, __event_emitter__=__event_emitter__
                    )

                    if pdf_path:
                        downloaded_files.append(pdf_path)
                        analysis = await self.analyze_pdf_content(
                            pdf_path, topic, recursive_depth, __event_emitter__
                        )
                        papers_metadata.append(
                            {
                                "paper_info": paper_info,
                                "pdf_path": pdf_path,
                                "analysis": analysis,
                            }
                        )
            else:
                # No open access PDF
                pass

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Successfully downloaded {len(downloaded_files)} open access papers",
                        "done": True,
                    },
                }
            )

        return {
            "search_results": search_results,
            "downloaded_files": downloaded_files,
            "papers_metadata": papers_metadata,
        }

    async def _search_semantic_scholar_with_retry(self, query: str, max_retries=5):
        for attempt in range(max_retries):
            try:
                return await self._rate_limited_request(
                    loop.run_in_executor(None, lambda: self.client.search_paper(query, limit=self.max_results)['data'])
                )
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    delay = (2 ** attempt) + random.uniform(0, 1)   # exponential backoff + jitter
                    await asyncio.sleep(delay)
                else:
                    raise
        raise Exception("Max retries exceeded")

    async def search_papers(
        self,
        topic: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search Semantic Scholar for papers on a given topic and return formatted results.
        Args:
            topic: Topic to search for (e.g., "transformer models")
        Returns:
            Formatted string containing paper details including titles, authors, dates,
            URLs and abstracts.
        """
        if not SEMANTIC_SCHOLAR_AVAILABLE:
            error_msg = "semanticscholar package is not available. Please install it: pip install semanticscholar"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return error_msg

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Searching Semantic Scholar...",
                        "done": False,
                    },
                }
            )
        try:
            loop = asyncio.get_running_loop()
            entries = await loop.run_in_executor(None, self._search_semantic_scholar_with_retry, topic)

            if not entries:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "No papers found", "done": True},
                        }
                    )
                return f"No papers found on Semantic Scholar related to '{topic}'"

            results = ""
            for i, entry in enumerate(entries, 1):
                title_text = entry.get('title', "Unknown Title").strip()
                authors_str = ", ".join(a['name'] for a in entry.get('authors', [])) or "Unknown Authors"
                summary_text = entry.get('abstract', "No summary available").strip()
                link_text = entry.get('url', "No link available")
                pdf_url = entry.get('openAccessPdf', {}).get('url', None) or "None"
                year = entry.get('year', "Unknown Year")

                results += f"{i}. {title_text}\n"
                results += f"   Authors: {authors_str}\n"
                results += f"   Published: {year}\n"
                results += f"   URL: {link_text}\n"
                results += f"   Open Access PDF: {pdf_url}\n"
                results += f"   Summary: {summary_text}\n\n"

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [summary_text],
                                "metadata": [{"source": pdf_url if pdf_url != "None" else link_text}],
                                "source": {"name": title_text},
                            },
                        }
                    )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Search completed", "done": True},
                    }
                )

            return results

        except Exception as e:
            error_msg = f"Unexpected error during search: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return error_msg

# ──────────────────────────────────────────────────────────────────────────────
#  TESTS – run when file is executed directly
# ──────────────────────────────────────────────────────────────────────────────

import unittest
import asyncio
import os
import tempfile
from pathlib import Path
import shutil

class TestSemanticScholarTool(unittest.IsolatedAsyncioTestCase):
    """Basic smoke / integration tests for the Semantic Scholar tool functionality"""

    def setUp(self):
        self.tools = Tools()
        # Use a temporary directory for downloads during tests
        self.test_download_dir = tempfile.mkdtemp()
        self.tools.download_dir = self.test_download_dir

    def tearDown(self):
        # Clean up temporary directory after each test
        if os.path.exists(self.test_download_dir):
            shutil.rmtree(self.test_download_dir, ignore_errors=True)

    async def test_search_papers_basic(self):
        """Test that search_papers returns some formatted results"""
        if not SEMANTIC_SCHOLAR_AVAILABLE:
            self.skipTest("semanticscholar package not installed")

        result = await self.tools.search_papers(
            topic="transformer neural networks",
            __event_emitter__=None
        )

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 50, "Search should return meaningful content")
        self.assertIn("Transformer", result, "Expected paper title fragment")
        self.assertIn("URL:", result, "Expected URL field in output")

    @unittest.skipUnless(SEMANTIC_SCHOLAR_AVAILABLE and PDF_READER_AVAILABLE,
                         "Requires semanticscholar and PyMuPDF")
    async def test_download_and_read_paper(self):
        """Download an open access paper and try to read its text"""
        # Use a known open access paper (e.g., Attention paper)
        pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
        title = "Attention Is All You Need"

        pdf_path = await self.tools.download_pdf(
            pdf_url=pdf_url,
            title=title,
            __event_emitter__=None
        )

        self.assertIsNotNone(pdf_path, "PDF should have been downloaded")
        self.assertTrue(os.path.isfile(pdf_path), "Downloaded file should exist")

        # Check filename
        filename = os.path.basename(pdf_path).lower()
        self.assertTrue("attention" in filename)

        text = await self.tools.read_pdf(pdf_path)
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 2000, "Should extract substantial text")
        self.assertIn("Transformer", text, "Expected content")

        # Cleanup
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    @unittest.skipUnless(PDF_READER_AVAILABLE, "PyMuPDF not available")
    async def test_sanitize_filename(self):
        """Test filename sanitization logic"""
        dirty = "Bad: Title /with\\ illegal*chars?.pdf"
        clean = self.tools.sanitize_filename(dirty)

        self.assertNotIn(":", clean)
        self.assertNotIn("/", clean)
        self.assertNotIn("\\", clean)
        self.assertNotIn("*", clean)
        self.assertTrue(clean.endswith(".pdf"))

    @unittest.skipUnless(SEMANTIC_SCHOLAR_AVAILABLE and PDF_READER_AVAILABLE,
                         "Requires semanticscholar + PyMuPDF")
    async def test_search_and_download_integration(self):
        """End-to-end test: search → download → basic analysis"""
        result = await self.tools.search_and_download_papers(
            topic="diffusion models",
            num_papers=1,
            analyze_images=False,
            recursive_depth=0,
            __event_emitter__=None
        )

        self.assertIsInstance(result, dict)
        self.assertIn("search_results", result)

        # May or may not download depending on open access, but check structure
        if result.get("downloaded_files"):
            pdf_path = result["downloaded_files"][0]
            self.assertTrue(os.path.isfile(pdf_path))

            analysis = result["papers_metadata"][0]["analysis"]
            self.assertIn("extracted_text", analysis)
            self.assertGreater(len(analysis["extracted_text"]), 500)

            # Cleanup
            for f in result["downloaded_files"]:
                if os.path.exists(f):
                    os.remove(f)

    async def test_extract_citations_smoke(self):
        """Check citation extraction doesn't crash"""
        fake_text = """
        References
        [1] Goodfellow et al. Generative Adversarial Nets. doi:10.48550/arXiv.1406.2661
        [2] arXiv:2010.09876 Ho et al. Denoising Diffusion Probabilistic Models.
        [3] Kingma & Welling. Auto-Encoding Variational Bayes. doi:10.48550/arXiv.1312.6114
        """
        citations = await self.tools.extract_citations(fake_text)

        self.assertIsInstance(citations, list)
        self.assertGreater(len(citations), 1)
        self.assertTrue(any("10.48550" in c for c in citations))


if __name__ == "__main__":
    print("Running Semantic Scholar Tool self-tests...")
    print("Make sure 'semanticscholar' and 'pymupdf' are installed for full coverage.\n")

    # Run asyncio-compatible unittest
    asyncio.run(unittest.main())

    print("\nTests completed.")