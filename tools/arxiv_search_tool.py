"""
title: searchthearxiv.com Tool
description: Tool to perform semantic search for relevant journals on arXiv via searchthearxiv.com, with PDF download and analysis capabilities
author: Haervwe, Tan Yong Sheng
author_urls:
  - https://github.com/Haervwe/
  - https://github.com/tan-yong-sheng/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.2.3
"""

import aiohttp
import asyncio
from typing import Any, Optional, Callable, Awaitable, List, Dict
from pydantic import BaseModel
import urllib.parse
import os
import tempfile
from pathlib import Path
import logging

try:
    from pypdf import PdfReader

    PDF_READER_AVAILABLE = True
except ImportError:
    PDF_READER_AVAILABLE = False
    logging.warning("PyPDF2 not available. PDF reading functionality will be limited.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tools:
    class UserValves(BaseModel):
        """No API keys required for arXiv search and PDF reading"""

        pass

    def __init__(self):
        self.base_url = "https://searchthearxiv.com/search"
        self.max_results = 5
        self.citation = False
        self.download_dir = os.path.join(tempfile.gettempdir(), "arxiv_downloads")

    async def download_pdf(
        self,
        arxiv_id: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Optional[str]:
        """
        Download a PDF file from arXiv for a given arXiv ID.

        Args:
            arxiv_id: The arXiv ID (e.g., "2301.12345")
            __event_emitter__: Optional event emitter for progress updates

        Returns:
            Path to the downloaded PDF file, or None if download fails
        """
        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Downloading PDF for arXiv ID: {arxiv_id}...",
                            "done": False,
                        },
                    }
                )

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
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

            # Generate safe filename from arXiv ID
            safe_filename = f"arxiv_{arxiv_id}.pdf"
            pdf_path = os.path.join(self.download_dir, safe_filename)

            with open(pdf_path, "wb") as f:
                f.write(content)

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"PDF downloaded successfully: {pdf_path}",
                            "done": True,
                        },
                    }
                )

            return pdf_path

        except aiohttp.ClientError as e:
            error_msg = f"Error downloading PDF from arXiv: {str(e)}"
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
        Read and extract text content from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content from the PDF, or None if reading fails
        """
        try:
            if not PDF_READER_AVAILABLE:
                raise ImportError(
                    "PyPDF2 is not available. Please install it: pip install pypdf"
                )

            reader = PdfReader(pdf_path)
            text_content = []

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)

            full_text = "\n".join(text_content)
            return full_text

        except ImportError:
            logger.error("PyPDF2 is not installed. Install it with: pip install pypdf")
            return None
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            return None

    async def analyze_pdf_content(
        self,
        pdf_path: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a PDF file to extract key information.

        Args:
            pdf_path: Path to the PDF file
            __event_emitter__: Optional event emitter for progress updates

        Returns:
            Dictionary containing analysis results including:
            - extracted_text: Full text extracted from PDF
            - page_count: Number of pages in the PDF
            - word_count: Estimated word count
            - analysis_summary: Brief analysis of the paper content
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
                    "PyPDF2 is not available. Please install it: pip install pypdf"
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

            reader = PdfReader(pdf_path)
            page_count = len(reader.pages)

            # Basic text analysis
            word_count = len(extracted_text.split())
            char_count = len(extracted_text)
            line_count = len(extracted_text.splitlines())

            # Simple content analysis (look for common sections)
            analysis_summary = {
                "total_pages": page_count,
                "estimated_words": word_count,
                "character_count": char_count,
                "line_count": line_count,
            }

            # Check for common paper sections
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
            analysis_summary["total_characters"] = char_count
            analysis_summary["avg_words_per_page"] = round(
                word_count / max(1, page_count), 2
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "PDF analysis completed", "done": True},
                    }
                )

            return {"extracted_text": extracted_text, **analysis_summary}

        except ImportError:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "PyPDF2 is not installed. Install with: pip install pypdf",
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
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Search for papers on a topic and download the specified number of PDFs.

        Args:
            topic: Topic to search for
            num_papers: Number of papers to download (default: 1)
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

        # Perform search
        search_results = await self.search_papers(
            topic, __event_emitter__=__event_emitter__
        )

        if "No papers found" in search_results:
            return {"error": "No papers found", "search_results": search_results}

        # Download top papers
        downloaded_files = []
        papers_data = search_results.split("\n\n")
        papers_metadata = []

        for i, paper_info in enumerate(papers_data[:num_papers], 1):
            if "URL:" in paper_info and "PDF URL:" in paper_info:
                # Extract arXiv ID from PDF URL
                pdf_url_match = paper_info.split("PDF URL:")[1].strip()
                arxiv_id = pdf_url_match.split("/")[-1].replace(".pdf", "")

                # Download PDF
                pdf_path = await self.download_pdf(
                    arxiv_id, __event_emitter__=__event_emitter__
                )

                if pdf_path:
                    downloaded_files.append(pdf_path)
                    papers_metadata.append(
                        {
                            "paper_info": paper_info,
                            "pdf_path": pdf_path,
                            "arxiv_id": arxiv_id,
                        }
                    )

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Successfully downloaded {len(downloaded_files)} papers",
                        "done": True,
                    },
                }
            )

        return {
            "search_results": search_results,
            "downloaded_files": downloaded_files,
            "papers_metadata": papers_metadata,
        }

    async def search_papers(
        self,
        topic: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search searchthearxiv.com for papers on a given topic and return formatted results.
        Args:
            topic: Topic to search for (e.g., "quantum computing", "transformer models")
        Returns:
            Formatted string containing paper details including titles, authors, dates,
            URLs and abstracts.
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Searching arXiv...",
                        "done": False,
                    },
                }
            )
        try:
            # Construct search query
            search_query = topic
            encoded_query = urllib.parse.quote(search_query)
            params = {"query": encoded_query}
            headers = {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/132.0.0.0 Safari/537.36",
                "x-requested-with": "XMLHttpRequest",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url, params=params, headers=headers, timeout=30
                ) as response:
                    response.raise_for_status()
                    # Use content_type=None to bypass MIME type checking.
                    root = await response.json(content_type=None)
            entries = root.get("papers", [])
            if not entries:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "No papers found", "done": True},
                        }
                    )
                return f"No papers found on arXiv related to '{topic}'"
            results = ""
            # Loop over each paper entry.
            for i, entry in enumerate(entries, 1):
                # Extract paper details with fallbacks
                title = entry.get("title")
                title_text = title.strip() if title else "Unknown Title"
                authors_str = entry.get("authors", "Unknown Authors")
                summary = entry.get("abstract")
                summary_text = summary.strip() if summary else "No summary available"
                link = entry.get("id")
                link_text = (
                    f"https://arxiv.org/abs/{link}" if link else "No link available"
                )
                pdf_link = (
                    f"https://arxiv.org/pdf/{link}" if link else "No link available"
                )
                year = entry.get("year")
                month = entry.get("month")
                pub_date = f"{month}-{int(year)}" if year and month else "Unknown Date"
                # Format paper entry
                results += f"{i}. {title_text}\n"
                results += f"   Authors: {authors_str}\n"
                results += f"   Published: {pub_date}\n"
                results += f"   URL: {link_text}\n"
                results += f"   PDF URL: {pdf_link}\n"
                results += f"   Summary: {summary_text}\n\n"
                # Emit citation data as provided.
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [summary_text],
                                "metadata": [{"source": pdf_link}],
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
        except aiohttp.ClientError as e:
            error_msg = f"Error searching arXiv: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error during search: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return error_msg