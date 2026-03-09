"""
title: arXiv Tool
description: Tool to perform search for relevant papers on arXiv using the official arXiv API via the arxiv Python package, with PDF download and analysis capabilities, including image extraction and recursive citation search.
author: Haervwe, Tan Yong Sheng
author_urls:
  - https://github.com/Haervwe/
  - https://github.com/tan-yong-sheng/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.2.5
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
import re
import unicodedata
import string
from datetime import datetime

try:
    import fitz  # PyMuPDF
    PDF_READER_AVAILABLE = True
except ImportError:
    PDF_READER_AVAILABLE = False
    logging.warning("PyMuPDF not available. PDF reading functionality will be limited.")

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    logging.warning("arxiv package not available. Please install it: pip install arxiv")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tools:
    class UserValves(BaseModel):
        """No API keys required for arXiv search and PDF reading"""

        pass

    def __init__(self):
        self.max_results = 5
        self.citation = False
        self.download_dir = os.path.join(tempfile.gettempdir(), "arxiv_downloads")
        if ARXIV_AVAILABLE:
            self.client = arxiv.Client()

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
        arxiv_id: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Optional[str]:
        """
        Download a PDF file from arXiv for a given arXiv ID, then rename based on title.

        Args:
            arxiv_id: The arXiv ID (e.g., "2301.12345")
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

            # Temporary filename
            temp_filename = f"arxiv_{arxiv_id}.pdf"
            temp_path = os.path.join(self.download_dir, temp_filename)

            with open(temp_path, "wb") as f:
                f.write(content)

            # Now read the PDF to get the title using PyMuPDF
            if PDF_READER_AVAILABLE:
                doc = fitz.open(temp_path)
                metadata = doc.metadata
                title = metadata.get('title', None)
                if not title:
                    # Fallback: extract from first page
                    first_page = doc.load_page(0)
                    text = first_page.get_text("text")
                    title_lines = text.splitlines()[:5]  # Assume title in first few lines
                    title = ' '.join(line.strip() for line in title_lines if line.strip()).strip()
                doc.close()
            else:
                title = None

            if not title or title == 'Unknown Title':
                title = f"arxiv_{arxiv_id}"

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
        Extract potential arXiv citations from the references section.

        Args:
            pdf_text: Full text of the PDF

        Returns:
            List of potential arXiv IDs or titles from citations
        """
        # Find references section
        lower_text = pdf_text.lower()
        ref_start = lower_text.find("references")
        if ref_start == -1:
            ref_start = lower_text.find("bibliography")
        if ref_start == -1:
            return []

        ref_text = pdf_text[ref_start:]
        
        # Extract potential arXiv IDs
        arxiv_ids = re.findall(r'arXiv:(\d{4}\.\d{4,5})(v\d+)?', ref_text)
        arxiv_ids = [id[0] for id in arxiv_ids]
        
        # Extract titles (heuristic: lines that look like [num] Author. Title. Year.)
        citations = re.findall(r'\[\d+\]\s+.*?\.', ref_text, re.DOTALL)
        titles = [cit.split('.')[1].strip() for cit in citations if '.' in cit]

        return list(set(arxiv_ids + titles))

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
            citations: List of citations (IDs or titles)
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
                # Could download and analyze further recursively
                # For now, just search; extend if needed

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
            Dictionary containing analysis results including:
            - extracted_text: Full text extracted from PDF
            - page_count: Number of pages in the PDF
            - word_count: Estimated word count
            - analysis_summary: Brief analysis of the paper content
            - image_paths: Paths to extracted images
            - image_descriptions: Text descriptions of images
            - citations: Extracted citations
            - recursive_results: Results from recursive searches
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

            # Extract images
            image_paths = await self.extract_images_from_pdf(pdf_path)
            # Describe images (placeholder)
            image_descriptions = [self.describe_image(path) for path in image_paths]

            # Extract citations
            citations = await self.extract_citations(extracted_text)

            # Recursive search if original_query provided
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
        Search for papers on a topic and download the specified number of PDFs.

        Args:
            topic: Topic to search for
            num_papers: Number of papers to download (default: 1)
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
                    analysis = await self.analyze_pdf_content(
                        pdf_path, topic, recursive_depth, __event_emitter__
                    )
                    papers_metadata.append(
                        {
                            "paper_info": paper_info,
                            "pdf_path": pdf_path,
                            "arxiv_id": arxiv_id,
                            "analysis": analysis,
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
        Search arXiv for papers on a given topic using the arxiv Python package and return formatted results.
        Args:
            topic: Topic to search for (e.g., "quantum computing", "transformer models")
        Returns:
            Formatted string containing paper details including titles, authors, dates,
            URLs and abstracts.
        """
        if not ARXIV_AVAILABLE:
            error_msg = "arxiv package is not available. Please install it: pip install arxiv"
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
                        "description": "Searching arXiv...",
                        "done": False,
                    },
                }
            )
        try:
            search = arxiv.Search(
                query=topic,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            entries = list(self.client.results(search))

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
                title_text = entry.title.strip() if entry.title else "Unknown Title"
                authors_str = ", ".join(author.name for author in entry.authors) if entry.authors else "Unknown Authors"
                summary_text = entry.summary.strip() if entry.summary else "No summary available"
                link_text = entry.entry_id if entry.entry_id else "No link available"
                pdf_link = entry.pdf_url if entry.pdf_url else "No link available"
                pub_date = entry.published.strftime("%B-%Y") if isinstance(entry.published, datetime) else "Unknown Date"

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

        except Exception as e:
            error_msg = f"Unexpected error during search: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return error_msg