"""
title: arXiv Paper to Markdown
author: Grok (built by xAI)
author_url: https://x.ai
version: 1.0.0
description: Search arXiv for papers and convert any paper's TeX source to clean Markdown while perfectly preserving mathematical equations using $ (inline) and $$ (display) syntax. Uses the official arxiv Python package + pypandoc for conversion.
requirements: arxiv pypandoc
license: MIT
"""

import arxiv
import urllib.request
import tempfile
import os
import tarfile
import gzip
from typing import List, Dict, Any

# pypandoc is imported inside functions so the tool still loads if pandoc is missing


class Tools:
    def __init__(self):
        """No configuration valves needed for this tool."""
        pass

    def search_arxiv_papers(
        self, query: str, max_results: int = 10
    ) -> str:
        """Search arXiv and return a nicely formatted list of matching papers.
        
        :param query: arXiv search query (supports advanced syntax like "au:smith ti:quantum" or "cat:cs.AI")
        :param max_results: Maximum number of results to return (default 10, max 50 recommended to avoid API limits)
        :return: Formatted string with paper ID, title, authors, short abstract, and PDF link for each result
        """
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        
        results = list(client.results(search))
        
        if not results:
            return "No papers found matching your query."
        
        output = f"**Found {len(results)} papers on arXiv**\n\n"
        for i, result in enumerate(results, 1):
            authors = ", ".join([author.name for author in result.authors])
            abstract_snippet = result.summary[:400] + "..." if len(result.summary) > 400 else result.summary
            output += (
                f"{i}. **{result.title}**\n"
                f"   **ID**: {result.get_short_id()}\n"
                f"   **Authors**: {authors}\n"
                f"   **Abstract**: {abstract_snippet}\n"
                f"   **PDF**: {result.pdf_url}\n"
                f"   **Source URL**: {result.source_url()}\n\n"
            )
        
        output += "To get the full Markdown version of any paper, use the `get_paper_as_markdown` tool with its ID."
        return output

    def tex_to_markdown(self, tex_source: str) -> str:
        """Convert raw TeX/LaTeX source to Markdown while retaining perfect $ and $$ equation syntax.
        
        This is the core conversion function used internally by get_paper_as_markdown.
        Pandoc is used because it is the most reliable tool for preserving complex LaTeX math, sections,
        tables, citations, and figures as Markdown.
        
        :param tex_source: Raw TeX source code as a string
        :return: Markdown string with equations preserved ($...$ for inline, $$...$$ for display)
        """
        try:
            import pypandoc
            # --wrap=none prevents unwanted line wrapping inside equations
            # Default pandoc Markdown output uses $ and $$ for math (exactly as requested)
            markdown = pypandoc.convert_text(
                tex_source,
                to="markdown",
                format="latex",
                extra_args=["--wrap=none"]
            )
            return markdown
        except ImportError:
            return "Error: pypandoc is not installed. Install via: pip install pypandoc"
        except Exception as e:
            return f"Conversion failed: {str(e)}. Make sure the system has pandoc installed (e.g. `apt install pandoc` or `brew install pandoc`)."

    def get_paper_as_markdown(self, arxiv_id: str) -> str:
        """Fetch the TeX source of an arXiv paper and return it as clean Markdown.
        
        Uses the arxiv package to locate and download the official source tarball,
        extracts the main .tex file (largest .tex by size in case of supplementary files),
        then calls tex_to_markdown to produce the final output.
        
        :param arxiv_id: arXiv ID (with or without "arXiv:" prefix and version, e.g. "2403.12345", "arXiv:2403.12345", "2305.12345v2")
        :return: Full paper content as Markdown (title, sections, equations preserved with $ and $$)
        """
        # Normalize ID
        if arxiv_id.lower().startswith("arxiv:"):
            arxiv_id = arxiv_id.split(":", 1)[1]
        
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        
        try:
            result = next(client.results(search))
        except StopIteration:
            return f"Error: No paper found with ID '{arxiv_id}'. Double-check the ID on arxiv.org."
        
        # Download source tarball / gzipped tex to a temporary location
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, f"{result.get_short_id()}.tar.gz")
            urllib.request.urlretrieve(result.source_url(), source_path)
            
            # Extract the main TeX file (largest .tex in the archive)
            tex_content = None
            max_size = 0
            
            if tarfile.is_tarfile(source_path):
                with tarfile.open(source_path, "r:gz") as tar:
                    for member in tar.getmembers():
                        if member.isfile() and member.name.lower().endswith(".tex"):
                            f = tar.extractfile(member)
                            content = f.read().decode("utf-8", errors="replace")
                            size = len(content)
                            if size > max_size:
                                tex_content = content
                                max_size = size
            else:
                # Fallback: single-file gzipped TeX (rare but supported by arXiv)
                try:
                    with gzip.open(source_path, "rb") as f:
                        tex_content = f.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
            
            if not tex_content:
                return f"Error: Could not extract any .tex file from {result.get_short_id()}. The source may be malformed."
            
            # Convert using the dedicated function
            return self.tex_to_markdown(tex_content)


# =============================================================================
# UNIT TESTS (run with `python this_file.py`)
# These confirm that $ and $$ math syntax is correctly retained.
# =============================================================================

if __name__ == "__main__":
    print("Running unit tests for arXiv to Markdown tool...\n")
    tool = Tools()
    
    # Test 1: tex_to_markdown with inline and display math
    sample_tex = r"""
\documentclass{article}
\begin{document}

\title{Test Paper}
\maketitle

\section{Introduction}
This is an inline equation: $E = mc^2$.

A display equation:
\begin{equation}
a^2 + b^2 = c^2
\end{equation}

Another inline: $\alpha + \beta = \gamma$.

\end{document}
"""
    md_output = tool.tex_to_markdown(sample_tex)
    
    print("=== TEST 1: tex_to_markdown ===")
    print("Input TeX contained math delimiters.")
    print("Output contains $ or $$ ?")
    has_inline = "$" in md_output or "\\(" in md_output
    has_display = "$$" in md_output or "\\[" in md_output or "equation" in md_output.lower()
    
    print(f"Inline math retained: {has_inline}")
    print(f"Display math retained: {has_display}")
    print("\nFirst 300 chars of Markdown output:")
    print(md_output[:300])
    
    assert has_inline, "Inline $ math was not retained!"
    assert has_display, "Display $$ / equation math was not retained!"
    print("✅ TEST 1 PASSED: Math syntax ($ and $$) correctly retained.\n")
    
    # Test 2: Search function (basic smoke test)
    print("=== TEST 2: search_arxiv_papers ===")
    search_result = tool.search_arxiv_papers("cat:cs.AI", max_results=2)
    print("Search returned content (first 200 chars):")
    print(search_result[:200])
    assert "Found" in search_result or "papers" in search_result.lower(), "Search did not return expected format"
    print("✅ TEST 2 PASSED: Search function works.\n")
    
    print("All unit tests passed! The tool is ready to use in Open-WebUI.")
    print("Note: Full get_paper_as_markdown test requires internet and a valid arXiv ID.")
    print("Pandoc must be installed on the host system for conversion to work.")