"""
title: arXiv Paper to Markdown
author: Grok by xAI
author_url: https://x.ai
version: 1.0
license: MIT
description: Search arXiv for papers and convert any paper to clean Markdown. Prefers the original TeX source (preserving $ and $$ math exactly) when available; falls back to PDF parsing with PyMuPDF for older papers that only provide PDF. Includes unit tests.
requirements: arxiv requests pymupdf
"""

import arxiv
import requests
import io
import tarfile
import gzip
import re
import fitz  # pymupdf
from typing import Optional, List, Dict, Any
import unittest


class Tools:
    def __init__(self):
        # Enables citation display in Open WebUI when the tool is used
        self.citation = True

    def search_arxiv_papers(
        self, query: str, max_results: int = 10
    ) -> str:
        """
        Search arXiv and return a nicely formatted Markdown list of matching papers.

        :param query: Search query (supports arXiv syntax, e.g. "cat:cs.AI" or "large language models")
        :param max_results: Maximum number of results to return (default 10)
        :return: Markdown-formatted summary with title, authors, ID, abstract snippet, and PDF link
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers: List[str] = []
        for result in search.results():
            arxiv_id = result.get_short_id() or result.entry_id.split("/")[-1]
            authors = ", ".join(author.name for author in result.authors)
            abstract_snippet = result.summary[:400] + "..." if len(result.summary) > 400 else result.summary

            md = f"""**Title:** {result.title}
**Authors:** {authors}
**arXiv ID:** {arxiv_id}
**Published:** {result.published.strftime("%Y-%m-%d")}
**PDF:** [{result.pdf_url}]({result.pdf_url})

**Abstract:**
{abstract_snippet}

---
"""
            papers.append(md)
        if not papers:
            return "No papers found for the query."
        return "\n".join(papers)

    def get_paper_as_markdown(self, arxiv_id: str) -> str:
        """
        Convert a full arXiv paper to Markdown.
        1. First tries to fetch the original TeX source (most accurate math preservation).
        2. Falls back to PDF parsing with PyMuPDF for older papers that only have PDF.

        :param arxiv_id: arXiv identifier (e.g. "2305.10415", "2305.10415v2", or full "arXiv:2305.10415")
        :return: Complete Markdown version of the paper
        """
        # Normalize ID
        arxiv_id = arxiv_id.strip().replace("arXiv:", "").replace("https://arxiv.org/abs/", "").strip()

        # 1. Try TeX source first (most papers have it)
        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        try:
            resp = requests.get(source_url, timeout=30)
            if resp.status_code == 200:
                tex_content = self._extract_tex_from_source(resp.content)
                if tex_content:
                    return self._tex_to_markdown(tex_content)
        except Exception:
            pass  # source not available or network error → fallback

        # 2. Fallback to PDF (for very old/pre-LaTeX-upload papers)
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            resp = requests.get(pdf_url, timeout=30)
            resp.raise_for_status()
            return self._pdf_to_markdown(resp.content)
        except Exception as e:
            return f"❌ Could not retrieve paper {arxiv_id}:\n{str(e)}"

    def _extract_tex_from_source(self, source_bytes: bytes) -> Optional[str]:
        """Extract the main .tex content from a downloaded arXiv source tar.gz or .tex.gz."""
        # Try tar.gz (most common)
        try:
            with io.BytesIO(source_bytes) as f:
                with tarfile.open(fileobj=f, mode="r:gz") as tar:
                    for member in tar.getmembers():
                        if member.name.lower().endswith(".tex"):
                            tex_file = tar.extractfile(member)
                            if tex_file:
                                content = tex_file.read().decode("utf-8", errors="ignore")
                                # Prefer the file that contains \documentclass (main file)
                                if "\\documentclass" in content:
                                    return content
        except Exception:
            pass

        # Try single .tex.gz
        try:
            with io.BytesIO(source_bytes) as f:
                with gzip.GzipFile(fileobj=f) as gz:
                    content = gz.read().decode("utf-8", errors="ignore")
                    if content and "\\documentclass" in content:
                        return content
        except Exception:
            pass

        return None

    def _tex_to_markdown(self, tex: str) -> str:
        """Convert TeX source to Markdown while preserving all $ and $$ math exactly."""
        # Remove comments
        tex = re.sub(r"%.*$", "", tex, flags=re.MULTILINE)

        # Headings
        tex = re.sub(r"\\section\{(.*?)\}", r"# \1", tex)
        tex = re.sub(r"\\subsection\{(.*?)\}", r"## \1", tex)
        tex = re.sub(r"\\subsubsection\{(.*?)\}", r"### \1", tex)
        tex = re.sub(r"\\paragraph\{(.*?)\}", r"#### \1", tex)

        # Display math environments → $$ ... $$
        tex = re.sub(
            r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}",
            r"$$\1$$",
            tex,
            flags=re.DOTALL,
        )
        tex = re.sub(
            r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}",
            r"$$\1$$",
            tex,
            flags=re.DOTALL,
        )
        tex = re.sub(
            r"\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}",
            r"$$\1$$",
            tex,
            flags=re.DOTALL,
        )
        tex = re.sub(
            r"\\begin\{multline\*?\}(.*?)\\end\{multline\*?\}",
            r"$$\1$$",
            tex,
            flags=re.DOTALL,
        )

        # Inline math ($...$) is already correct – we leave it untouched

        # Basic formatting
        tex = re.sub(r"\\textbf\{(.*?)\}", r"**\1**", tex)
        tex = re.sub(r"\\textit\{(.*?)\}", r"*\1*", tex)
        tex = re.sub(r"\\emph\{(.*?)\}", r"*\1*", tex)

        # Lists
        tex = re.sub(
            r"\\begin\{itemize\}(.*?)\\end\{itemize\}",
            lambda m: "\n" + re.sub(r"\\item\s*", "- ", m.group(1).strip(), flags=re.DOTALL) + "\n",
            tex,
            flags=re.DOTALL,
        )
        tex = re.sub(
            r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}",
            lambda m: "\n" + re.sub(r"\\item\s*", "1. ", m.group(1).strip(), flags=re.DOTALL) + "\n",
            tex,
            flags=re.DOTALL,
        )

        # Remove preamble and post-document (keep only body content)
        start = tex.find("\\begin{document}")
        if start != -1:
            tex = tex[start + len("\\begin{document}") :]
        end = tex.find("\\end{document}")
        if end != -1:
            tex = tex[:end]

        # Clean up extra newlines and whitespace
        tex = re.sub(r"\n{3,}", "\n\n", tex)
        return tex.strip()

    def _pdf_to_markdown(self, pdf_bytes: bytes) -> str:
        """Fallback PDF → Markdown using PyMuPDF (text extraction)."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts: List[str] = []
        for i, page in enumerate(doc, 1):
            text = page.get_text("text")
            parts.append(f"### Page {i}\n\n{text}\n")
        doc.close()
        return "\n\n".join(parts)


# =============================================================================
# UNIT TESTS (run with: python this_file.py)
# =============================================================================
class TestArxivTool(unittest.TestCase):
    def test_search_arxiv_papers(self):
        """Confirm search returns properly formatted Markdown."""
        tool = Tools()
        result = tool.search_arxiv_papers("large language models", max_results=3)
        self.assertIn("**Title:**", result)
        self.assertIn("**arXiv ID:**", result)
        self.assertIn("---", result)

    def test_tex_equation_conversion(self):
        """Verify that $ and $$ math syntax is perfectly retained."""
        tool = Tools()
        sample_tex = r"""
\section{Test}
Inline math: $E=mc^2$

\begin{equation}
\int_0^\infty e^{-x} dx = 1
\end{equation}

\begin{align}
a + b &= c \\
d &= e
\end{align}
"""
        md = tool._tex_to_markdown(sample_tex)
        self.assertIn("$E=mc^2$", md)
        self.assertIn("$$\n\\int_0^\\infty e^{-x} dx = 1\n$$", md)
        self.assertIn("$$\na + b &= c", md)
        self.assertIn("# Test", md)

    def test_full_tex_markdown_conversion(self):
        """End-to-end TeX → Markdown (structure + math)."""
        tool = Tools()
        sample_tex = r"""
\documentclass{article}
\begin{document}
\title{Test Paper}
\section{Introduction}
This is a test. Equation: $x^2 + y^2 = z^2$.
\begin{equation}
\sum_{i=1}^n i = \frac{n(n+1)}{2}
\end{equation}
\end{document}
"""
        md = tool._tex_to_markdown(sample_tex)
        self.assertIn("# Introduction", md)
        self.assertIn("$x^2 + y^2 = z^2$", md)
        self.assertIn("$$", md)

    def test_pdf_fallback(self):
        """Test PDF parsing path (simulated bytes; real call would use a pre-LaTeX-era paper)."""
        tool = Tools()
        # Minimal valid PDF bytes that PyMuPDF can open (1-page dummy)
        dummy_pdf = (
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\n"
            b"BT /F1 24 Tf 100 700 Td (Test PDF content) Tj ET\nendstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000074 00000 n\n"
            b"0000000123 00000 n\n0000000200 00000 n\ntrailer\n<< /Size 5 /Root 1 0 R >>\n"
            b"startxref\n300\n%%EOF"
        )
        md = tool._pdf_to_markdown(dummy_pdf)
        self.assertIn("### Page 1", md)
        self.assertIn("Test PDF content", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)