"""
Open-WebUI Tool: arXiv Paper Search & Markdown Converter

This is a complete, self-contained tool for Open-WebUI.
Place this file in your Open-WebUI tools directory (usually ./tools/)
or upload it via the UI.

Dependencies (install via pip in your Open-WebUI environment):
    pip install arxiv pymupdf requests

Optional but strongly recommended for perfect LaTeX → Markdown conversion
(with $ and $$ equation syntax preserved):
    apt-get install pandoc   # or equivalent in your container

The tool exposes two functions:
- search_arxiv: Search arXiv and return formatted results.
- get_paper_markdown: Fetch a paper by arXiv ID → full Markdown (LaTeX preferred,
  multi-file \input{} resolved, equations kept as $ / $$. Falls back to PDF
  text extraction for ancient papers without source).

Unit tests are included at the bottom and can be run with:
    python this_file.py
"""

import arxiv
import requests
import tempfile
import os
import re
import tarfile
import io
import subprocess
import unittest
import fitz  # PyMuPDF
from typing import Dict, Optional


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Normalize arXiv ID to the canonical form (e.g. 1706.03762)."""
    arxiv_id = arxiv_id.strip()
    if "arxiv.org" in arxiv_id.lower():
        arxiv_id = arxiv_id.split("/")[-1]
    if arxiv_id.lower().startswith("arxiv:"):
        arxiv_id = arxiv_id[6:]
    return arxiv_id


def resolve_inputs(tex_content: str, file_dict: Dict[str, str]) -> str:
    """
    Recursively resolve all \input{...} and \include{...} statements.
    Works for both flat and simple subdirectory structures (common in arXiv tarballs).
    """
    def replacer(match: re.Match) -> str:
        arg = match.group(1).strip().strip('"\'')
        # Try exact match first, then basename + .tex
        candidates = [
            arg,
            arg + ".tex" if not arg.endswith(".tex") else arg,
            os.path.basename(arg),
            os.path.basename(arg) + ".tex" if not os.path.basename(arg).endswith(".tex") else os.path.basename(arg),
        ]
        for cand in candidates:
            if cand in file_dict:
                included = file_dict[cand]
                # Recurse
                return resolve_inputs(included, file_dict)
        # Not found → leave original (rare, but safe)
        return match.group(0)

    # Match both \input and \include
    pattern = r'\\(?:input|include)\s*\{([^}]+)\}'
    # Recursion is safe (max depth in real papers is < 10)
    return re.sub(pattern, replacer, tex_content)


def tex_to_markdown(tex_str: str) -> str:
    """
    Convert full LaTeX string to Markdown using pandoc (preferred) with
    $ / $$ equation syntax. Falls back gracefully if pandoc is missing.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False, encoding="utf-8") as f:
        f.write(tex_str)
        tex_path = f.name

    try:
        # pandoc command that reliably produces Markdown with math
        cmd = [
            "pandoc",
            tex_path,
            "-f", "latex",
            "-t", "markdown",
            "--standalone",
            "--wrap=none",
        ]
        md_output = subprocess.check_output(cmd, text=True, timeout=60, stderr=subprocess.STDOUT)
        # Standardize display math to $$...$$ (pandoc sometimes emits \[ \])
        md_output = md_output.replace(r"\[", "$$").replace(r"\]", "$$")
        return md_output
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        # Graceful fallback when pandoc is not installed
        return f"**⚠️ pandoc not available – returning raw LaTeX (equations preserved as-is).**\n\n{tex_str[:20000]}...\n\n(Pandoc error: {str(e)})"
    finally:
        if os.path.exists(tex_path):
            os.unlink(tex_path)


def get_paper_markdown(arxiv_id: str) -> str:
    """
    Main tool function.
    Returns the full paper as clean Markdown.
    - Prefers LaTeX source (single or multi-file with \input{}).
    - Retains perfect $ / $$ equation syntax.
    - Falls back to PDF text extraction for very old papers without source.
    """
    arxiv_id = normalize_arxiv_id(arxiv_id)

    # 1. Try LaTeX source first (most papers)
    src_url = f"https://arxiv.org/src/{arxiv_id}"
    try:
        r = requests.get(src_url, timeout=30, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 2000:  # arbitrary threshold to catch empty responses
            content = r.content
            content_type = r.headers.get("Content-Type", "").lower()
            content_disp = r.headers.get("Content-Disposition", "").lower()

            file_dict: Dict[str, str] = {}

            # Detect tarball vs single .tex
            if ("tar" in content_type or content_disp[:-1].endswith((".tar.gz", ".tgz")) or
                src_url.endswith((".tar.gz", ".tgz"))):
                tar = tarfile.open(fileobj=io.BytesIO(content), mode="r:gz")
                for member in tar.getmembers():
                    if member.isfile() and member.name.lower().endswith((".tex", ".sty", ".cls")):
                        try:
                            with tar.extractfile(member) as f:
                                file_dict[os.path.basename(member.name)] = f.read().decode("utf-8", errors="replace")
                        except Exception:
                            continue
            else:
                # Single file (sometimes gzipped)
                if content[:2] == b"\x1f\x8b":  # gzip magic
                    import gzip
                    content = gzip.decompress(content)
                file_dict["main.tex"] = content.decode("utf-8", errors="replace")

            # Find main .tex (contains \documentclass or old \documentstyle)
            main_content = None
            for fname, cont in file_dict.items():
                if r"\\documentclass" in cont or r"\\documentstyle" in cont:
                    main_content = cont
                    break
            if main_content is None:
                # Fallback: longest .tex file
                tex_files = [c for f, c in file_dict.items() if f.lower().endswith(".tex")]
                if tex_files:
                    main_content = max(tex_files, key=len)

            if main_content:
                full_tex = resolve_inputs(main_content, file_dict)
                return tex_to_markdown(full_tex)

    except Exception:
        # Any network/processing error → fall through to PDF
        pass

    # 2. PDF fallback (old papers or source unavailable)
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        r = requests.get(pdf_url, timeout=30)
        if r.status_code != 200:
            return f"❌ Paper {arxiv_id} not found (PDF download failed)."

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(r.content)
            pdf_path = f.name

        doc = fitz.open(pdf_path)
        markdown_parts = []
        for i, page in enumerate(doc):
            # Best-effort text extraction (preserves layout reasonably well)
            text = page.get_text("text")
            if text.strip():
                markdown_parts.append(f"### Page {i+1}\n\n{text.strip()}\n\n")
        doc.close()
        os.unlink(pdf_path)

        if not markdown_parts:
            return f"❌ Paper {arxiv_id} is an image-only PDF (no extractable text)."

        return f"# Paper {arxiv_id} (PDF fallback – equations approximated as text)\n\n" + "".join(markdown_parts)

    except Exception as e:
        return f"❌ Failed to retrieve paper {arxiv_id}:\n{str(e)}"


def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search arXiv and return a nicely formatted Markdown list of results.
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    results = []
    for result in search.results():
        short_id = result.get_short_id() or result.entry_id.split("/")[-1]
        authors = ", ".join([str(a) for a in result.authors])
        abstract_snippet = result.summary[:400] + "..." if len(result.summary) > 400 else result.summary

        results.append(
            f"**[{result.title}]({result.entry_id})**  \n"
            f"**arXiv:** {short_id}  \n"
            f"**Authors:** {authors}  \n"
            f"**Published:** {result.published.strftime('%Y-%m-%d') if result.published else 'N/A'}  \n"
            f"**Abstract:** {abstract_snippet}  \n"
            f"[PDF]({result.pdf_url}) | [Source](https://arxiv.org/src/{short_id})  \n"
            "---"
        )

    if not results:
        return f"No results found for query: **{query}**"

    return f"# arXiv Search Results for “{query}”\n\n" + "\n\n".join(results)


# ====================== UNIT TESTS ======================

class TestArxivTool(unittest.TestCase):

    def test_search(self):
        """Confirm paper searching works with a well-known paper."""
        result = search_arxiv("Attention Is All You Need", max_results=1)
        self.assertIn("Attention Is All You Need", result)
        self.assertIn("1706.03762", result)

    def test_equation_conversion_inline(self):
        """Test that inline $...$ syntax is retained."""
        sample = r"The energy is given by $E = mc^2$."
        md = tex_to_markdown(sample)
        self.assertIn("$E = mc^2$", md)

    def test_equation_conversion_block(self):
        """Test that block equations become $$...$$."""
        sample = r"""
\begin{equation}
a^2 + b^2 = c^2
\end{equation}
"""
        md = tex_to_markdown(sample)
        self.assertIn("$$", md)
        self.assertIn("a^2 + b^2 = c^2", md)

    def test_full_markdown_conversion_actual_latex_paper(self):
        """Full end-to-end test on a real multi-file LaTeX paper (Attention Is All You Need)."""
        result = get_paper_markdown("1706.03762")
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 1000)
        # Should contain markdown headers and equations
        self.assertIn("# ", result)  # at least one section header
        self.assertTrue("$" in result or "$$" in result)

    def test_pre_latex_pdf_paper(self):
        """
        Test PDF fallback path with a real (very old) paper.
        Note: Most early arXiv papers still have source, but the PDF branch is exercised
        if source download fails or is empty. This confirms parsing works.
        """
        # Use an early arXiv paper (1992) – will likely hit PDF fallback in some environments
        result = get_paper_markdown("hep-th/9201001")
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 500)
        # PDF fallback always starts with a header
        self.assertIn("### Page", result)

    def test_non_existent_paper(self):
        """Confirm graceful error for a non-existent paper."""
        result = get_paper_markdown("9999.99999")
        self.assertIn("❌", result)
        self.assertIn("not found", result.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)