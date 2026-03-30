import arxiv
import tempfile
import tarfile
import os
import re
import gzip
from pathlib import Path
import fitz  # PyMuPDF
import unittest
from typing import List, Dict, Optional


# ==================== INTERNAL HELPERS ====================

def flatten_latex(tex_dir: str, main_tex_filename: str) -> str:
    """
    Recursively inline all \input{} and \include{} statements to produce a single flat LaTeX string.
    Handles nested includes piece-by-piece as required.
    """
    def read_file(file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

    def inline_inputs(content: str, base_dir: str) -> str:
        def replace_input(match: re.Match) -> str:
            filename = match.group(1).strip()
            if not filename.endswith(".tex"):
                filename += ".tex"
            file_path = os.path.join(base_dir, filename)
            if os.path.exists(file_path):
                included = read_file(file_path)
                # Recurse for nested \input / \include
                return inline_inputs(included, base_dir)
            return match.group(0)  # leave unchanged if file missing

        pattern = r"\\(?:input|include)\s*\{([^}]+)\}"
        return re.sub(pattern, replace_input, content, flags=re.IGNORECASE)

    main_path = os.path.join(tex_dir, main_tex_filename)
    full_content = read_file(main_path)
    return inline_inputs(full_content, tex_dir)


def latex_to_markdown(tex: str) -> str:
    """
    Convert flattened LaTeX to Markdown.
    - Sections become ## / ### headings.
    - All math environments ($...$, $$...$$, \[...\], equation, align, etc.) are preserved exactly
      with matching $ or $$ delimiters (the original LaTeX syntax is kept untouched).
    - Common text formatting is converted to Markdown.
    - Preamble, document envs, and non-math LaTeX commands are stripped or simplified.
    """
    # Step 1: Convert structural commands (before math protection)
    tex = re.sub(r"\\section\*?\{([^}]+)\}", r"## \1", tex)
    tex = re.sub(r"\\subsection\*?\{([^}]+)\}", r"### \1", tex)
    tex = re.sub(r"\\subsubsection\*?\{([^}]+)\}", r"#### \1", tex)
    tex = re.sub(r"\\paragraph\{([^}]+)\}", r"**\1**", tex)
    tex = re.sub(r"\\begin\{abstract\}", "## Abstract\n", tex)
    tex = re.sub(r"\\end\{abstract\}", "", tex)

    # Step 2: Protect ALL math blocks with placeholders so they are never stripped
    math_blocks: List[str] = []

    def store_math(match: re.Match) -> str:
        idx = len(math_blocks)
        # Store the ORIGINAL math syntax (guarantees matching $ or $$)
        math_blocks.append(match.group(0))
        return f"__MATH_BLOCK_{idx}__"

    # Display math (various forms)
    tex = re.sub(r"\\\[(.*?)\]", store_math, tex, flags=re.DOTALL)
    tex = re.sub(r"\$\$(.*?)\$\$", store_math, tex, flags=re.DOTALL)
    tex = re.sub(r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", store_math, tex, flags=re.DOTALL)
    tex = re.sub(r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}", store_math, tex, flags=re.DOTALL)
    tex = re.sub(r"\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}", store_math, tex, flags=re.DOTALL)

    # Inline math $...$ (non-greedy, avoids escaped or double $)
    tex = re.sub(r"(?<!\\)\$([^\$]+?)\$(?!\$)", store_math, tex)

    # Step 3: Strip / simplify remaining LaTeX (math blocks are safe)
    tex = re.sub(r"\\emph\{([^}]+)\}", r"*\1*", tex)
    tex = re.sub(r"\\textbf\{([^}]+)\}", r"**\1**", tex)
    tex = re.sub(r"\\textit\{([^}]+)\}", r"*\1*", tex)
    tex = re.sub(r"\\([a-zA-Z]+)\{([^}]*)\}", r"\2", tex)  # general command -> content
    tex = re.sub(r"\\[a-zA-Z]+(\s+|$)", " ", tex)  # commands without braces
    tex = re.sub(r"\\begin\{[^}]+?\}(.*?)\\end\{[^}]+?\}", r"\1", tex, flags=re.DOTALL)  # other envs

    # Step 4: Cleanup whitespace
    tex = re.sub(r"\s{2,}", " ", tex)
    tex = tex.replace("\\", "")  # final backslashes

    # Step 5: Restore math blocks (original syntax guaranteed to have matching $ / $$)
    for i, block in enumerate(math_blocks):
        tex = tex.replace(f"__MATH_BLOCK_{i}__", block)

    # Step 6: Remove any remaining preamble
    lines = tex.split("\n")
    cleaned: List[str] = []
    skip_preamble = True
    for line in lines:
        stripped = line.strip()
        if any(kw in stripped for kw in ["\\documentclass", "\\usepackage", "\\title{", "\\author{", "\\date{"]):
            continue
        if "\\begin{document}" in stripped:
            skip_preamble = False
            continue
        if not skip_preamble:
            cleaned.append(line)
    tex = "\n".join(cleaned).strip()

    return tex


def pdf_to_markdown(pdf_path: str) -> str:
    """
    Fallback parser for papers without usable TeX source (old/pre-LaTeX PDFs).
    Uses PyMuPDF to extract as much structured text as possible.
    Equations appear as rendered text/symbols (no $ delimiters possible from PDF).
    """
    try:
        doc = fitz.open(pdf_path)
        markdown_parts: List[str] = []
        for page_num, page in enumerate(doc):
            # Extract plain text (best effort for arXiv PDFs)
            text = page.get_text("text")
            markdown_parts.append(f"### Page {page_num + 1}\n\n{text.strip()}\n\n")
        doc.close()
        return "".join(markdown_parts).strip()
    except Exception as e:
        return f"PDF parsing failed: {str(e)}"


# ==================== PUBLIC TOOL FUNCTIONS (for Open-WebUI) ====================

def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search arXiv for papers using the official arxiv Python package.

    :param query: Search query (supports arXiv syntax, e.g. "cat:cs.AI" or title keywords)
    :param max_results: Maximum number of results to return (default 5)
    :return: List of paper metadata dictionaries
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results: List[Dict] = []
    for result in client.results(search):
        results.append({
            "id": result.get_short_id(),  # e.g. 2403.12345v1
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "published": str(result.published),
            "pdf_url": result.pdf_url,
            "arxiv_url": result.entry_id,
        })
    return results


def convert_paper_to_markdown(arxiv_id: str) -> str:
    """
    Main tool function: Convert an arXiv paper to Markdown.
    1. Attempts to download TeX source (handles single .tex or multi-file .tar.gz with \input{}).
    2. Flattens multi-file TeX piece-by-piece via recursive inlining.
    3. Converts flattened TeX → Markdown while guaranteeing proper matching $ / $$ math syntax.
    4. Falls back to PDF parsing with PyMuPDF for old papers without TeX source.
    Returns the full Markdown string ready for the model.

    :param arxiv_id: arXiv ID (e.g. "1706.03762" or "2403.12345")
    :return: Markdown content of the paper
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        result = next(client.results(search), None)
        if not result:
            return f"Paper {arxiv_id} not found on arXiv."

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Try TeX source first
            try:
                source_path = result.download_source(dirpath=tmp_dir)
                extracted_tex = False

                # Handle various source formats
                if tarfile.is_tarfile(source_path):
                    with tarfile.open(source_path) as tar:
                        tar.extractall(path=tmp_dir)
                    extracted_tex = True
                elif source_path.endswith(".gz"):
                    # Single-file .tex.gz
                    tex_name = Path(source_path).name.replace(".gz", "")
                    with gzip.open(source_path, "rb") as f_in:
                        with open(os.path.join(tmp_dir, tex_name), "wb") as f_out:
                            f_out.write(f_in.read())
                    extracted_tex = True

                if extracted_tex:
                    # Locate main .tex (prefer file containing \documentclass)
                    tex_files = list(Path(tmp_dir).rglob("*.tex"))
                    if not tex_files:
                        raise ValueError("No .tex files found in source")

                    main_tex = None
                    for tf in tex_files:
                        with open(tf, "r", encoding="utf-8", errors="ignore") as f:
                            if "\\documentclass" in f.read(2000):  # peek first 2k chars
                                main_tex = tf.name
                                break
                    if not main_tex:
                        # Fallback: largest file
                        main_tex = max(tex_files, key=lambda p: p.stat().st_size).name

                    flat_tex = flatten_latex(tmp_dir, main_tex)
                    return latex_to_markdown(flat_tex)

            except Exception as source_err:
                # TeX failed → fallback to PDF (covers pre-LaTeX/old papers)
                pdf_path = result.download_pdf(dirpath=tmp_dir)
                return pdf_to_markdown(pdf_path)

    except Exception as e:
        return f"Error processing paper {arxiv_id}: {str(e)}"


# ==================== UNIT TESTS ====================

class TestArxivTool(unittest.TestCase):
    """Unit tests for paper searching, equation conversion, full Markdown conversion (LaTeX + PDF), and error handling."""

    def test_paper_searching(self):
        """Confirm search_arxiv returns valid results."""
        results = search_arxiv("large language models", max_results=2)
        self.assertGreater(len(results), 0)
        paper = results[0]
        self.assertIn("id", paper)
        self.assertIn("title", paper)
        self.assertIn("abstract", paper)
        self.assertIsInstance(paper["authors"], list)

    def test_equation_conversion(self):
        """Test that math syntax is preserved with exact matching $ / $$ delimiters."""
        sample_tex = r"""
\section{Test Section}
Text with inline math $E=mc^2$ and another $x^2+y^2=z^2$.
Display equation: $$\sum_{i=1}^n i = \frac{n(n+1)}{2}$$
And bracketed: \[a^2 + b^2 = c^2\]
Mixed text with equation: The speed of light is $c \approx 3 \times 10^8$ m/s.
"""
        md = latex_to_markdown(sample_tex)
        self.assertIn("## Test Section", md)
        self.assertIn("$E=mc^2$", md)
        self.assertIn("$x^2+y^2=z^2$", md)
        self.assertIn("$$\\sum_{i=1}^n i = \\frac{n(n+1)}{2}$$", md)
        self.assertIn("\\[a^2 + b^2 = c^2\\]", md)
        self.assertIn("$c \\approx 3 \\times 10^8$", md)

    def test_full_markdown_conversion_latex_paper(self):
        """Full conversion test on a real modern LaTeX paper (multi-file capable)."""
        paper_id = "1706.03762"  # "Attention Is All You Need" – has full TeX source
        md = convert_paper_to_markdown(paper_id)
        self.assertNotIn("Error", md)
        self.assertGreater(len(md), 2000)  # substantial content
        self.assertIn("## ", md)  # headings from sections
        # Math should be present with proper delimiters
        self.assertTrue(any(delim in md for delim in ["$", "$$", "\\["]))

    def test_pre_latex_pdf_paper(self):
        """Test PDF fallback path on a real paper (simulates pre-LaTeX / source-unavailable case)."""
        paper_id = "1706.03762"  # reuse known paper; PDF path is always exercised in fallback logic
        # Directly exercise PDF parser for isolation
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])
        result = next(client.results(search))
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = result.download_pdf(dirpath=tmp)
            md = pdf_to_markdown(pdf_path)
        self.assertGreater(len(md), 1000)
        self.assertIn("Page 1", md)  # structure from PDF parser

    def test_non_existent_paper(self):
        """Test graceful error handling for a non-existent paper ID."""
        md = convert_paper_to_markdown("9999.99999")
        self.assertIn("not found", md.lower()) or self.assertIn("Error", md)


if __name__ == "__main__":
    unittest.main()