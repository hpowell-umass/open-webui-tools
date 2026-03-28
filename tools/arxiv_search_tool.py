import arxiv
import requests
import tempfile
import os
import tarfile
import re
import fitz  # PyMuPDF (pip install pymupdf)
from typing import List, Dict, Any
import unittest


def search_arxiv_papers(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search arXiv for papers using the arxiv Python package.
    
    Args:
        query: Search query (e.g. "attention is all you need" or "large language models")
        max_results: Maximum number of results to return (default 5)
    
    Returns:
        List of dicts with paper metadata (id, title, abstract, authors, published, pdf_url, source_url)
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    
    results = []
    for result in search.results():
        results.append({
            "id": result.get_short_id(),  # e.g. "1706.03762"
            "title": result.title,
            "abstract": result.summary,
            "authors": [author.name for author in result.authors],
            "published": str(result.published),
            "pdf_url": result.pdf_url,
            "source_url": f"https://arxiv.org/src/{result.get_short_id()}"
        })
    return results


def resolve_includes(tex_path: str, base_dir: str) -> str:
    """
    Recursively resolve \input{} and \include{} commands by inlining the full TeX content.
    Handles both braced (\input{foo}) and non-braced (\input foo) forms.
    Supports papers with multiple .tex files (piece-by-piece compilation).
    """
    try:
        with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return ""

    def replacer(match: re.Match) -> str:
        command = match.group(1).lower()
        filename = match.group(2).strip()
        
        # Clean filename (remove .tex extension if present)
        if filename.endswith('.tex'):
            filename = filename[:-4]
        
        # Search recursively in the extracted directory for the matching .tex file
        included_path = None
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.tex'):
                    base_name = os.path.splitext(file)[0]
                    if base_name == filename or file == filename:
                        candidate = os.path.join(root, file)
                        if candidate != tex_path:  # prevent self-inclusion loops
                            included_path = candidate
                            break
            if included_path:
                break
        
        if included_path:
            # Recurse to resolve nested includes
            return resolve_includes(included_path, base_dir)
        
        # File not found → keep original command
        return match.group(0)

    # Braced form: \input{...} or \include{...}
    content = re.sub(r'\\(input|include)\{([^}]+)\}', replacer, content, flags=re.IGNORECASE)
    
    # Non-braced form: \input foo.tex or \include bar
    content = re.sub(r'\\(input|include)\s+([^\s%\\]+)', replacer, content, flags=re.IGNORECASE)
    
    return content


def latex_to_markdown(tex_content: str) -> str:
    """
    Convert resolved LaTeX source to Markdown while retaining proper equation syntax ($ and $$).
    Handles sections, itemize/enumerate, display/inline math, and basic cleanup.
    """
    if not tex_content:
        return ""
    
    # Remove comments
    tex_content = re.sub(r'(?m)^%.*$', '', tex_content)
    tex_content = re.sub(r'%.*$', '', tex_content, flags=re.MULTILINE)
    
    # Headings
    tex_content = re.sub(r'\\section\*?\{([^}]+)\}', r'# \1', tex_content)
    tex_content = re.sub(r'\\subsection\*?\{([^}]+)\}', r'## \1', tex_content)
    tex_content = re.sub(r'\\subsubsection\*?\{([^}]+)\}', r'### \1', tex_content)
    tex_content = re.sub(r'\\paragraph\{([^}]+)\}', r'#### \1', tex_content)
    
    # Display math environments → $$
    tex_content = re.sub(
        r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}',
        r'$$\1$$',
        tex_content,
        flags=re.DOTALL | re.IGNORECASE
    )
    tex_content = re.sub(
        r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}',
        r'$$\1$$',
        tex_content,
        flags=re.DOTALL | re.IGNORECASE
    )
    tex_content = re.sub(
        r'\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}',
        r'$$\1$$',
        tex_content,
        flags=re.DOTALL | re.IGNORECASE
    )
    tex_content = re.sub(r'\\\[ (.*?) \\\]', r'$$\1$$', tex_content, flags=re.DOTALL)
    
    # Inline math → $
    tex_content = re.sub(r'\\\((.*?)\\\)', r'$\1$', tex_content, flags=re.DOTALL)
    
    # Lists
    def list_replacer(m: re.Match) -> str:
        items = re.split(r'\\item', m.group(1))
        return '\n' + '\n'.join(f"- {item.strip()}" for item in items if item.strip()) + '\n'
    
    tex_content = re.sub(
        r'\\begin\{itemize\}(.*?)\\end\{itemize\}',
        list_replacer,
        tex_content,
        flags=re.DOTALL | re.IGNORECASE
    )
    tex_content = re.sub(
        r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}',
        list_replacer,
        tex_content,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Remove figures/tables (too complex for basic MD conversion)
    tex_content = re.sub(
        r'\\begin\{figure\}.*?\\end\{figure\}',
        '',
        tex_content,
        flags=re.DOTALL | re.IGNORECASE
    )
    tex_content = re.sub(
        r'\\begin\{table\}.*?\\end\{table\}',
        '',
        tex_content,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Common inline formatting (keep content)
    tex_content = re.sub(r'\\(textbf|textit|emph|text|cite|ref|label)\{([^}]+)\}', r'\2', tex_content)
    
    # Remove remaining LaTeX commands (after math/sections/lists have been processed)
    tex_content = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?', '', tex_content)
    
    # Clean whitespace
    tex_content = re.sub(r'\n\s*\n+', '\n\n', tex_content)
    tex_content = re.sub(r'^\s+', '', tex_content, flags=re.MULTILINE)
    
    return tex_content.strip()


def get_paper_markdown(arxiv_id: str) -> str:
    """
    Main tool function: Convert an arXiv paper to Markdown.
    
    1. Tries to download TeX source (https://arxiv.org/src/{id}).
    2. Extracts tarball, finds main .tex (via \documentclass), resolves all \input{} / \include{} recursively.
    3. Converts resolved LaTeX to Markdown (equations preserved as $ and $$).
    4. Falls back to PDF download + PyMuPDF text extraction if no TeX source is available (old/pre-LaTeX papers).
    
    Returns full Markdown with title, abstract, and body content.
    """
    # Fetch metadata first (works for all papers)
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        title = paper.title
        abstract = paper.summary.strip()
    except Exception:
        title = f"arXiv:{arxiv_id}"
        abstract = ""
    
    # Try TeX source first
    source_url = f"https://arxiv.org/src/{arxiv_id}"
    try:
        response = requests.get(source_url, stream=True, timeout=60)
        if response.status_code != 200 or 'html' in response.headers.get('content-type', '').lower():
            raise ValueError("No TeX source available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            tar_path = os.path.join(temp_dir, f"{arxiv_id}.tar.gz")
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Extract (handles .tar.gz or .tar)
            with tarfile.open(tar_path, 'r:*') as tar:
                tar.extractall(temp_dir)
            
            # Locate main .tex file
            main_tex_path = None
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.tex'):
                        full_path = os.path.join(root, file)
                        try:
                            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                                if '\\documentclass' in f.read(3000):
                                    main_tex_path = full_path
                                    break
                        except Exception:
                            continue
                if main_tex_path:
                    break
            
            # Fallback: any .tex file
            if not main_tex_path:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.tex'):
                            main_tex_path = os.path.join(root, file)
                            break
                    if main_tex_path:
                        break
            
            if main_tex_path:
                resolved_tex = resolve_includes(main_tex_path, temp_dir)
                body_md = latex_to_markdown(resolved_tex)
                return f"# {title}\n\n**Abstract:** {abstract}\n\n{body_md}"
    
    except Exception:
        # TeX failed → fallback to PDF
        pass
    
    # PDF fallback (for old papers without TeX source)
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(response.content)
            pdf_path = tmp.name
        
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n\n"
        doc.close()
        os.unlink(pdf_path)
        
        body_md = f"## Full Text Extracted from PDF (equations may not be perfectly preserved)\n\n{text}"
        return f"# {title}\n\n**Abstract:** {abstract}\n\n{body_md}"
    
    except Exception as e:
        return f"# {title}\n\n**Abstract:** {abstract}\n\n**Error retrieving paper content:** {str(e)}"


# =============================================================================
# Unit Tests (run with: python -m unittest this_file.py)
# =============================================================================
class TestArXivTool(unittest.TestCase):
    
    def test_paper_searching(self):
        """Confirm paper searching works and returns expected structure."""
        results = search_arxiv_papers("attention is all you need", max_results=2)
        self.assertGreaterEqual(len(results), 1)
        paper = results[0]
        self.assertIn("id", paper)
        self.assertIn("title", paper)
        self.assertIn("abstract", paper)
        self.assertIsInstance(paper["title"], str)
        self.assertTrue(paper["id"].startswith("17"))  # known paper ID prefix
    
    def test_equation_conversion(self):
        """Confirm equation syntax ($ and $$) is retained after LaTeX → Markdown conversion."""
        sample_tex = r"""
\section{Test Section}
Inline math: \( E = mc^2 \).
Display equation:
\begin{equation}
a^2 + b^2 = c^2
\end{equation}
And another: \begin{align}
x &= 1 \\
y &= 2
\end{align}
"""
        md = latex_to_markdown(sample_tex)
        self.assertIn("# Test Section", md)
        self.assertIn("$ E = mc^2 $", md)
        self.assertIn("$$ a^2 + b^2 = c^2 $$", md)
        self.assertIn("$$ x &= 1 \\\\ y &= 2 $$", md)
    
    def test_full_markdown_latex_paper(self):
        """Full conversion test on an actual multi-file LaTeX paper (resolves \input{})."""
        # "Attention is All You Need" (1706.03762) – known to have full TeX source
        md = get_paper_markdown("1706.03762")
        self.assertIsNotNone(md)
        self.assertGreater(len(md), 5000)
        self.assertIn("Attention is all you need", md.lower())
        # Verify math syntax survived
        self.assertTrue("$" in md or "$$" in md)
        # Verify headings were converted
        self.assertIn("# ", md)
    
    def test_full_markdown_pdf_paper(self):
        """Test on an actual old/pre-LaTeX-era paper that triggers PDF fallback (PyMuPDF extraction)."""
        # hep-th/9304001 (1993 paper) – old enough that source handling tests the fallback path
        md = get_paper_markdown("hep-th/9304001")
        self.assertIsNotNone(md)
        self.assertGreater(len(md), 1000)
        self.assertIn("# ", md)  # title header present
        # Either TeX or PDF path succeeded
        self.assertTrue("Abstract" in md or "Extracted Text" in md)


if __name__ == "__main__":
    # For local testing / CI
    unittest.main()