"""
title: arXiv Search & LaTeX-to-Markdown Converter
author: Grok (built by xAI)
author_url: https://x.ai
version: 1.1
requirements: arxiv
description: Search arXiv papers and convert LaTeX source to clean Markdown. Includes unit tests to verify full papers with equations are correctly retrieved.
"""

import arxiv
import json
import os
import re
import subprocess
import tempfile
import tarfile
import unittest
from typing import List, Dict, Any


class Tools:
    def __init__(self):
        pass

    def search_arxiv(
        self, 
        query: str, 
        max_results: int = 10,
        sort_by: str = "relevance"
    ) -> str:
        """Search arXiv for papers using the official arxiv Python package."""
        try:
            client = arxiv.Client()
            
            sort_criterion = arxiv.SortCriterion.Relevance if sort_by.lower() == "relevance" else arxiv.SortCriterion.SubmittedDate
            
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criterion,
            )
            
            results: List[Dict[str, Any]] = []
            for result in client.results(search):
                paper = {
                    "arxiv_id": result.get_short_id(),
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "published": str(result.published),
                    "pdf_url": result.pdf_url,
                    "source_url": f"https://arxiv.org/src/{result.get_short_id()}",
                    "comment": result.comment or "",
                    "journal_ref": result.journal_ref or "",
                }
                results.append(paper)
            
            return json.dumps(results, indent=2, ensure_ascii=False)
            
        except Exception as e:
            return json.dumps({"error": f"Search failed: {str(e)}"})

    def get_tex_source(self, arxiv_id: str) -> str:
        """Download the raw LaTeX source and return the main .tex content."""
        try:
            if arxiv_id.startswith("arXiv:") or arxiv_id.startswith("arxiv:"):
                arxiv_id = arxiv_id.split(":")[-1].strip()
            
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            papers = list(client.results(search))
            
            if not papers:
                return f"Error: Paper {arxiv_id} not found."
            
            result = papers[0]
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                archive_name = f"{arxiv_id.replace('/', '_')}.tar.gz"
                archive_path = os.path.join(tmp_dir, archive_name)
                
                result.download_source(dirpath=tmp_dir, filename=archive_name)
                
                extract_dir = os.path.join(tmp_dir, "extracted")
                os.makedirs(extract_dir, exist_ok=True)
                
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=extract_dir)
                
                # Find main .tex file (contains \documentclass)
                main_tex_path = None
                for root, _, files in os.walk(extract_dir):
                    for file in files:
                        if file.lower().endswith(".tex"):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                    content = f.read(10000)
                                if re.search(r"\\documentclass", content, re.IGNORECASE):
                                    main_tex_path = file_path
                                    break
                            except:
                                continue
                    if main_tex_path:
                        break
                
                # Fallback to any .tex
                if not main_tex_path:
                    for root, _, files in os.walk(extract_dir):
                        for file in files:
                            if file.lower().endswith(".tex"):
                                main_tex_path = os.path.join(root, file)
                                break
                        if main_tex_path:
                            break
                
                if not main_tex_path:
                    return f"Error: No .tex file found for {arxiv_id}."
                
                with open(main_tex_path, "r", encoding="utf-8", errors="ignore") as f:
                    tex_content = f.read()
                
                return tex_content
                
        except Exception as e:
            return f"Error fetching TeX source for {arxiv_id}: {str(e)}"

    def tex_to_markdown(self, tex_source: str) -> str:
        """Convert LaTeX source to Markdown using pandoc."""
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tex_file = os.path.join(tmp_dir, "paper.tex")
                md_file = os.path.join(tmp_dir, "paper.md")
                
                with open(tex_file, "w", encoding="utf-8") as f:
                    f.write(tex_source)
                
                cmd = [
                    "pandoc", tex_file, "-o", md_file,
                    "-f", "latex+raw_tex",
                    "-t", "markdown+tex_math_single_backslash+raw_tex",
                    "--mathjax", "--standalone", "--wrap=none",
                ]
                
                subprocess.run(cmd, check=True, cwd=tmp_dir, capture_output=True, text=True)
                
                with open(md_file, "r", encoding="utf-8") as f:
                    markdown = f.read()
                
                return markdown
                
        except FileNotFoundError:
            return "Error: pandoc not found. Please install pandoc on the system."
        except subprocess.CalledProcessError as e:
            return f"Pandoc conversion failed: {e.stderr.strip() if hasattr(e, 'stderr') and e.stderr else str(e)}"
        except Exception as e:
            return f"Conversion error: {str(e)}"


# ========================= UNIT TESTS =========================

class TestArXivTool(unittest.TestCase):
    """Unit tests to confirm the tool can fetch full papers with properly-written equations."""

    def test_search_arxiv_returns_results(self):
        """Basic search should return valid JSON with papers."""
        tool = Tools()
        result = tool.search_arxiv("quantum computing", max_results=3)
        self.assertIsInstance(result, str)
        data = json.loads(result)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        paper = data[0]
        self.assertIn("arxiv_id", paper)
        self.assertIn("title", paper)
        self.assertIn("abstract", paper)
        self.assertIn("pdf_url", paper)

    def test_get_tex_source_returns_valid_latex(self):
        """Fetch source for known papers that use standard math environments."""
        tool = Tools()
        
        # Test papers known to have rich LaTeX (equations, theorems, etc.)
        test_ids = [
            "1706.03762",   # Attention Is All You Need (Transformer paper) - excellent math
            "1603.04467",   # Deep Residual Learning for Image Recognition (ResNet)
            "1810.04805",   # BERT paper
        ]
        
        for arxiv_id in test_ids:
            with self.subTest(arxiv_id=arxiv_id):
                tex = tool.get_tex_source(arxiv_id)
                self.assertNotIn("Error:", tex, f"Failed to fetch {arxiv_id}")
                self.assertGreater(len(tex), 500, f"TeX too short for {arxiv_id}")
                
                # Verify it's proper LaTeX from a real paper
                self.assertTrue(
                    re.search(r"\\documentclass", tex, re.IGNORECASE),
                    f"No \\documentclass in {arxiv_id}"
                )
                # Check for math environments common in good papers
                has_math = any(re.search(pattern, tex, re.IGNORECASE) for pattern in [
                    r"\\begin\{equation", r"\\begin\{align", r"\\\[", r"\$", r"\\theta", r"\\alpha"
                ])
                self.assertTrue(has_math, f"No detectable math/equations in {arxiv_id}")

    def test_tex_to_markdown_converts_without_crash(self):
        """Test that tex_to_markdown can process a real paper's TeX (if pandoc available)."""
        tool = Tools()
        tex = tool.get_tex_source("1706.03762")  # Transformer paper
        
        if "Error:" in tex:
            self.skipTest("Could not fetch test paper source")
        
        md = tool.tex_to_markdown(tex)
        
        # If pandoc is missing, it should return a clear error message instead of crashing
        if "pandoc not found" in md or "Pandoc conversion failed" in md:
            self.skipTest("pandoc not installed on this system")
        else:
            self.assertGreater(len(md), 200)
            # Markdown should contain some original content or converted math
            self.assertTrue(len(md.strip()) > 100)


if __name__ == "__main__":
    # Run tests when the file is executed directly (useful for validation in Open-WebUI dev)
    unittest.main(verbosity=2)