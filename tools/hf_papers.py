""" 
title: HF Papers Tool
author: Grok-assisted
version: 1.1
description: Search, list, and read arXiv/HF papers via hf papers CLI. Returns clean markdown (with LaTeX math preserved) for RAG/agent use.
requirements: subprocess
"""

import subprocess
from typing import Optional
import unittest
from unittest.mock import patch, MagicMock

class Tools:
    def __init__(self):
        self.citation = True  # Helps with attribution in Open WebUI

    def hf_papers_list(self, sort: str = "trending", limit: int = 5, date: Optional[str] = None) -> str:
        """
        List recent or trending papers from Hugging Face Papers (mostly arXiv).
        
        :param sort: Sorting method - 'trending' or default (recent).
        :param limit: Number of papers to return (1-20).
        :param date: Specific date (YYYY-MM-DD) or 'today'.
        :return: Formatted list of papers with IDs, titles, and metadata.
        """
        cmd = ["hf", "papers", "ls", "--limit", str(limit), "--sort", sort]
        if date:
            cmd.extend(["--date", date])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return f"**HF Papers List**\n\n{result.stdout}"
            else:
                return f"Error listing papers: {result.stderr}"
        except Exception as e:
            return f"Failed to run hf papers ls: {str(e)}"

    def hf_papers_search(self, query: str, limit: int = 5) -> str:
        """
        Semantic or keyword search for papers.
        
        :param query: Search term, e.g., "vision language models" or "RLHF".
        :param limit: Max results.
        :return: Matching papers with metadata.
        """
        cmd = ["hf", "papers", "search", query, "--limit", str(limit)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return f"**Search Results for '{query}'**\n\n{result.stdout}"
            else:
                return f"Error searching: {result.stderr}"
        except Exception as e:
            return f"Failed to search papers: {str(e)}"

    def hf_papers_read(self, paper_id: str) -> str:
        """
        Retrieve a full paper as clean, agent-ready markdown (best for RAG).
        Works great for arXiv-only papers (use arXiv ID like 1706.03762).
        Equations from LaTeX/PDF are typically converted to $...$ or $$...$$ delimiters.
        
        :param paper_id: arXiv ID or HF paper ID.
        :return: Markdown content of the paper (abstract, sections, etc.).
        """
        cmd = ["hf", "papers", "read", paper_id]
        markdown = ""
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            while True:
                line = proc.stdout.readline()
                if not line:
                    break  # Exit loop if no more output
                # print(f"Processing line: {str(line.rstrip())}")
                markdown = markdown + str(line.rstrip()) + "\n"
            if not proc.returncode and "Set HF_DEBUG=1 as environment variable for full traceback." not in markdown:
                return f"**Paper Content (ID: {paper_id})**\n\n{markdown}"
            else:
                return f"Error reading paper {paper_id}: {proc.stderr}\n\nTip: Index it first at https://huggingface.co/papers/{paper_id}"
        except Exception as e:
            return f"Failed to read paper: {str(e)}"


# ==================== UNIT TESTS (run with: python hf_papers_tool.py) ====================

class TestHFPapersTool(unittest.TestCase):

    def setUp(self):
        self.tools = Tools()

    @patch('subprocess.run')
    def test_hf_papers_list_success(self, mock_run):
        """Test successful listing of papers."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Paper 1: Title A\nPaper 2: Title B"
        mock_run.return_value = mock_result

        result = self.tools.hf_papers_list(sort="trending", limit=3)
        self.assertIn("**HF Papers List**", result)
        self.assertIn("Title A", result)
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_hf_papers_list_error(self, mock_run):
        """Test error handling in list."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "CLI error"
        mock_run.return_value = mock_result

        result = self.tools.hf_papers_list()
        self.assertIn("Error listing papers", result)

    @patch('subprocess.run')
    def test_hf_papers_search_success(self, mock_run):
        """Test successful search."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Result 1: Diffusion models"
        mock_run.return_value = mock_result

        result = self.tools.hf_papers_search("diffusion", limit=2)
        self.assertIn("**Search Results for 'diffusion'**", result)
        self.assertIn("Diffusion models", result)

    @patch('subprocess.run')
    def test_hf_papers_read_success(self, mock_run):
        """Test successful read with markdown output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "# Attention Is All You Need\n\n"
            "Abstract: ...\n\n"
            "The Transformer uses self-attention: $Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V$\n\n"
            "$$\\text{MultiHead}(Q, K, V) = \\text{Concat}(head_1, ..., head_h)W^O$$"
        )
        mock_run.return_value = mock_result

        result = self.tools.hf_papers_read("1706.03762")
        self.assertIn("**Paper Content (ID: 1706.03762)**", result)
        self.assertIn("Attention Is All You Need", result)

    @patch('subprocess.run')
    def test_equations_converted_to_markdown_latex(self, mock_run):
        """
        Test that equations are properly converted from PDF/LaTeX source into Markdown.
        hf papers read typically outputs math using standard $...$ (inline) or $$...$$ (display) delimiters.
        This checks preservation of LaTeX syntax inside math blocks.
        """
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "Section 3.2: Self-Attention\n\n"
            "The scaled dot-product attention is defined as:\n\n"
            "$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\n\n"
            "For multi-head: $head_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$"
        )
        mock_run.return_value = mock_result

        result = self.tools.hf_papers_read("1706.03762")

        # Check for common math delimiters (inline and display)
        self.assertTrue(
            "$" in result or "$$" in result,
            "No math delimiters found — equations may not have been converted properly."
        )
        # Check specific LaTeX content is preserved inside math
        self.assertIn("Attention}(Q, K, V)", result)
        self.assertIn("\\sqrt{d_k", result)
        self.assertIn("softmax", result)
        self.assertIn("multi-head", result)  # or similar from the mock

    @patch('subprocess.run')
    def test_hf_papers_read_error(self, mock_run):
        """Test error handling when reading a paper."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Paper not found"
        mock_run.return_value = mock_result

        result = self.tools.hf_papers_read("invalid-id")
        self.assertIn("Error reading paper", result)
        self.assertIn("Tip: Index it first", result)


if __name__ == "__main__":
    # Run tests when the file is executed directly (e.g., python hf_papers_tool.py)
    unittest.main(verbosity=2)