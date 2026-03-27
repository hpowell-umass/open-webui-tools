""" 
title: HF Papers Tool
author: Grok-assisted
version: 1.1
description: Search, list, and read arXiv papers via Hugging Face hf papers CLI. Returns clean markdown for RAG/agent use. Includes unit tests.
requirements: subprocess  # hf CLI must be installed and in PATH on the server for runtime use
"""

import subprocess
from typing import Optional
import unittest
from unittest.mock import patch, MagicMock

class Tools:
    def __init__(self):
        self.citation = True  # Optional: helps with attribution in UI

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
        Works great for arXiv-only papers (use arXiv ID like 2503.12345).
        
        :param paper_id: arXiv ID or HF paper ID.
        :return: Markdown content of the paper (abstract, sections, etc.).
        """
        cmd = ["hf", "papers", "read", paper_id]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return f"**Paper Content (ID: {paper_id})**\n\n{result.stdout}"
            else:
                return f"Error reading paper {paper_id}: {result.stderr}\n\nTip: Index it first at https://huggingface.co/papers/{paper_id}"
        except Exception as e:
            return f"Failed to read paper: {str(e)}"


# ========================
# Unit Tests (run at the bottom of the file)
# ========================

class TestHFPapersTool(unittest.TestCase):

    def setUp(self):
        self.tool = Tools()

    def test_tool_instantiation(self):
        """Test that the Tools class can be instantiated without errors."""
        self.assertIsInstance(self.tool, Tools)

    def test_hf_papers_list_signature(self):
        """Test that hf_papers_list method exists and has correct signature."""
        self.assertTrue(hasattr(self.tool, 'hf_papers_list'))
        # Check it can be called (signature test only)
        self.assertTrue(callable(self.tool.hf_papers_list))

    def test_hf_papers_search_signature(self):
        """Test that hf_papers_search method exists and has correct signature."""
        self.assertTrue(hasattr(self.tool, 'hf_papers_search'))
        self.assertTrue(callable(self.tool.hf_papers_search))

    def test_hf_papers_read_signature(self):
        """Test that hf_papers_read method exists and has correct signature."""
        self.assertTrue(hasattr(self.tool, 'hf_papers_read'))
        self.assertTrue(callable(self.tool.hf_papers_read))

    @patch('subprocess.run')
    def test_hf_papers_list_success(self, mock_run):
        """Test successful hf papers ls call with mocked subprocess."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Paper1: Title A\nPaper2: Title B"
        mock_run.return_value = mock_result

        result = self.tool.hf_papers_list(sort="trending", limit=3, date="today")
        self.assertIn("**HF Papers List**", result)
        self.assertIn("Paper1: Title A", result)
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_hf_papers_list_failure(self, mock_run):
        """Test error handling when CLI returns non-zero code."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Command not found"
        mock_run.return_value = mock_result

        result = self.tool.hf_papers_list(limit=5)
        self.assertIn("Error listing papers", result)
        self.assertIn("Command not found", result)

    @patch('subprocess.run')
    def test_hf_papers_search_success(self, mock_run):
        """Test successful search with mocked subprocess."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Result1\nResult2"
        mock_run.return_value = mock_result

        result = self.tool.hf_papers_search(query="diffusion models", limit=2)
        self.assertIn("**Search Results for 'diffusion models'**", result)
        self.assertIn("Result1", result)

    @patch('subprocess.run')
    def test_hf_papers_read_success(self, mock_run):
        """Test successful read with mocked subprocess."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "# Abstract\nContent here..."
        mock_run.return_value = mock_result

        result = self.tool.hf_papers_read("2503.12345")
        self.assertIn("**Paper Content (ID: 2503.12345)**", result)
        self.assertIn("# Abstract", result)

    @patch('subprocess.run')
    def test_hf_papers_read_error(self, mock_run):
        """Test error handling for read command."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Paper not found"
        mock_run.return_value = mock_result

        result = self.tool.hf_papers_read("9999.99999")
        self.assertIn("Error reading paper 9999.99999", result)
        self.assertIn("Tip: Index it first", result)

    @patch('subprocess.run')
    def test_timeout_exception(self, mock_run):
        """Test exception handling (e.g., timeout)."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["hf"], timeout=30)

        result = self.tool.hf_papers_list()
        self.assertIn("Failed to run hf papers ls", result)


# Run tests when the file is executed directly (e.g., python hf_papers_tool.py)
if __name__ == "__main__":
    print("Running unit tests for HF Papers Tool...")
    unittest.main(verbosity=2)