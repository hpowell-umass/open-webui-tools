"""
title: arXiv Search & LaTeX-to-Markdown Converter
author: Grok (built by xAI)
author_url: https://x.ai
version: 1.0
requirements: arxiv
description: Search arXiv papers using the official arxiv Python package and convert any paper's LaTeX source to clean Markdown (with math support). Requires pandoc installed on the system for conversion.
"""

import arxiv
import json
import os
import re
import subprocess
import tempfile
import tarfile
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
        """Search arXiv for papers using the official arxiv Python package.
        
        Supports advanced arXiv query syntax (e.g. "au:Einstein ti:relativity cat:physics").
        
        :param query: Search query string
        :param max_results: Maximum number of results (default: 10, max recommended ~50)
        :param sort_by: Sort criterion - "relevance" or "submittedDate" (default: relevance)
        :return: JSON array of paper metadata including arXiv ID (for use with other functions), title, authors, abstract, PDF URL, and source URL
        """
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
                    "arxiv_id": result.get_short_id(),  # e.g. "2503.12345" - use this with get_tex_source
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
        """Download the raw LaTeX source tarball for a paper using the arxiv package and return the main .tex content as a string.
        
        :param arxiv_id: arXiv ID (short form like "2503.12345" or full "arXiv:2503.12345")
        :return: Full LaTeX source of the main .tex file (ready for tex_to_markdown) or error message
        """
        try:
            # Normalize ID
            if arxiv_id.startswith("arXiv:"):
                arxiv_id = arxiv_id.split(":")[-1]
            
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            papers = list(client.results(search))
            
            if not papers:
                return f"Error: Paper {arxiv_id} not found."
            
            result = papers[0]
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Download source using arxiv package (always a tar.gz)
                archive_name = f"{arxiv_id}.tar.gz"
                archive_path = os.path.join(tmp_dir, archive_name)
                
                # The arxiv package download_source signature: download_source(dirpath=".", filename=None)
                result.download_source(dirpath=tmp_dir, filename=archive_name)
                
                # Extract to a subdirectory
                extract_dir = os.path.join(tmp_dir, "extracted")
                os.makedirs(extract_dir, exist_ok=True)
                
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=extract_dir)
                
                # Find the main .tex file (contains \documentclass)
                main_tex_path = None
                for root, _, files in os.walk(extract_dir):
                    for file in files:
                        if file.lower().endswith(".tex"):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                    content = f.read(5000)  # first 5k chars is enough
                                if re.search(r"\\documentclass", content, re.IGNORECASE):
                                    main_tex_path = file_path
                                    break
                            except:
                                continue
                    if main_tex_path:
                        break
                
                # Fallback: first .tex file
                if not main_tex_path:
                    for root, _, files in os.walk(extract_dir):
                        for file in files:
                            if file.lower().endswith(".tex"):
                                main_tex_path = os.path.join(root, file)
                                break
                        if main_tex_path:
                            break
                
                if not main_tex_path:
                    return f"Error: No .tex file found in source for {arxiv_id}."
                
                # Read the full main .tex
                with open(main_tex_path, "r", encoding="utf-8", errors="ignore") as f:
                    tex_content = f.read()
                
                return tex_content
                
        except Exception as e:
            return f"Error fetching TeX source for {arxiv_id}: {str(e)}"

    def tex_to_markdown(self, tex_source: str) -> str:
        """Convert LaTeX source (from get_tex_source) to clean Markdown using pandoc.
        
        Preserves equations (as $...$ or $$...$$), sections, lists, tables, etc.
        Note: pandoc must be installed on the system (see instructions below).
        
        :param tex_source: Raw LaTeX source string
        :return: Markdown version of the paper
        """
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tex_file = os.path.join(tmp_dir, "paper.tex")
                md_file = os.path.join(tmp_dir, "paper.md")
                
                # Write LaTeX to disk
                with open(tex_file, "w", encoding="utf-8") as f:
                    f.write(tex_source)
                
                # Run pandoc (works best when all included files are present; simple papers convert perfectly)
                cmd = [
                    "pandoc",
                    tex_file,
                    "-o", md_file,
                    "-f", "latex+raw_tex",
                    "-t", "markdown+tex_math_single_backslash+raw_tex",
                    "--mathjax",
                    "--standalone",
                    "--wrap=none",
                ]
                
                subprocess.run(
                    cmd,
                    check=True,
                    cwd=tmp_dir,
                    capture_output=True,
                    text=True,
                )
                
                with open(md_file, "r", encoding="utf-8") as f:
                    markdown = f.read()
                
                return markdown
                
        except FileNotFoundError:
            return "Error: pandoc not found. Install it with: apt-get install -y pandoc (or equivalent for your OS)."
        except subprocess.CalledProcessError as e:
            return f"Pandoc conversion failed: {e.stderr.strip() if e.stderr else str(e)}"
        except Exception as e:
            return f"Conversion error: {str(e)}"