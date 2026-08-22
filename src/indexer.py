import os
import json
import re
import math
import frontmatter
from pathlib import Path
from typing import Optional, Dict, Any, List
from openai import OpenAI
from src.config import  KNOWLEDGE_BASE_DIR, OPENAI_API_KEY, EMBEDDING_MODEL

class KnowledgeBaseIndexer:
    def __init__(self, kb_dir: Path = KNOWLEDGE_BASE_DIR, index_cache_file: Path = Path ("data/kb_index.json")):
        self.kb_dir = kb_dir
        self.index_cache_file = index_cache_file
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.chunks: List[Dict[str, Any]] = []

    def _split_markdown_by_headings(self, text: str, filename: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits markdown text into chunks based on headings.
        Each chunk is a dict with 'filename', 'heading', 'content', and 'metadata'.
        """
        chunks = []
        current_heading = "Introduction"
        current_content = []
        for line in text.splitlines():
            heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
            if heading_match:
                # Save the previous chunk if it exists
                if current_content:
                    chunks.append({
                        "filename": filename,
                        "heading": current_heading,
                        "content": "\n".join(current_content).strip(),
                        "metadata": metadata
                    })
                    current_content = []
                current_heading = heading_match.group(2).strip()
            else:
                current_content.append(line)
        # Add the last chunk
        if current_content:
            chunks.append({

                "filename": filename,
                "heading": current_heading,
                "content": "\n".join(current_content).strip(),
                "metadata": metadata
            })
        return chunks

    def load_and_chunk_documents(self) -> List[Dict[str, Any]]:
        """
        Loads markdown documents from the knowledge base directory,
        splits them into chunks, and returns a list of chunk dicts.
        """
        all_chunks = []
        if not self.kb_dir.exists():
            print(f"Knowledge base directory {self.kb_dir} does not exist.")
            return all_chunks
        for file_path in sorted(self.kb_dir.glob("*.md")):
            try:
                post = frontmatter.load(file_path)
                metadata = post.metadata
                doc_status = str(metadata.get("status", "draft")).lower().strip()
                metadata["status"] = doc_status
                metadata["filename"] = file_path.name
                chunks = self._split_markdown_by_headings(post.content, file_path.name, metadata)
                all_chunks.extend(chunks)
            except Exception as e:
              print(f"Error processing {file_path}: {e}")
        self.chunks = all_chunks
        return all_chunks

    def generate_embeddings(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Generates embeddings for each chunk using the specified embedding model.
        If force_refresh is True, it will regenerate embeddings even if they exist.
        """
        if not force_refresh and self.index_cache_file.exists():
            try:
                with open(self.index_cache_file, "r", encoding="utf-8") as f:
                 self.chunks = json.load(f)
                 return self.chunks
            except Exception as e:
                print(f"Error loading cache: {e}")
        if not self.client:
            raise ValueError("OpenAI API key is not set. Cannot generate embeddings.")
        if not self.chunks:
            self.load_and_chunk_documents()
        texts_to_embed = [chunk["content"] for chunk in self.chunks]
        print(f"Generating embeddings for {len(texts_to_embed)} chunks...")
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts_to_embed
            )
            for i, item in enumerate(response.data):
                self.chunks[i]["embedding"] = item.embedding

            self.index_cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_cache_file, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            print(f"Embeddings generated and saved to {self.index_cache_file}")
        except Exception as e:
            print(f"Error generating embeddings: {e}")

        return self.chunks
if __name__ == "__main__":
        indexer = KnowledgeBaseIndexer()
        chunks = indexer.load_and_chunk_documents()
        print(f"Total chunks created: {len(chunks)}")
        for chunk in chunks[:3]:  # Print first 3 chunks for inspection
            print(f"Filename: {chunk['filename']}, Heading: {chunk['heading']}, Content snippet: {chunk['content'][:150]}...")