import math
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, TOP_K_RETRIEVAL
from src.indexer import KnowledgeBaseIndexer

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class KnowledgeBaseRetriever:
    def __init__(self, indexer: Optional[KnowledgeBaseIndexer] = None):
        self.indexer = indexer or KnowledgeBaseIndexer()
        self.chunks = self.indexer.load_and_chunk_documents()
        
        # Check if valid OpenAI key exists
        is_valid_key = bool(OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your_"))
        self.client = OpenAI(api_key=OPENAI_API_KEY) if is_valid_key else None
        
        if self.client:
            try:
                self.chunks = self.indexer.generate_embeddings()
            except Exception as e:
                print(f"Embeddings skipped ({e}), using BM25 fallback.")

    def retrieve(
        self, 
        query: str, 
        top_k: int = TOP_K_RETRIEVAL, 
        include_superseded: bool = False
    ) -> Dict[str, Any]:
        if not query or not query.strip():
            return {"query": query, "best_score": 0.0, "is_insufficient": True, "chunks": []}

        # Step 1: Filter out superseded and internal docs by default
        eligible_chunks = []
        for chunk in self.chunks:
            meta = chunk.get("metadata", {})
            status = str(meta.get("status", "active")).lower().strip()
            filename = chunk.get("filename", "").lower()

            if "internal" in filename or meta.get("doc_type") == "internal":
                continue
            if not include_superseded and status in ["superseded", "legacy", "archived"]:
                continue
            eligible_chunks.append(chunk)

        if not eligible_chunks:
            return {"query": query, "best_score": 0.0, "is_insufficient": True, "chunks": []}

        # Step 2: Vector Search (agar embeddings available hain)
        has_embeddings = any("embedding" in c for c in eligible_chunks)
        if self.client and has_embeddings:
            try:
                query_res = self.client.embeddings.create(model=EMBEDDING_MODEL, input=query)
                query_embedding = query_res.data[0].embedding
                
                scored_chunks = []
                for chunk in eligible_chunks:
                    emb = chunk.get("embedding")
                    if emb:
                        sim = cosine_similarity(query_embedding, emb)
                        scored_chunks.append((sim, chunk))
                
                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                top_matches = scored_chunks[:top_k]
                best_score = top_matches[0][0] if top_matches else 0.0
                is_insufficient = best_score < 0.35

                return {
                    "query": query,
                    "best_score": round(best_score, 4),
                    "is_insufficient": is_insufficient,
                    "chunks": [
                        {
                            "filename": c["filename"],
                            "heading": c["heading"],
                            "content": c["content"],
                            "score": round(s, 4),
                            "metadata": c.get("metadata", {})
                        }
                        for s, c in top_matches
                    ]
                }
            except Exception as e:
                print(f"Vector search exception ({e}), falling back to BM25...")

        # Step 3: BM25 Keyword Search Fallback
        query_tokens = re.findall(r"\w+", query.lower())
        tokenized_eligible = [
            re.findall(r"\w+", (c.get("heading", "") + " " + c.get("content", "")).lower())
            for c in eligible_chunks
        ]
        bm25_model = BM25Okapi(tokenized_eligible)
        doc_scores = bm25_model.get_scores(query_tokens)

        scored_bm25 = list(zip(doc_scores, eligible_chunks))
        scored_bm25.sort(key=lambda x: x[0], reverse=True)
        top_matches = scored_bm25[:top_k]

        best_score = top_matches[0][0] if top_matches else 0.0
        is_insufficient = best_score <= 0.0

        return {
            "query": query,
            "best_score": round(float(best_score), 4),
            "is_insufficient": is_insufficient,
            "chunks": [
                {
                    "filename": c["filename"],
                    "heading": c["heading"],
                    "content": c["content"],
                    "score": round(float(s), 4),
                    "metadata": c.get("metadata", {})
                }
                for s, c in top_matches
            ]
        }


if __name__ == "__main__":
    retriever = KnowledgeBaseRetriever()
    test_query = "What is the return window for standard items?"
    print(f"\n--- Searching for: '{test_query}' ---")
    results = retriever.retrieve(test_query)
    
    print(f"Insufficient Info Flag: {results['is_insufficient']} (Best Score: {results['best_score']})")
    for r in results["chunks"]:
        print(f"\n[Source: {r['filename']} > {r['heading']}] (Score: {r['score']})")
        print(r["content"][:200] + "...")