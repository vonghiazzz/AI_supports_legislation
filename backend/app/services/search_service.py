import json
import os
import numpy as np
from app.config import settings
from app.services.embedding_service import embedding_service
from typing import List, Optional, Dict, Any 

class SearchService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("[*] Initializing Search Service (FAISS + BM25)...")
            cls._instance = super(SearchService, cls).__new__(cls)
            cls._instance.index = None
            cls._instance.metadata = []
            cls._instance.bm25 = None
            cls._instance.tokenized_corpus = []
            cls._instance.cross_encoder = None
            cls._instance.data_loaded = False
        return cls._instance

    def _ensure_data_loaded(self):
        if not self.data_loaded:
            print("[*] Lazy loading Search Service data and models...")
            
            # 1. Khởi tạo Embedding model trước
            _ = embedding_service.encode("test")
            
            # 2. Khởi tạo Reranker
            model_name = settings.RERANK_MODEL
            print(f"[*] Loading CrossEncoder Reranker: {model_name}...")
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder(model_name, device='cpu')
            except Exception as e:
                print(f"❌ Error loading CrossEncoder: {e}. Reranking will be disabled.")
                self.cross_encoder = None
            
            # 3. Load Data & Indices
            self.load_data()
                
            self.data_loaded = True

    def load_data(self):
        import faiss
        index_path = settings.INDEX_PATH
        metadata_path = settings.METADATA_PATH
        print(f"[*] SearchService loading metadata from: {metadata_path}")
        print(f"[*] SearchService loading index from: {index_path}")

        # Load Metadata
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                print(f"✅ Loaded {len(self.metadata)} metadata records")
                
                # Khởi tạo BM25
                print("[*] Initializing BM25 index...")
                try:
                    from rank_bm25 import BM25Okapi
                    import re
                    
                    # Tokenize dân dã cho tiếng Việt
                    def tokenize(text):
                        return re.findall(r'\w+', text.lower())
                    
                    self.tokenized_corpus = []
                    for item in self.metadata:
                        # Kết hợp Content, Title và Keywords để BM25 tìm chính xác hơn
                        text = f"{item.get('title', '')} {item.get('content', '')} {' '.join(item.get('keywords', []))}"
                        self.tokenized_corpus.append(tokenize(text))
                        
                    self.bm25 = BM25Okapi(self.tokenized_corpus)
                    print("✅ BM25 index ready.")
                except Exception as e:
                    print(f"❌ Error initializing BM25: {e}. BM25 will be disabled.")
                    self.bm25 = None
                    
            except Exception as e:
                print(f"❌ Error reading metadata: {e}")
                self.metadata = []
        else:
            print(f"❌ Metadata file not found at {metadata_path}")

        # Load FAISS
        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
                print(f"✅ Loaded FAISS index from {index_path}")
            except Exception as e:
                print(f"❌ Error reading FAISS index: {e}")
                self.index = None
        else:
            print(f"❌ FAISS index not found at {index_path}")

    def search_vector(self, query: str, top_k: int = 20):
        """Tìm kiếm bằng vector (FAISS)."""
        if self.index is None: return []
        
        query_vector = embedding_service.encode(query).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_vector, k=top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx]
                results.append({
                    "law_id": item.get("law_id"),
                    "title": item.get("title", "Không có tiêu đề"),
                    "content": item.get("content"),
                    "category": item.get("category"),
                    "chapter": item.get("chapter"),
                    "section": item.get("section"),
                    "keywords": item.get("keywords", []),
                    "tags": item.get("tags", []),
                    "type": item.get("type"),
                    "priority": item.get("priority", 0.5),
                    "vector_score": float(distances[0][i]),
                    "id": idx
                })
        return results

    def search_bm25(self, keywords: Any, top_k: int = 10):
        """Tìm kiếm bằng từ khóa (BM25). Chấp nhận List hoặc Dict."""
        if self.bm25 is None or not keywords: return []
        
        # Merge các keywords thành 1 list tokens
        import re
        search_tokens = []
        
        # Nếu là Dict từ LLMService mới (có keywords và lexical_keywords)
        if isinstance(keywords, dict):
            # Gộp tất cả các list con lại
            target_list = []
            for val in keywords.values():
                if isinstance(val, list):
                    target_list.extend(val)
        else:
            target_list = keywords # Giả định là list[str]

        for kw in target_list:
            if isinstance(kw, str):
                search_tokens.extend(re.findall(r'\w+', kw.lower()))
            
        if not search_tokens: return []
        
        scores = self.bm25.get_scores(search_tokens)
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            if scores[idx] > 0:
                item = self.metadata[idx]
                results.append({
                    "law_id": item.get("law_id"),
                    "title": item.get("title", "Không có tiêu đề"),
                    "content": item.get("content"),
                    "category": item.get("category"),
                    "chapter": item.get("chapter"),
                    "section": item.get("section"),
                    "keywords": item.get("keywords", []),
                    "tags": item.get("tags", []),
                    "type": item.get("type"),
                    "priority": item.get("priority", 0.5),
                    "bm25_score": float(scores[idx]),
                    "id": int(idx)
                })
        return results

    def search_hybrid(self, queries: List[str], keywords: List[str], top_k: int = 5):
        """Hybrid Search: Multi-query Vector + BM25 + Rerank."""
        self._ensure_data_loaded()
        
        all_results = {} # Dùng dict để deduplicate theo 'id' (index trong metadata)

        # 1. Multi-query Vector Search
        for q in queries:
            print(f"[*] Vector searching for: {q[:50]}...")
            vec_res = self.search_vector(q, top_k=20)
            for r in vec_res:
                rid = r["id"]
                if rid not in all_results:
                    # Khởi tạo mặc định cho bm25_score
                    r["bm25_score"] = 0.0
                    all_results[rid] = r
                else:
                    # Update score nếu tìm thấy score cao hơn
                    if r["vector_score"] > all_results[rid].get("vector_score", 0.0):
                        all_results[rid]["vector_score"] = r["vector_score"]

        # 2. BM25 Search
        if keywords:
            print(f"[*] BM25 searching for: {keywords}...")
            bm_res = self.search_bm25(keywords, top_k=20)
            for r in bm_res:
                rid = r["id"]
                if rid not in all_results:
                    # Khởi tạo mặc định cho vector_score
                    r["vector_score"] = 0.0
                    all_results[rid] = r
                else:
                    all_results[rid]["bm25_score"] = r["bm25_score"]
                    # Đảm bảo source được lấy từ BM25 nếu Vector search chưa có (hiếm)
                    if not all_results[rid].get("source") and r.get("source"):
                        all_results[rid]["source"] = r["source"]

        if not all_results:
            return []

        # 3. Chuyển list sang rerank
        candidates = list(all_results.values())
        
        # Lấy query gốc (thường là query cuối cùng hoặc đầu tiên trong list)
        main_query = queries[-1] if queries else ""
        
        # 4. Rerank bằng CrossEncoder
        if self.cross_encoder and main_query:
            try:
                cross_inp = [[main_query, c["content"]] for c in candidates]
                cross_scores = self.cross_encoder.predict(cross_inp)
                
                for i in range(len(candidates)):
                    candidates[i]["score"] = float(cross_scores[i])
                
                # Threshold Filtering (> 0)
                candidates = [c for c in candidates if c.get("score", -10) > 0]
                candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
            except Exception as e:
                print(f"⚠️ Reranking error: {e}. Using hybrid fallback.")
                self._apply_hybrid_fallback(candidates)
        else:
            # No reranker fallback
            self._apply_hybrid_fallback(candidates)

        return candidates[:top_k]

    def _apply_hybrid_fallback(self, candidates: List[Dict[str, Any]]):
        """Áp dụng scoring 0.6*Vector + 0.4*BM25 với normalization đơn giản."""
        if not candidates: return

        # Lấy max scores để normalize
        max_vec = max([c.get("vector_score", 0) for c in candidates]) or 1.0
        max_bm25 = max([c.get("bm25_score", 0) for c in candidates]) or 1.0

        for c in candidates:
            # Normalize scores về [0, 1]
            vec_norm = c.get("vector_score", 0) / max_vec
            bm25_norm = c.get("bm25_score", 0) / max_bm25
            
            # Hybrid Score (0.6/0.4)
            c["score"] = 0.6 * vec_norm + 0.4 * bm25_norm
        
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates[:top_k]

    def search(self, query: str, top_k: int = 5):
        """Legacy method for backward compatibility."""
        return self.search_hybrid([query], [], top_k)

search_service = SearchService()
