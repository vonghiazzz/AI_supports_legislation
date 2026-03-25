import json
import os
import numpy as np
from app.config import settings
from app.services.embedding_service import embedding_service

class SearchService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("[*] Initializing Search Service (FAISS)...")
            cls._instance = super(SearchService, cls).__new__(cls)
            cls._instance.index = None
            cls._instance.metadata = []
            cls._instance.cross_encoder = None
            cls._instance.data_loaded = False
        return cls._instance

    def _ensure_data_loaded(self):
        if not self.data_loaded:
            print("[*] Lazy loading Search Service data and models...")
            
            # 1. Khởi tạo Embedding model trước (Sử dụng torch/SentenceTransformer)
            # Điều này đảm bảo torch được load trước faiss
            _ = embedding_service.encode("test")
            
            # 2. Khởi tạo Reranker (Cũng sử dụng torch)
            model_name = settings.RERANK_MODEL
            print(f"[*] Loading CrossEncoder Reranker: {model_name}...")
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder(model_name, device='cpu')
            except Exception as e:
                print(f"❌ Error loading CrossEncoder: {e}. Reranking will be disabled.")
                self.cross_encoder = None
            
            # 3. Load FAISS cuối cùng (Lazy import faiss inside load_data)
            self.load_data()
                
            self.data_loaded = True

    def load_data(self):
        import faiss
        index_path = settings.INDEX_PATH
        metadata_path = settings.METADATA_PATH
        print(f"[*] SearchService loading metadata from: {metadata_path}")
        print(f"[*] SearchService loading index from: {index_path}")

        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
                print(f"✅ Loaded FAISS index from {index_path}")
            except Exception as e:
                print(f"❌ Error reading FAISS index: {e}")
                self.index = None
        else:
            print(f"❌ FAISS index not found at {index_path}")

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                print(f"✅ Loaded {len(self.metadata)} metadata records from {metadata_path}")
            except Exception as e:
                print(f"❌ Error reading metadata: {e}")
                self.metadata = []
        else:
            print(f"❌ Metadata file not found at {metadata_path}")

    def search(self, query: str, top_k: int = 5):
        self._ensure_data_loaded()
        if self.index is None:
            print("⚠️ FAISS index is not initialized.")
            return []

        import faiss
        # 1. Encode query
        query_vector = embedding_service.encode(query).astype('float32').reshape(1, -1)
        
        # 2. Search FAISS initially for top 20 (as requested)
        fetch_k = 20
        distances, indices = self.index.search(query_vector, k=fetch_k)
        
        # 3. Format results and prepare inputs for Reranking
        results = []
        cross_inp = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx]
                results.append({
                    "law_id": item.get("law_id"),
                    "content": item.get("content"),
                    "faiss_score": float(distances[0][i]),
                    "source": item.get("source")
                })
                cross_inp.append([query, item.get("content", "")])
                
        if not results:
            return []

        # 4. Rerank using CrossEncoder (nếu load thành công)
        if self.cross_encoder:
            try:
                cross_scores = self.cross_encoder.predict(cross_inp)
        
                for i in range(len(results)):
                    results[i]["score"] = float(cross_scores[i])
                
                # 5. Threshold Filtering: chỉ giữ điểm rerank > 0
                results = [res for res in results if res.get("score", -10) > 0]
                
                # 6. Sort by CrossEncoder score (higher is better)
                results = sorted(results, key=lambda x: x["score"], reverse=True)
            except Exception as e:
                print(f"⚠️ Error during reranking: {e}. Falling back to FAISS scores.")
                for res in results:
                    res["score"] = res["faiss_score"]
                results = sorted(results, key=lambda x: x["score"], reverse=True)
        else:
            # Fallback if no reranker
            for res in results:
                res["score"] = res["faiss_score"]
            results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]

search_service = SearchService()
