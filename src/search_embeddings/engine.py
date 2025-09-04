
import os
import time
from typing import List, Tuple

import torch
import faiss
import numpy as np
import pandas as pd
from src import config as cfg
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = cfg.DEFAULT_MODEL_NAME
DEFAULT_EMBEDDING_PATH = cfg.DEFAULT_EMBEDDING_PATH
DEFAULT_FAISS_INDEX_PATH = cfg.DEFAULT_FAISS_INDEX_PATH
DEFAULT_BATCH_SIZE = cfg.DEFAULT_BATCH_SIZE

class EmbeddingSearchEngine:
    """
    Класс для семантического поиска с использованием FAISS.
    """

    def __init__(self,
                 model_name: str = DEFAULT_MODEL_NAME,
                 embedding_path: str = DEFAULT_EMBEDDING_PATH,
                 faiss_index_path: str = DEFAULT_FAISS_INDEX_PATH):
        """Инициализация движка."""
        self.embedding_path = embedding_path
        self.faiss_index_path = faiss_index_path
        self.device = self._get_optimal_device()
        
        print(f"Loading sentence transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Model {model_name} loaded successfully on device '{self.device}'.")

        self.index: faiss.Index = None
        self.original_texts: List[str] = None

    def _get_optimal_device(self) -> str:
        """Определяет наилучшее доступное устройство."""
        if torch.backends.mps.is_available():
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def build_index(self, texts: pd.Series, force_rebuild: bool = False):
        """
        Создает векторный индекс FAISS.
        Если кэшированные файлы существуют, загружает их.
        """
        self.original_texts = texts.tolist()

        if os.path.exists(self.faiss_index_path) and not force_rebuild:
            print(f"Loading pre-built FAISS index from {self.faiss_index_path}...")
            self.index = faiss.read_index(self.faiss_index_path)
            print(f"FAISS index loaded. Contains {self.index.ntotal} vectors.")
            return

        if os.path.exists(self.embedding_path) and not force_rebuild:
            print(f"Loading pre-computed embeddings from {self.embedding_path}...")
            embeddings = np.load(self.embedding_path)
        else:
            print("Building embeddings from scratch...")
            start_time = time.time()
            documents_with_prefix = ["search_document: " + str(text) for text in texts]
            embeddings = self.model.encode(
                documents_with_prefix,
                show_progress_bar=True,
                batch_size=DEFAULT_BATCH_SIZE,
                convert_to_numpy=True,
                device=self.device
            )
            end_time = time.time()
            print(f"Embeddings built in {end_time - start_time:.2f} seconds.")
            print(f"Saving embeddings to {self.embedding_path}...")
            os.makedirs(os.path.dirname(self.embedding_path), exist_ok=True)
            np.save(self.embedding_path, embeddings)

        print("Building FAISS index...")
        start_time = time.time()
        
        d = embeddings.shape[1]
        
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(d)
        
        self.index.add(embeddings)
        
        end_time = time.time()
        print(f"FAISS index built in {end_time - start_time:.2f} seconds. Contains {self.index.ntotal} vectors.")

        print(f"Saving FAISS index to {self.faiss_index_path}...")
        faiss.write_index(self.index, self.faiss_index_path)
        print("FAISS index saved successfully.")

    def search(self, query: str, top_n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Выполняет семантический поиск с использованием FAISS.
        """
        if self.index is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")

        start_time = time.time()
        
        query_with_prefix = "search_query: " + query
        query_embedding = self.model.encode(
            query_with_prefix,
            convert_to_numpy=True
        ).reshape(1, -1)
        
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_n)
        
        end_time = time.time()
        print(f"FAISS search completed in {end_time - start_time:.4f} seconds.")

        results = []
        for i in range(top_n):
            doc_index = indices[0][i]
            similarity_score = round(distances[0][i], 4)
            if doc_index != -1:
                doc_text = self.original_texts[doc_index]
                results.append((doc_index, doc_text, similarity_score))
            
        return results