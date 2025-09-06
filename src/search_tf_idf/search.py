import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TFIDF_NGRAM_RANGE, TOP_N_SEARCH

class TfidfSearch:
    """
    Класс для инкапсуляции логики TF-IDF поиска.
    Хранит в себе векторизатор, матрицу и оригинальные тексты.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE)
        self.matrix = None
        self.original_texts_df = None

    def build_index(self, lemmatized_texts: pd.Series, original_texts: pd.Series):
        """
        Строит TF-IDF матрицу на основе корпуса лемматизированных текстов.
        Сохраняет оригинальные тексты для возврата в результатах поиска.
        """
        print("Building TF-IDF index...")
        self.matrix = self.vectorizer.fit_transform(lemmatized_texts)
        self.original_texts_df = original_texts.reset_index(drop=True)
        print("TF-IDF index built successfully.")

    def search(self, query: str, preprocessor_func, top_n: int = TOP_N_SEARCH) -> List[Tuple[int, str]]:
        """
        Выполняет поиск по построенному индексу.
        
        Args:
            query (str): Поисковый запрос.
            preprocessor_func: Функция для предобработки запроса.
            top_n (int): Количество лучших результатов для возврата.
        
        Returns:
            List[Tuple[int, str]]: Список кортежей (индекс документа, текст документа).
        """
        if self.matrix is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")
        
        processed_query = preprocessor_func(query)
        query_vector = self.vectorizer.transform([processed_query])
        
        cosine_sim = cosine_similarity(query_vector, self.matrix).flatten()
        
        count = min(top_n, self.matrix.shape[0]) 
        
        top_indices = np.argpartition(cosine_sim, -count)[-count:]
        sorted_top_indices = top_indices[np.argsort(cosine_sim[top_indices])][::-1]
        
        results = [
            (idx, self.original_texts_df.iloc[idx]) for idx in sorted_top_indices
        ]
        
        return results