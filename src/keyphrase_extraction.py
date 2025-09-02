import yake
from typing import List
from src.config import YAKE_LANGUAGE, YAKE_MAX_NGRAM_SIZE, YAKE_NUM_KEYWORDS

kw_extractor = yake.KeywordExtractor(
    lan=YAKE_LANGUAGE,
    n=YAKE_MAX_NGRAM_SIZE,
    dedupLim=0.9,
    top=YAKE_NUM_KEYWORDS,
    features=None
)

def extract_yake_keyphrases(text: str) -> List[str]:
    """
    Извлекает ключевые фразы из текста с помощью YAKE.
    Возвращает список ключевых фраз (только строки).
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    keywords_with_scores = kw_extractor.extract_keywords(text)
    
    keywords = [kw for kw, score in keywords_with_scores]
    
    return keywords