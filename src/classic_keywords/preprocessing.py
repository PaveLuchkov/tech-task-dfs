import re
import nltk
from functools import lru_cache
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
from src.config import MIN_WORD_COUNT
from src.config import MAX_AVG_WORD_LEN

try:
    russian_stopwords = stopwords.words("russian")
except LookupError:
    nltk.download('stopwords', quiet=True)
    russian_stopwords = stopwords.words("russian")

morph = MorphAnalyzer()



def is_valid_abstract(text: str) -> bool:
    """
    Проверяет, является ли аннотация валидной на основе заданных критериев.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    words = text.split()
    word_count = len(words)
    if word_count < MIN_WORD_COUNT:
        return False
    total_chars_in_words = len("".join(words))
    if total_chars_in_words == 0:
        return False
    avg_word_length = total_chars_in_words / word_count
    if avg_word_length > MAX_AVG_WORD_LEN:
        return False
    return True

@lru_cache(maxsize=100_000)
def lemmatize_word(token: str) -> str:
    """Лемматизирует одно слово с кешированием."""
    return morph.parse(token)[0].normal_form

def preprocess_text(text: str) -> str:
    """
    Выполняет полную предобработку текста для TF-IDF.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^а-яa-z0-9\-]', ' ', text)
    tokens = text.split()
    lemmatized_tokens = [lemmatize_word(token) for token in tokens]
    cleaned_tokens = [
        token for token in lemmatized_tokens
        if token not in russian_stopwords and len(token) > 2
    ]
    return " ".join(cleaned_tokens)