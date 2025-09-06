# Параметры очистки данных
MIN_WORD_COUNT = 25
MAX_AVG_WORD_LEN = 15

# Параметры для YAKE
YAKE_LANGUAGE = "ru"
YAKE_MAX_NGRAM_SIZE = 3
YAKE_NUM_KEYWORDS = 7

# Параметры для TF-IDF
TFIDF_NGRAM_RANGE = (1, 3)
TOP_N_SEARCH = 5

# DEFAULT_MODEL_NAME = "google/embeddinggemma-300m"
DEFAULT_MODEL_NAME = "ai-forever/FRIDA"
DEFAULT_EMBEDDING_PATH = "../data/embeddings.npy"
DEFAULT_FAISS_INDEX_PATH = "../data/faiss_index.bin"
DEFAULT_BATCH_SIZE = 64