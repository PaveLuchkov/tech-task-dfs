import pandas as pd

def load_and_sample_data(file_path: str, sample_size: int = 200) -> pd.DataFrame:
    """
    Загружает данные и выбирает случайную выборку
    """
    df = pd.read_parquet(file_path)
    print(f"Загружено {len(df)} аннотаций")
    
    # Выбираем случайные 200 аннотаций
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    print(f"Выбрано {len(df_sample)} аннотаций для обработки")
    
    return df_sample