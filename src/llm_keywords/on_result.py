import pandas as pd

def save_results(df: pd.DataFrame, output_path: str = "../data/df_with_llmkeyphrases.parquet"):
    """Сохраняет результаты в файл"""
    df.to_parquet(output_path, index=False)
    print(f"\n💾 Результаты сохранены в {output_path}")

def display_sample_results(df: pd.DataFrame, n_samples: int = 3):
    """Показывает примеры результатов"""
    print(f"\n=== ПРИМЕРЫ РЕЗУЛЬТАТОВ ===")
    
    df_with_keyphrases = df[df['keyphrases'].notna()]
    
    if len(df_with_keyphrases) == 0:
        print("Нет обработанных аннотаций")
        return
    
    sample_df = df_with_keyphrases.sample(n=min(n_samples, len(df_with_keyphrases)))
    
    for idx, row in sample_df.iterrows():
        print(f"\n--- Аннотация {idx + 1} ---")
        print(f"Текст: {row['abstract'][:200]}...")
        print(f"Ключевые фразы: {row['keyphrases']}")
        print("-" * 50)