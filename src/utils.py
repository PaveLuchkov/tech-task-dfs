import pandas as pd


def load_dataset(file_name: str = "df_cleaned"):
    data_path = f'../data/{file_name}.parquet'
    try:
        df = pd.read_parquet(data_path)
        print(f"Данные успешно загружены из {data_path}. Количество записей: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"Ошибка: Файл {data_path} не найден. Убедитесь, что вы запустили ноутбук с EDA и предобработкой.")
        return None