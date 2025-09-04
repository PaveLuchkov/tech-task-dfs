import os
import time
import asyncio
import pandas as pd
import pandas as pd
from google import genai
from .extracrt_keyphrases import RateLimiter
from .extracrt_keyphrases import extract_keyphrases_batch_async

async def process_all_abstracts_async(df: pd.DataFrame, batch_size: int = 20, 
                                    max_concurrent: int = 5, 
                                    max_retries: int = 3) -> pd.DataFrame:
    """Обрабатывает все аннотации батчами с повторными попытками"""
    
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    rate_limiter = RateLimiter(max_requests_per_minute=14)
    
    df = df.copy()
    df['keyphrases'] = None
    df['keyphrases'] = df['keyphrases'].astype('object')
    
    abstracts = df['abstract'].tolist()
    total_batches = (len(abstracts) + batch_size - 1) // batch_size
    
    print(f"Будет обработано {total_batches} батчей по {batch_size} аннотаций")
    print(f"Максимум одновременных запросов: {max_concurrent}")
    print(f"Максимум повторных попыток: {max_retries}")
    
    tasks = []
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(abstracts))
        current_batch = abstracts[start_idx:end_idx]
        
        task = extract_keyphrases_batch_async(
            client, current_batch, start_idx, batch_idx, rate_limiter, 
            max_retries=max_retries
        )
        tasks.append(task)
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_task(task):
        async with semaphore:
            return await task
    
    bounded_tasks = [bounded_task(task) for task in tasks]
    
    print("\n🚀 Запуск параллельной обработки...")
    start_time = time.time()
    
    results = await asyncio.gather(*bounded_tasks, return_exceptions=True)
    
    end_time = time.time()
    
    successful_batches = 0
    failed_batches = 0
    retry_stats = {}
    
    for result in results:
        if isinstance(result, Exception):
            print(f"Критическое исключение в одном из батчей: {result}")
            failed_batches += 1
            continue
            
        batch_idx, parsed_result = result
        if parsed_result:
            successful_batches += 1
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(abstracts))
            
            for i, idx in enumerate(range(start_idx, end_idx)):
                key = f"annotation_{idx + 1}"
                if key in parsed_result:
                    keyphrases_list = parsed_result[key]
                    df.at[idx, 'keyphrases'] = keyphrases_list
        else:
            failed_batches += 1
    
    print(f"\n=== 📊 ДЕТАЛЬНАЯ СТАТИСТИКА ===")
    print(f"✅ Успешно обработано: {successful_batches}/{total_batches} батчей")
    print(f"❌ Не удалось обработать: {failed_batches}/{total_batches} батчей")
    print(f"📈 Процент успеха: {successful_batches/total_batches*100:.1f}%")
    print(f"\n⏱️  Обработка завершена за {end_time - start_time:.1f} секунд")
    
    return df