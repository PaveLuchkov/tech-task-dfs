from typing import List
from typing import Dict
from google.genai import types
from .response_schema import create_response_schema
from .create_batch_prompt import create_batch_prompt

import time
import asyncio
import threading
from google.genai import types
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

class RateLimiter:
    """Rate limiter для контроля количества запросов в минуту"""
    
    def __init__(self, max_requests_per_minute: int = 15):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()
        self.total_requests = 0
        self.total_waits = 0
    
    async def acquire(self):
        """Ждет, пока не появится слот для запроса"""
        with self.lock:
            self.total_requests += 1
            now = time.time()
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests:
                sleep_time = 60 - (now - self.requests[0]) + 0.1
                self.total_waits += 1
                print(f"⏳ Rate limit: ожидание {sleep_time:.1f}с (запрос #{self.total_requests})")
                await asyncio.sleep(sleep_time)
                return await self.acquire()
            
            self.requests.append(now)

async def extract_keyphrases_batch_async(client, abstracts_batch: List[str], 
                                       start_idx: int, batch_idx: int, 
                                       rate_limiter: RateLimiter,
                                       model: str = "gemini-2.5-flash",
                                       max_retries: int = 3) -> tuple[int, Optional[Dict]]:
    """Асинхронно извлекает ключевые фразы для батча аннотаций с повторными попытками"""
    
    prompt = create_batch_prompt(abstracts_batch, start_idx)
    schema = create_response_schema(len(abstracts_batch), start_idx)
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        system_instruction=[
            types.Part.from_text(
                text="Ты эксперт по анализу научных текстов. Извлекай наиболее значимые ключевые фразы и термины, которые лучше всего характеризуют содержание каждой аннотации."
            ),
        ],
    )
    
    for attempt in range(max_retries + 1):
        try:
            await rate_limiter.acquire()
            
            print(f"Отправка батча {batch_idx + 1} (аннотации {start_idx + 1}-{start_idx + len(abstracts_batch)})" + 
                  (f" - попытка {attempt + 1}" if attempt > 0 else ""))
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                response = await loop.run_in_executor(
                    executor, 
                    lambda: client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=generate_content_config,
                    )
                )
            
            print(f"✓ Батч {batch_idx + 1} обработан успешно" + 
                  (f" (попытка {attempt + 1})" if attempt > 0 else ""))
            return batch_idx, response.parsed
            
        except Exception as e:
            error_msg = str(e)
            is_rate_limit_error = "429" in error_msg or "quota" in error_msg.lower()
            is_server_error = "503" in error_msg or "502" in error_msg or "overloaded" in error_msg.lower()
            is_retryable = is_rate_limit_error or is_server_error
            
            if attempt < max_retries and is_retryable:
                delay = (2 ** attempt) * 5
                print(f"⚠️  Ошибка в батче {batch_idx + 1} (попытка {attempt + 1}): {e}")
                print(f"   Повтор через {delay} секунд...")
                await asyncio.sleep(delay)
                continue
            else:
                print(f"✗ Батч {batch_idx + 1} окончательно не обработан после {attempt + 1} попыток: {e}")
                return batch_idx, None
    
    return batch_idx, None