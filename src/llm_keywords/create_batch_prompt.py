from typing import List

def create_batch_prompt(abstracts_batch: List[str], start_idx: int) -> str:
    """
    Создает промпт для батча аннотаций
    """
    prompt = "Извлеки 2-5 ключевых фраз/слов из каждой аннотации. Ключевые фразы должны отражать основную тематику и важные концепции.\n\n"
    
    for i, abstract in enumerate(abstracts_batch):
        prompt += f"Аннотация {start_idx + i + 1}:\n{abstract}\n\n"
    
    return prompt