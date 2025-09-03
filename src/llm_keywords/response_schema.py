from google import genai

def create_response_schema(batch_size: int, start_idx: int) -> genai.types.Schema:
    """
    Создает схему ответа для батча
    """
    properties = {}
    for i in range(batch_size):
        key = f"annotation_{start_idx + i + 1}"
        properties[key] = genai.types.Schema(
            type=genai.types.Type.ARRAY,
            items=genai.types.Schema(
                type=genai.types.Type.STRING,
            ),
        )
    
    return genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties=properties,
    )