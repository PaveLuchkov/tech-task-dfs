def evaluate_search_engine(search_engine, queries, ground_truth, top_n=5, **search_kwargs):
    metrics = {
        "mrr": [],         # Mean Reciprocal Rank
        "precision_at_5": [] # Точность на 5 результатах
    }

    for query, relevant_docs in ground_truth.items():
        # Передаем и top_n, и все остальные специфичные аргументы
        search_results = search_engine.search(query, top_n=top_n, **search_kwargs)
        
        # ВАЖНО: TF-IDF возвращает (индекс, текст), а Embedding (индекс, текст, скор)
        # Унифицируем результат
        if len(search_results[0]) == 3: # Если есть score
             retrieved_ids = [idx for idx, text, score in search_results]
        else:
             retrieved_ids = [idx for idx, text in search_results]

        # Считаем Reciprocal Rank для текущего запроса
        rank = 0
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_docs:
                rank = 1 / i
                break
        metrics["mrr"].append(rank)

        # Считаем Precision@5 для текущего запроса
        hits = len(set(retrieved_ids) & set(relevant_docs))
        precision = hits / 5.0
        metrics["precision_at_5"].append(precision)

    # Усредняем метрики по всем запросам
    avg_mrr = sum(metrics["mrr"]) / len(metrics["mrr"])
    avg_p5 = sum(metrics["precision_at_5"]) / len(metrics["precision_at_5"])
    
    return {"MRR": avg_mrr, "Precision@5": avg_p5}