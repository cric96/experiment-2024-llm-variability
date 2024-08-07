import json
import logging
import os

from tqdm.auto import tqdm

from core.utils.llm import LlmService, OllamaService


def store_pandas_in(dataset, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    dataset.to_csv(filename + ".csv", index=False)
    dataset.to_json(filename + ".json")
    dataset.to_pickle(filename + ".pkl")


def remove_sensitive_rows(dataset, cache_file, service):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            rows_to_remove = json.load(f)
    else:
        end = len(dataset)
        rows_to_remove = []
        for chuck in tqdm(range(0, end)):
            (ok, error) = service.check(dataset["question"][chuck:chuck + 1].tolist())
            if not ok:
                rows_to_remove.append(chuck)
    return rows_to_remove


def embed_questions_if_not_cached(service: LlmService, dataset, embeddings_file: str):
    if os.path.exists(embeddings_file):
        logging.warning("Embeddings already computed")
        with open(embeddings_file, 'r') as f:
            return json.load(f)
    else:
        logging.warning("Computing embeddings")
        embeddings = []
        chuck = 100
        for i in tqdm(range(0, len(dataset), chuck)):
            current = service.embedChucks(dataset[i:i + chuck]['question'].tolist())
            embeddings.extend(current)
        with open(embeddings_file, 'w') as f:
            json.dump(embeddings, f)
        logging.warning(f"Embeddings computed and stored in {embeddings_file}")
        return embeddings


def produce_response_for(llm_service: LlmService, dataset, max_tokens=250, how_many=3):
    store = []  ## row, responses
    for f in tqdm(list(dataset.iterrows())):
        question = f[1]["question"]
        collect_replies = []
        try:
            for i in range(how_many):
                response_service = llm_service.complete(question, max_tokens)
                collect_replies.append(response_service)
        except Exception as e:
            print(f"Error: {e}")
            collect_replies = []
            continue
        store.append((f[0], collect_replies))
    return store

def services_loader(file):
    services = {}

    with open(file, 'r') as f:
        llms = json.load(f)
        for llm in llms:
            if "filename" in llms[llm] and "where" in llms[llm]:
                services[llm] = LlmService.from_file(llms[llm]["where"], llms[llm]["filename"])
            elif "model" in llms[llm]:
                services[llm] = OllamaService(llms[llm]["model"])
        return services
