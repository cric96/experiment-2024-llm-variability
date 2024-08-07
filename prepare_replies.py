"""
This script is used to generate replies for the bot.
It uses the configuration of LLM passed
It also leverage a cache to avoid to recompute the same replies
"""
import argparse
import json
import logging
import os
import time

import pandas
from datasets import tqdm

from core.utils import produce_response_for
from core.utils import services_loader
from core.utils import store_pandas_in

# argparser set, it accepts the rows cache file
parser = argparse.ArgumentParser(description='Prepare the dataset')
parser.add_argument("--dataset_file",
                    type=str,
                    help="The dataset file",
                    default="resources/datasets/sampled_questions_gpt-3.5.json")
parser.add_argument("--services_file",
                    type=str,
                    help="The file with the services configuration",
                    default="resources/replies/services.json")

parser.add_argument("--cache_file",
                    type=str,
                    help="The cache file for the rows to remove for safety concern",
                    default="resources/replies/data.json")

args = parser.parse_args()

if __name__ == "__main__":
    replies = {}
    if os.path.exists(args.cache_file):
        with open(args.cache_file, 'r') as f:
            replies = json.load(f)

    services = services_loader(args.services_file)
    dataset = pandas.read_json(args.dataset_file)

    for service in services:
        if service in replies:
            logging.warning(f"Already done: {service}")
            continue
        print(f"Policy: {service}")
        start = time.time()
        replies[service] = produce_response_for(llm_service=services[service], dataset=dataset, max_tokens=250,how_many=3)
        end = time.time()
        print(f"Time: {end - start}")
    # add human answers
    human_responses = []
    for i in tqdm(range(0, len(dataset))):
        data = dataset[i:i + 1]["human_answers"]
        responses = []
        for response in data.tolist()[0]:
            result = [response for _ in range(3)]
            responses.append(result)
        human_responses.append([i, responses])
    replies["human"] = human_responses

    cache_dir = os.path.dirname(args.cache_file)
    os.makedirs(cache_dir, exist_ok=True)
    # get the name without the extension
    cache_name = os.path.basename(args.cache_file).split(".")[0]
    with open(f"{cache_dir}/{cache_name}.pkl", 'wb') as f:
        pandas.to_pickle(replies, f)
    with open(f"{cache_dir}/{cache_name}.json", 'w') as f:
        json.dump(replies, f)

    # store in a new dataset
    copy = dataset.copy()
    for mode in replies:
        if mode == "human": ## we already have human answers
            continue
        all_adding = replies[mode]
        copy[f"{mode}_response"] = ""
        copy[f"{mode}_response"] = copy[f"{mode}_response"].astype(object)
        for row in replies[mode]:
            index = row[0]
            responses = row[1]
            copy.at[index, f"{mode}_response"] = responses
        logging.warning(f"Done: {mode}")
    where = "resources/datasets/"
    os.makedirs(os.path.dirname(where), exist_ok=True)
    store_pandas_in(copy, where + "dataset_with_llm_responses")