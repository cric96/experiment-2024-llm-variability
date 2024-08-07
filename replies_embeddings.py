"""
This script is used to create the embeddings of the replies for all models and human dataset.
It uses mainly the replies data (json)
"""
import argparse
import json
import logging
import time
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from tqdm.auto import tqdm

from core.charting import pca_chart, umap_chart, tsne_chart
from core.utils.llm import LlmService
from core.utils import embed_questions_if_not_cached

argparser = argparse.ArgumentParser(description='Embed replies')
argparser.add_argument('--replies',
                       type=str,
                       help='The replies data',
                       default="resources/replies/data.json")
argparser.add_argument('--service',
                       type=str,
                       help='The service to use for the embeddings',
                       default="resources/services/text-embedding.json")
argparser.add_argument('--embeddings-file',
                       type=str,
                       help='The file to store the embeddings',
                       default="resources/embeddings/embeddings-openai.json")

args = argparser.parse_args()

if __name__ == "__main__":
    with open(args.replies, 'r') as f:
        replies = json.load(f)
    service = LlmService.from_file(os.path.dirname(args.service), os.path.basename(args.service))
    embeddings = {}
    if os.path.exists(args.embeddings_file):
        logging.warning("Embeddings already computed")
        with open(args.embeddings_file, 'r') as f:
            embeddings = json.load(f)
    else:
        for mode in replies:
            start = time.time()
            if mode in embeddings:
                logging.warning(f"Already done: {mode}")
                continue
            embeddings[mode] = []
            logging.warning("Processing: ", mode)
            for i in tqdm(range(0, len(replies[mode]))):
                responses = replies[mode][i][1]
                embeddings = []
                for response in responses:
                    result = embeddings.embed(response)[0]
                    embeddings.append(result)
                embeddings[mode].append(embeddings)
            end = time.time()
            logging.warning(len(embeddings[mode]))
            logging.warning(f"Time: {end - start}")
    ## plots
    logging.warning("Creating charts -- overall picture")
    modes_count = len(embeddings)
    fig, axs = plt.subplots(modes_count, 3, figsize=(20, 40))
    i = 0
    for mode in embeddings:
        print(f"Processing: {mode}")
        # flatten all the embedding
        flatten = [item for sublist in embeddings[mode] for item in sublist]
        pca_chart(flatten, None, alpha=0.1, title=f"PCA {mode}", axis=axs[i, 0])
        umap_chart(flatten, None, alpha=0.1, title=f"UMAP {mode}", axis=axs[i, 1])
        tsne_chart(np.array(flatten), None, alpha=0.1, title=f"TSNE {mode}", axis=axs[i, 2])
        i += 1
    # store figure

    plt.savefig("charts/embeddings/replies.pdf")
    # embeddings as one big picture
    all_classes = []
    all_embeddings = []
    for mode in embeddings:
        for i in range(0, len(embeddings[mode])):
            all_classes.extend([mode for _ in range(3)])
            all_embeddings.extend(embeddings[mode][i])
    # array of colors from the all_classes
    colors = cm.rainbow(np.linspace(0, 1, len(embeddings.keys())))
    array_classes = list(embeddings.keys())

    def index_from_class(c):
        return array_classes.index(c)


    color_classes = [colors[index_from_class(c)] for c in all_classes]
    logging.warning("Creating charts -- ensemble PCA")
    pca_chart(all_embeddings, where="charts/embeddings/replies_all_pca.pdf", alpha=0.5, size=5, labels=all_classes)
    logging.warning("Creating charts -- ensemble UMAP")
    umap_chart(all_embeddings, where="charts/embeddings/replies_all_umap.pdf", alpha=0.5, size=5, labels=all_classes)
    logging.warning("Creating charts -- ensemble TSNE")
    tsne_chart(all_embeddings, where="charts/embeddings/replies_all_tsne.pdf", alpha=0.5, size=5, labels=all_classes)
