"""
This script is used to produced embeddings of question for a given model and dataset.
It accepts the file in which load the embeddings, the service to used and the dataset to load
"""
import argparse
import logging
import os
import numpy as np
import pandas as pd
from core.utils.llm import LlmService
from core.utils import embed_questions_if_not_cached
from core.charting import pca_chart, tsne_chart, umap_chart

argparser = argparse.ArgumentParser(description='Embed questions')
argparser.add_argument('--dataset',
                       type=str,
                       help='The dataset to load',
                       default="resources/datasets/dataset_humans_cleaned.pkl")
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
    # directory from file
    service = LlmService.from_file(os.path.dirname(args.service), os.path.basename(args.service))
    dataset = pd.read_pickle(args.dataset)
    embeddings = np.array(embed_questions_if_not_cached(service, dataset, args.embeddings_file))
    dir = os.path.dirname(args.embeddings_file)
    name = os.path.basename(args.embeddings_file).split(".")[0]
    os.makedirs(f"charts/embedding/{name}", exist_ok=True)
    logging.warning("PCA chart")
    pca_chart(embeddings, f"charts/embedding/{name}/pca.pdf")
    logging.warning("TSN chart")
    tsne_chart(embeddings, f"charts/embedding/{name}/tsn.pdf")
    logging.warning("UMAP chart")
    umap_chart(embeddings, f"charts/embedding/{name}/umap.pdf")
