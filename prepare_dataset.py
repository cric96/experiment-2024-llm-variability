import argparse
import logging
from datasets import load_dataset
from core import remove_sensitive_rows
# Utils
from core import store_pandas_in
# LLM interactions
from core.utils import OpenAiService

# argparser set, it accepts the rows cache file
parser = argparse.ArgumentParser(description='Prepare the dataset')
parser.add_argument('--rows_cache_babbage',
                    type=str,
                    help='The cache file for the rows to remove for safety concern (babbage)',
                    default="resources/filtered/rows_to_remove_babbage.json")
parser.add_argument('--rows_cache_got',
                    type=str,
                    help='The cache file for the rows to remove for safety concern (gpt35)',
                    default="resources/filtered/rows_to_remove_gpt35.json")
# config for the open ai services
parser.add_argument("--babbage_config", type=str,
                    help="The file with configuration for the Babbage service",
                    default="babbage.json"
                    )
parser.add_argument("--gpt35_config", type=str,
                    help="The file with configuration for the GPT-3.5 service",
                    default="gpt35.json"
                    )

args = parser.parse_args()

babbage = OpenAiService.from_file("resources/services", args.babbage_config)
gpt35 = OpenAiService.from_file("resources/services", args.gpt35_config)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    logging.warning("Loading dataset")
    dataset = load_dataset("Hello-SimpleAI/HC3", name='all')
    dataset = dataset['train'].to_pandas()

    logging.warning(f"Dataset loaded with {len(dataset)} rows")
    dataset["human_answers_length"] = dataset["human_answers"].apply(lambda x: len(x))
    dataset["chatgpt_answers_length"] = dataset["chatgpt_answers"].apply(lambda x: len(x))
    dataset_humans = dataset[dataset["human_answers_length"] >= 3]
    dataset_humans = dataset_humans[dataset_humans["chatgpt_answers_length"] >= 0].copy()

    logging.warning(f"Dataset filtered with {len(dataset_humans)} rows (>=3 human answers and >=0 chatgpt answers)")
    logging.warning("Removing sensitive rows (babbage)")
    rows_to_remove_babbage = remove_sensitive_rows(dataset_humans, args.rows_cache_babbage, babbage)
    logging.warning(f"Rows to remove (babbage): {len(rows_to_remove_babbage)}")
    dataset_humans = dataset_humans.drop(rows_to_remove_babbage)
    logging.warning(f"Dataset filtered with {len(dataset_humans)} rows (babbage)")
    logging.warning("Storing the filtered dataset")
    store_pandas_in(dataset_humans, "resources/filtered/dataset_humans")

    logging.warning("Sample of the dataset")
    sampled = dataset_humans[dataset_humans.chatgpt_answers_length > 0].sample(1100, random_state=42)
    logging.warning("Removing sensitive rows (gpt35)")
    rows_to_remove_gpt35 = remove_sensitive_rows(sampled, args.rows_cache_got, gpt35)

    logging.warning(f"Rows to remove (gpt35): {len(rows_to_remove_gpt35)}")
    sampled = sampled.reset_index().drop(rows_to_remove_gpt35)
    logging.warning(f"Dataset filtered with {len(sampled)} rows (gpt35)")
    # Store the dataset (1000 samples)
    sampled = sampled[:1000]
    logging.warning(f"Storing the filtered dataset {len(sampled)} rows")
    store_pandas_in(sampled, "resources/filtered/sampled_questions_extended")
