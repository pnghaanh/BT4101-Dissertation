from typing import List

from datasets import IterableDataset, load_dataset


def load_wikitext(dataset_name: str, dataset_config_name: str, dataset_split: str):
    """Load and filter WikiText examples using the benchmark's canonical rules."""
    raw_dataset = load_dataset(
        dataset_name,
        dataset_config_name,
        split=dataset_split,
        streaming=False,
    )

    def wikitext_generator():
        for i, ex in enumerate(raw_dataset):
            if i % 2 != 0:
                continue
            if not ex["text"].strip():
                continue
            if ex["text"].strip().startswith("="):
                continue
            yield ex

    return IterableDataset.from_generator(wikitext_generator)


def build_wikitext_prompts(
    dataset_name: str = "wikitext",
    dataset_config_name: str = "wikitext-2-raw-v1",
    dataset_split: str = "train",
    n_prompts: int = 500,
) -> List[str]:
    dataset = load_wikitext(dataset_name, dataset_config_name, dataset_split)

    prompts = []
    for example in dataset:
        prompts.append(example["text"])
        if len(prompts) >= n_prompts:
            break
    return prompts
