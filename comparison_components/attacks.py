from typing import Callable, Dict

import random


def apply_paraphrase_attack(text: str, attack_strength: float = 0.3) -> str:
    words = text.split()
    if len(words) == 0:
        return text
    num_to_replace = max(1, int(len(words) * attack_strength))
    indices = random.sample(range(len(words)), min(num_to_replace, len(words)))
    replacements = [
        "really", "very", "extremely", "quite", "rather",
        "thing", "fact", "matter", "aspect", "element",
        "good", "great", "excellent", "nice", "fine",
        "way", "method", "approach", "technique", "process",
        "help", "support", "aid", "assist", "facilitate",
    ]
    for idx in indices:
        words[idx] = random.choice(replacements)
    return " ".join(words)


def apply_shuffle_attack(text: str, attack_strength: float = 0.3) -> str:
    sentences = text.split(".")
    if len(sentences) == 0:
        return text
    num_shuffle = max(1, int(len(sentences) * attack_strength))
    indices = random.sample(range(len(sentences)), min(num_shuffle, len(sentences)))
    for idx in indices:
        words = sentences[idx].split()
        if len(words) > 1:
            random.shuffle(words)
            sentences[idx] = " ".join(words)
    return ".".join(sentences)


def apply_deletion_attack(text: str, attack_strength: float = 0.3) -> str:
    words = text.split()
    if len(words) == 0:
        return text
    num_to_delete = max(1, int(len(words) * attack_strength))
    indices = sorted(random.sample(range(len(words)), min(num_to_delete, len(words))), reverse=True)
    for idx in indices:
        if idx < len(words):
            words.pop(idx)
    return " ".join(words)


def get_attack_registry() -> Dict[str, Dict[str, Callable]]:
    return {
        "paraphrase": {"strength": 0.3, "func": apply_paraphrase_attack},
        "shuffle": {"strength": 0.5, "func": apply_shuffle_attack},
        "deletion": {"strength": 0.1, "func": apply_deletion_attack},
    }
