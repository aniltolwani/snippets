import time
import random
import functools
from typing import List
import torch

def timeit(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.3f} seconds")
        return result

    return new_func

def salt(s: str) -> str:
    """Add random noise to text by trimming random amounts from start and end."""
    a = min(random.randrange(0,10), len(s)+1)
    b = min(random.randrange(0,10), len(s)-a+1)
    return s[a:len(s)-b]

def salt_list(l: List[str]) -> List[str]:
    """Apply salt function to a list of strings."""
    return [salt(s) for s in l]

def collate_fn(batch):
    """Collate function for DataLoader."""
    positives, negatives = zip(*batch)

    collated_positives = torch.stack([item["input_ids"] for item in positives])
    positive_attention_masks = torch.stack(
        [item["attention_mask"] for item in positives]
    )

    nested_negatives = torch.stack(
        [torch.stack([neg["input_ids"] for neg in neg_list]) for neg_list in negatives]
    )
    collated_negatives = nested_negatives.view(-1, nested_negatives.size(-1))

    nested_negative_masks = torch.stack(
        [torch.stack([neg["attention_mask"] for neg in neg_list]) for neg_list in negatives]
    )
    negative_attention_masks = nested_negative_masks.view(-1, nested_negative_masks.size(-1))

    return {
        "positives": (collated_positives, positive_attention_masks),
        "negatives": (collated_negatives, negative_attention_masks),
    } 