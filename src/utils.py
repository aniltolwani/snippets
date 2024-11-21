import time
import random
import functools
from typing import List
import torch
from tqdm import tqdm

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

def calculate_mrr_at_k(model, dataloader, device, k=1000):
    """Calculate Mean Reciprocal Rank at K."""
    print(f"\nCalculating MRR@{k}...")
    reciprocal_ranks = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            positives, positive_masks = batch["positives"]
            negatives, negative_masks = batch["negatives"]
            
            positives = positives.to(device)
            positive_masks = positive_masks.to(device)
            negatives = negatives.to(device)
            negative_masks = negative_masks.to(device)

            positive_embeddings = model(positives, attention_mask=positive_masks)
            negative_embeddings = model(negatives, attention_mask=negative_masks)
            
            # Compute rankings
            all_embeddings = torch.cat([positive_embeddings, negative_embeddings], dim=0)
            scores = torch.matmul(positive_embeddings, all_embeddings.T)
            rankings = (scores >= scores[:, 0].unsqueeze(1)).sum(1)
            
            # Calculate reciprocal ranks
            for rank in rankings:
                if rank <= k:
                    reciprocal_ranks.append(1.0 / rank.item())
                else:
                    reciprocal_ranks.append(0.0)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(f"MRR@{k}: {mrr:.4f}")
    return mrr

def quick_sanity_check(model, dataloader, device):
    """Quick sanity check using actual data."""
    with torch.no_grad():
        batch = next(iter(dataloader))
        positives, positive_masks = batch["positives"]
        negatives, negative_masks = batch["negatives"]
        
        positives = positives.to(device)
        positive_masks = positive_masks.to(device)
        negatives = negatives.to(device)
        negative_masks = negative_masks.to(device)

        positive_embeddings = model(positives, attention_mask=positive_masks)
        negative_embeddings = model(negatives, attention_mask=negative_masks)
        
        # Check if positive pairs score higher than negatives
        pos_scores = torch.matmul(positive_embeddings, positive_embeddings.T).diagonal()
        neg_scores = torch.matmul(positive_embeddings, negative_embeddings.T).mean(dim=1)
        
        passed = (pos_scores > neg_scores).float().mean().item() > 0.5
        print(f"\nSanity Check {'Passed' if passed else 'Failed'}")
        print(f"Avg Positive Score: {pos_scores.mean():.4f}")
        print(f"Avg Negative Score: {neg_scores.mean():.4f}")
        return passed