import os
import json
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.optim as optim

from src.model import ParagraphTransformerModel, ContrastiveLoss
from src.dataset import ParagraphPairsDataset
from src.utils import salt_list, collate_fn

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Device setup
    if not torch.cuda.is_available():
        print("No GPU available!")
        exit()
    
    device = torch.device("cuda:0")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.transformer_model)
    model = ParagraphTransformerModel(
        transformer_model=cfg.transformer_model,
        tokenizer=tokenizer,
        embedding_dim=cfg.embedding_dim,
    ).to(device)
    model = DataParallel(model)

    # Load data
    paragraphs = []
    paragraph_counts = []
    references = []
    reference_counts = []

    # Load and process data
    for dataset_path in cfg.dataset_paths:
        files = os.listdir(dataset_path)[:cfg.max_files]
        num_files_to_use = int(len(files) * cfg.train_percentage)
        for file in tqdm(files[:num_files_to_use]):
            if file.endswith('.jsonl'):
                with open(os.path.join(dataset_path, file), "r") as f:
                    line = f.readline().strip()
                    if line:
                        article = json.loads(line)
                        paragraphs.append(article.get("paragraphs", []))
                        paragraph_counts.append(article.get("num_paragraphs", 0))
                        references.append(article.get("references", []))
                        reference_counts.append(article.get("num_references", 0))

    # Apply data augmentation
    paragraphs = [salt_list(p) for p in tqdm(paragraphs)]
    references = [salt_list(r) for r in tqdm(references)]

    # Create datasets
    dataset = ParagraphPairsDataset(
        paragraph_counts=paragraph_counts,
        query_count=cfg.num_queries,
        neg_in_article=cfg.num_negative_examples,
        paragraphs=paragraphs,
        references=references,
        reference_counts=reference_counts,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )

    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    # Training setup
    criterion = ContrastiveLoss(tau_initial=cfg.tau_initial)
    optimizer = optim.AdamW(model.parameters(), 
                           lr=cfg.learning_rate, 
                           weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2, verbose=True)
    writer = SummaryWriter()

    # Training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(cfg.num_epochs):
        # Training
        model.train()
        for i, batch in enumerate(train_dataloader):
            positives, positive_masks = batch["positives"]
            negatives, negative_masks = batch["negatives"]
            
            positives = positives.to(device)
            positive_masks = positive_masks.to(device)
            negatives = negatives.to(device)
            negative_masks = negative_masks.to(device)

            positive_embeddings = model(positives, attention_mask=positive_masks)
            negative_embeddings = model(negatives, attention_mask=negative_masks)
            
            loss = criterion(positive_embeddings, negative_embeddings)
            loss.backward()
            
            if (i + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                positives, positive_masks = batch["positives"]
                negatives, negative_masks = batch["negatives"]
                
                positives = positives.to(device)
                positive_masks = positive_masks.to(device)
                negatives = negatives.to(device)
                negative_masks = negative_masks.to(device)

                positive_embeddings = model(positives, attention_mask=positive_masks)
                negative_embeddings = model(negatives, attention_mask=negative_masks)
                
                loss = criterion(positive_embeddings, negative_embeddings)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save(
                model.module.state_dict(),
                f"model_{cfg.output_suffix}_best.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.patience:
            print("Early stopping triggered.")
            break

    writer.close()

if __name__ == "__main__":
    main() 