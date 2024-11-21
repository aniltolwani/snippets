from dataclasses import dataclass, field
from typing import List
from pathlib import Path

@dataclass
class TrainingConfig:
    """Configuration for training the paragraph transformer model"""
    
    # Data parameters
    max_files: int = 3000
    train_percentage: float = 1.0
    dataset_paths: List[str] = field(default_factory=lambda: ["./cleaned_paragraph_data"])
    
    # Model parameters
    transformer_model: str = "distilbert-base-uncased"
    embedding_dim: int = 768
    num_queries: int = 2
    num_negative_examples: int = 2
    dropout: float = 0.1
    use_gradient_checkpointing: bool = False
    
    # Training parameters
    learning_rate: float = 0.01
    weight_decay: float = 0.01
    num_epochs: int = 20
    patience: int = 3
    tau_initial: float = -3.0
    batch_size: int = 32
    num_workers: int = 4
    max_length: int = 512
    gradient_accumulation_steps: int = 4
    
    # Checkpointing
    output_suffix: str = "drop_small_original"
    checkpoint_dir: Path = Path("./checkpoints")
    
    # Evaluation
    accuracy_frequency: int = 5000
    accuracy_trials: int = 1000
    
    def __post_init__(self):
        """Create checkpoint directory if it doesn't exist"""
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)