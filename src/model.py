import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedTokenizerFast
from typing import Optional, Union, Dict, Any

class ParagraphTransformerModel(nn.Module):
    """
    A specialized bi-encoder transformer model for generating paragraph embeddings.
    
    This model uses a pre-trained transformer backbone with additional architectural 
    optimizations for paragraph-level semantic understanding:
    
    Architecture:
    - Transformer Base: Pretrained transformer (default: DistilBERT)
    - Pooling: CLS token pooling with attention masking
    - Projection Head: Linear layer with normalized outputs
    - Dropout: Optional dropout for regularization
    
    The model is designed for contrastive learning with:
    - Efficient gradient flow through residual connections
    - L2 normalized embeddings for stable training
    - Optional gradient checkpointing for memory efficiency
    
    Args:
        transformer_model (str): HuggingFace model identifier
        tokenizer (PreTrainedTokenizerFast): Associated tokenizer
        embedding_dim (int): Output embedding dimension
        dropout (float, optional): Dropout rate. Defaults to 0.1
        use_gradient_checkpointing (bool, optional): Enable gradient checkpointing. Defaults to False
    
    Attributes:
        transformer (AutoModel): Backbone transformer model
        fc (nn.Linear): Projection head
        dropout (nn.Dropout): Dropout layer
        layer_norm (nn.LayerNorm): Layer normalization
    """

    def __init__(
        self,
        transformer_model: str,
        tokenizer: PreTrainedTokenizerFast,
        embedding_dim: int,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.transformer = AutoModel.from_pretrained(transformer_model)
        self.transformer.resize_token_embeddings(len(tokenizer))
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()

        # Additional layers for robust embedding generation
        hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_length]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_length]
            return_hidden_states (bool, optional): Return transformer hidden states. Defaults to False

        Returns:
            Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: 
                - If return_hidden_states=False: Normalized embeddings [batch_size, embedding_dim]
                - If return_hidden_states=True: Tuple of (embeddings, hidden_states)
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden_states,
        )

        # Get CLS token embedding with attention masking
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        
        # Project to embedding dimension
        embeddings = self.fc(cls_embedding)
        embeddings = self.layer_norm(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        if return_hidden_states:
            return embeddings, outputs.hidden_states
        return embeddings

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Utility method for encoding text into embeddings.

        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            normalize (bool, optional): Apply L2 normalization. Defaults to True

        Returns:
            torch.Tensor: Encoded embeddings
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(input_ids, attention_mask)
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class ContrastiveLoss(nn.Module):
    """
    Implementation of InfoNCE/NT-Xent loss for contrastive learning.
    
    This loss function pulls positive pairs together while pushing apart negative pairs
    in the embedding space. It includes:
    - Learned temperature parameter for similarity scaling
    - Support for multiple negative samples per positive pair
    - Numerically stable computation
    
    The loss is computed as:
    L = -log(exp(sim(q,d+)/τ) / Σ(exp(sim(q,d)/τ)))
    
    where:
    - q: query embedding
    - d+: positive document embedding
    - d: all document embeddings (positive + negatives)
    - τ: learned temperature parameter
    - sim: similarity function (default: dot product)
    
    Args:
        tau_initial (float, optional): Initial temperature value. Defaults to 1.0
        similarity_function (callable, optional): Similarity function. Defaults to torch.matmul
    """

    def __init__(
        self,
        tau_initial: float = 1.0,
        similarity_function: callable = torch.matmul,
    ):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(tau_initial, dtype=torch.float32))
        self._sim = similarity_function

    def forward(
        self, 
        positive_embeddings: torch.Tensor, 
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the contrastive loss.

        Args:
            positive_embeddings (torch.Tensor): Query embeddings [batch_size, embedding_dim]
            negative_embeddings (torch.Tensor): Negative example embeddings [batch_size * num_negatives, embedding_dim]

        Returns:
            torch.Tensor: Scalar loss value
        """
        B, E = positive_embeddings.shape
        N = negative_embeddings.shape[0] // B

        # Computing similarity scores with temperature scaling
        similarity_scores = self._sim(
            positive_embeddings, 
            torch.cat([positive_embeddings, negative_embeddings], dim=0).T
        ) / self.tau.exp()
        
        assert similarity_scores.shape == (B, B * (N + 1)), \
            f"Shape mismatch in similarity scores. Expected ({B}, {B * (N + 1)}), got {similarity_scores.shape}"

        # Labels for positive pairs (diagonal entries)
        labels = torch.arange(B, device=similarity_scores.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_scores, labels)

        return loss 