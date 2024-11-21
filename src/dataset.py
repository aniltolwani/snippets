import random
from typing import List, Dict, Tuple, Callable
from itertools import chain
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast, BatchEncoding

TokenizedParagraph = Dict[str, torch.Tensor]
ItemOutput = Tuple[TokenizedParagraph, List[TokenizedParagraph]]
Coordinate = Tuple[int, int]

class ParagraphPairsDataset(Dataset):
    """
    Dataset for handling paragraph pairs for contrastive training in neural search.
    Manages pairs of paragraphs where each pair consists of a query and document paragraph.
    """

    def __init__(
        self,
        paragraph_counts: List[int],
        query_count: int,
        neg_in_article: int,
        paragraphs: List[List[str]],
        references: List[List[str]],
        reference_counts: List[int],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
    ):
        self.paragraph_counts = paragraph_counts
        self.reference_counts = reference_counts
        self.query_count = query_count
        self.neg_in_article = neg_in_article

        self.query_paragraphs: List[Coordinate] = []
        self.document_paragraphs_positive: List[Coordinate] = []
        self.document_paragraphs_negative: List[List[Coordinate]] = []

        self.paragraphs = [p + r for p,r in zip(paragraphs, references)]

        self.separator_token = tokenizer.sep_token or "[SEP]"
        self.tokenize: Callable[[str, str], BatchEncoding] = lambda query, doc: tokenizer(
            query,
            doc,
            padding="max_length",
            truncation="longest_first",
            max_length=max_length,
            return_tensors="pt",
        )
        
        self._populate_query_paragraphs()
        self._populate_document_paragraphs_positive()
        self._populate_document_paragraphs_negative()

    def _populate_query_paragraphs(self) -> None:
        self.query_paragraphs = [
            (example_index, query_index)
            for example_index, paragraph_count in enumerate(self.paragraph_counts)
            for query_index in random.sample(
                population=range(paragraph_count - 1),
                k=min(self.query_count, paragraph_count - 1),
            )
        ]

    def _populate_document_paragraphs_positive(self) -> None:
        self.document_paragraphs_positive = [
            (example_index, query_index + 1)
            for example_index, query_index in self.query_paragraphs
        ]

    def _populate_document_paragraphs_negative(self) -> None:
        def generate_negatives(example_index, query_index):
            valid_range = list(
                chain(
                    range(query_index + 1),
                    range(
                        query_index + 2, 
                        self.paragraph_counts[example_index]
                    ),
                )
            )
            normal_negatives = [
                (example_index, paragraph_index)
                for paragraph_index in random.choices(
                    population=valid_range,
                    k=self.neg_in_article,
                )
            ]
            ref_negatives = [
                (example_index, self.paragraph_counts[example_index] + 
                 random.randrange(0, self.reference_counts[example_index]))
                if self.reference_counts[example_index] > 0
                else (example_index, random.choice(valid_range))
            ]

            return normal_negatives + ref_negatives

        self.document_paragraphs_negative = [
            generate_negatives(example_index, query_index)
            for example_index, query_index in self.query_paragraphs
        ]

    def _extract_at_coordinate(self, paragraph_coordinate: Coordinate) -> str:
        return self.paragraphs[paragraph_coordinate[0]][paragraph_coordinate[1]]

    def _tokenize_paragraph(self, query: str, document: str) -> TokenizedParagraph:
        return {k: v.squeeze(0) for k, v in self.tokenize(query, document).items()}

    def __len__(self):
        return len(self.query_paragraphs)

    def __getitem__(self, index: int) -> ItemOutput:
        query_text = self._extract_at_coordinate(self.query_paragraphs[index])
        positive_text = self._extract_at_coordinate(
            self.document_paragraphs_positive[index]
        )
        negative_texts = [
            self._extract_at_coordinate(coordinate)
            for coordinate in self.document_paragraphs_negative[index]
        ]

        return (
            self._tokenize_paragraph(query_text, positive_text),
            [self._tokenize_paragraph(query_text, text) for text in negative_texts],
        )

class EvaluationDataset(Dataset):
    def __init__(self, val_dataset, negatives_pool, sample_size=99):
        self.val_dataset = val_dataset
        self.negatives_pool = negatives_pool
        self.sample_size = sample_size

    def __len__(self):
        return len(self.val_dataset)

    def __getitem__(self, idx):
        query, positive, _ = self.val_dataset[idx]

        potential_negatives = [
            item
            for item in self.negatives_pool
            if not torch.equal(item["input_ids"], positive["input_ids"])
        ]
        negatives = random.sample(potential_negatives, self.sample_size)

        return (query, positive, negatives) 