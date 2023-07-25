import torch
import pandas as pd
from typing import Union, List, Optional
from .layers.transformers import TransformerEncoder, ATTENTION_TYPE, ScaledDotProductAttention
from .layers.positional_encoding import SinusoidalPositionalEncoding, POSITIONAL_ENCODING_TYPE
from .layers import Dropout
from ._conversions import strings_to_tensor, tensor_to_classes, tensor_to_probabilities
from ._conversions import classes_to_tensor
from ._neural_network import NeuralNetworkClassifier
from ._loss_functions import cross_entropy
from pygmalion.tokenizers._utilities import Tokenizer


class TextClassifier(NeuralNetworkClassifier):

    def __init__(self, classes: List[str],
                 tokenizer: Tokenizer,
                 n_stages: int, projection_dim: int, n_heads: int,
                 activation: str = "relu",
                 dropout: Union[float, None] = None,
                 mask_padding: bool = True,
                 gradient_checkpointing: bool = True,
                 positional_encoding_type: Optional[POSITIONAL_ENCODING_TYPE] = SinusoidalPositionalEncoding,
                 positional_encoding_kwargs: dict={},
                 attention_type: ATTENTION_TYPE = ScaledDotProductAttention,
                 attention_kwargs: dict = {}):
        """
        Parameters
        ----------
        classes : list of str
            the class names
        tokenizer : Tokenizer
            tokenizer of the input sentences
        n_stages : int
            number of stages in the encoder and decoder
        projection_dim : int
            dimension of a single attention head
        n_heads : int
            number of heads for the multi-head attention mechanism
        activation : str
            activation function
        dropout : float or None
            dropout probability if any
        mask_padding : bool
            If True, PAD tokens are masked in attention
        gradient_checkpointing : bool
            If True, uses gradient checkpointing to reduce memory usage during
            training at the expense of computation time.
        positional_encoding_type : POSITIONAL_ENCODING_TYPE or None
            type of absolute positional encoding
        positional_encoding_kwargs : dict
            additional kwargs passed to positional_encoding_type initializer
        attention_type : ATTENTION_TYPE
            type of attention for multi head attention
        attention_kwargs : dict
            additional kwargs passed to attention_type initializer
        """
        super().__init__(classes)
        self.mask_padding = mask_padding
        embedding_dim = projection_dim*n_heads
        self.tokenizer = tokenizer
        self.embedding = torch.nn.Embedding(self.tokenizer.n_tokens,
                                                  embedding_dim)
        self.dropout_input = Dropout(dropout)
        if positional_encoding_type is None:
            self.positional_encoding = None
        else:
            self.positional_encoding = positional_encoding_type(embedding_dim, **positional_encoding_kwargs)
        self.transformer_encoder = TransformerEncoder(n_stages, projection_dim, n_heads,
                                                      dropout=dropout, activation=activation,
                                                      attention_type=attention_type,
                                                      gradient_checkpointing=gradient_checkpointing,
                                                      **attention_kwargs)
        self.head = torch.nn.Linear(embedding_dim, len(self.classes))

    def forward(self, X: torch.Tensor):
        """
        performs the encoding part of the network

        Parameters
        ----------
        X : torch.Tensor
            tensor of longs of shape (N, L) with:
            * N : number of sentences
            * L : words per sentence

        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, C) with C the number of classes
        """
        X = X.to(self.device)
        padding_mask = (X == self.tokenizer.PAD) if self.mask_padding else None
        N, L = X.shape
        X = self.embedding(X)
        if self.positional_encoding is not None:
            X = self.positional_encoding(X)
        X = self.dropout_input(X.reshape(N*L, -1)).reshape(N, L, -1)
        X = self.transformer_encoder(X, padding_mask)
        return self.head(X.mean(dim=1))

    def loss(self, x, y_target, weights=None, class_weights=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            tensor of long of shape (N, L)
        y_target : torch.Tensor
            tensor of long of shape (N,)
        """
        x, y_target = x.to(self.device), y_target.to(self.device)
        y_pred = self(x)
        return cross_entropy(y_pred, y_target, weights, class_weights)

    @property
    def device(self) -> torch.device:
        return self.head.weight.device

    def _x_to_tensor(self, x: List[str],
                     device: Optional[torch.device] = None,
                     max_input_sequence_length: Optional[int] = None,
                     raise_on_longer_sequences: bool = False):
        return strings_to_tensor(x, self.tokenizer, device,
                                 max_sequence_length=max_input_sequence_length,
                                 raise_on_longer_sequences=raise_on_longer_sequences,
                                 add_start_end_tokens=False)

    def _y_to_tensor(self, y: List[str],
                     device: Optional[torch.device] = None) -> torch.Tensor:
        return classes_to_tensor(y, self.classes, device=device)

    def _tensor_to_y(self, tensor: torch.Tensor) -> List[str]:
        return tensor_to_classes(tensor, self.classes)

    def _tensor_to_proba(self, tensor: torch.Tensor) -> pd.DataFrame:
        return tensor_to_probabilities(tensor, self.classes)
