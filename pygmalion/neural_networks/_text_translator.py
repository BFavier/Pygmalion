import torch
import numpy as np
from typing import Union, List, Dict, Tuple, Optional, Literal
from .layers.transformers import TransformerEncoder, TransformerDecoder, ATTENTION_TYPE
from .layers import LearnedPositionalEncoding, SinusoidalPositionalEncoding
from ._conversions import sentences_to_tensor, tensor_to_sentences
from ._conversions import floats_to_tensor
from ._neural_network import NeuralNetwork
from ._loss_functions import cross_entropy
from pygmalion.tokenizers._utilities import Tokenizer, SpecialToken


class TextTranslator(NeuralNetwork):

    def __init__(self, tokenizer_input: Tokenizer,
                 tokenizer_output: Tokenizer,
                 n_stages: int, projection_dim: int, n_heads: int,
                 activation: str = "relu",
                 dropout: Union[float, None] = None,
                 positional_encoding_type: Literal["sinusoidal", "learned", None] = "sinusoidal",
                 mask_padding: bool = True,
                 attention_type: ATTENTION_TYPE = "scaled dot product",
                 RPE_radius: Optional[int] = None,
                 input_sequence_length: Optional[int] = None,
                 output_sequence_length: Optional[int] = None,
                 low_memory: bool = True):
        """
        Parameters
        ----------
        ...
        """
        super().__init__()
        self.mask_padding = mask_padding
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        embedding_dim = projection_dim*n_heads
        self.tokenizer_input = tokenizer_input
        self.tokenizer_output = tokenizer_output
        self.embedding_input = torch.nn.Embedding(self.tokenizer_input.n_tokens,
                                                  embedding_dim)
        self.embedding_output = torch.nn.Embedding(self.tokenizer_output.n_tokens,
                                                embedding_dim)
        self.dropout_input = torch.nn.Dropout(dropout) if dropout is not None else None
        self.dropout_output = torch.nn.Dropout(dropout) if dropout is not None else None
        if positional_encoding_type == "sinusoidal":
            self.positional_encoding_input = SinusoidalPositionalEncoding()
            self.positional_encoding_output = SinusoidalPositionalEncoding()
        elif positional_encoding_type == "learned":
            assert input_sequence_length is not None and output_sequence_length is not None
            self.positional_encoding_input = LearnedPositionalEncoding(input_sequence_length, embedding_dim)
            self.positional_encoding_output = LearnedPositionalEncoding(output_sequence_length, embedding_dim)
        elif positional_encoding_type is None:
            self.positional_encoding_input = None
            self.positional_encoding_output = None
        else:
            raise ValueError(f"Unexpected positional encoding type '{positional_encoding_type}'")
        self.transformer_encoder = TransformerEncoder(n_stages, projection_dim, n_heads,
                                                      dropout=dropout, activation=activation,
                                                      RPE_radius=RPE_radius, attention_type=attention_type,
                                                      low_memory=low_memory)
        self.transformer_decoder = TransformerDecoder(n_stages, projection_dim, n_heads,
                                                      dropout=dropout, activation=activation,
                                                      RPE_radius=RPE_radius, attention_type=attention_type,
                                                      low_memory=low_memory)
        self.head = torch.nn.Linear(embedding_dim, self.tokenizer_output.n_tokens)

    def forward(self, X: torch.Tensor, padding_mask: Optional[torch.Tensor]):
        return self.encode(X, padding_mask)

    def encode(self, X: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        performs the encoding part of the network

        Parameters
        ----------
        X : torch.Tensor
            tensor of longs of shape (N, L) with:
            * N : number of sentences
            * L : words per sentence
        padding_mask : torch.Tensor or None
            tensor of booleans of shape (N, L)

        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, L, D) with D the embedding dimension
        """
        X = X.to(self.device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)
        N, L = X.shape
        X = self.embedding_input(X)
        X = self.positional_encoding_input(X)
        if self.dropout_input is not None:
            X = self.dropout_input(X.reshape(N*L, -1)).reshape(N, L, -1)
        X = self.transformer_encoder(X, padding_mask)
        return X

    def decode(self, Y: torch.Tensor, encoded: torch.Tensor, encoded_padding_mask: Optional[torch.Tensor]):
        """
        performs the decoding part of the network

        Parameters
        ----------
        Y : torch.Tensor
            tensor of long of shape (N, Ly) with:
            * N : number of sentences
            * Ly : words per sentence in the output language
        encoded : torch.Tensor
            tensor of floats of shape (N, Lx, D) with:
            * N : number of sentences
            * Lx : words per sentence in the input language
            * D : embedding dim
        encoded_padding_mask : torch.Tensor or None
            tensor of booleans of shape (N, L)

        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, Ly, D)
        """
        N, L = Y.shape
        Y = self.embedding_output(Y)
        Y = self.positional_encoding_output(Y)
        if self.dropout_output is not None:
            Y = self.dropout_output(Y.reshape(N*L, -1)).reshape(N, L, -1)
        Y = self.transformer_decoder(Y, encoded, encoded_padding_mask)
        return self.head(Y)

    def loss(self, x, y_target, weights=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            tensor of long of shape (N, Li)
        y_target : torch.Tensor
            tensor of long of shape (N, Lt)
        """
        x, y_target = x.to(self.device), y_target.to(self.device)
        class_weights = torch.ones(self.tokenizer_output.n_tokens, device=self.device)
        class_weights[self.tokenizer_output.PAD] = 0.
        padding_mask = (x == self.tokenizer_input.PAD) if self.mask_padding else None
        encoded = self(x, padding_mask)
        y_pred = self.decode(y_target[:, :-1], encoded, padding_mask)
        return cross_entropy(y_pred.transpose(1, 2), y_target[:, 1:],
                             weights, class_weights)

    def predict(self, sequences: List[str], max_tokens: int = 100,
                n_beams: int = 1) -> List[str]:
        """
        Predict a translation for the given sequences using beam search,
        outputing at most 'max_tokens' tokens.
        If 'n_beams' is 1, this is equivalent to predicting the single token
        with the highest likelyhood at each step.
        """
        self.eval()
        with torch.no_grad():
            X = self._x_to_tensor(sequences, self.device)
            START = self.tokenizer_output.START
            END = self.tokenizer_output.END
            PAD = self.tokenizer_input.PAD
            n_classes = self.tokenizer_output.n_tokens
            encoded_padding_mask = (X == PAD) if self.mask_padding else None
            encoded = self(X, encoded_padding_mask)
            N, _, D = encoded.shape
            encoded_expanded = encoded.unsqueeze(1).repeat(1, n_beams, 1, 1).reshape(N*n_beams, -1, D)
            predicted = torch.zeros((N, 1, 0), device=X.device, dtype=torch.long)
            log_likelyhood = torch.zeros((N, 1), device=X.device, dtype=torch.float)
            n_predicted_tokens = torch.zeros((N, 1), device=X.device, dtype=torch.long)
            intermediate = [torch.zeros((N, 0, D), device=X.device)
                            for _ in self.transformer_encoder.stages]
            I = torch.full([N, 1], START,
                           dtype=torch.long, device=X.device)
            for i in range(max_tokens):
                stop = (predicted == END).any(dim=-1)
                if stop.all():
                    break
                Q = self.embedding_output(I)
                if self.positional_encoding_output is not None:
                    Q = self.positional_encoding_output(Q)
                intermediate, Q = self.transformer_decoder.predict(
                    intermediate, Q, encoded, encoded_padding_mask, i)
                # lookup the beam/token that lead to highest mean log likelyhood
                log_p = torch.log(torch.softmax(self.head(Q.reshape(N, -1, D)), dim=-1))
                all_log_likelyhoods = log_likelyhood.unsqueeze(-1) + torch.masked_fill(log_p, stop.unsqueeze(-1), 0.)
                n_predicted_tokens = n_predicted_tokens + (~stop)
                mean_log_likelyhoods = all_log_likelyhoods / n_predicted_tokens.unsqueeze(-1)
                _, indexes = mean_log_likelyhoods.reshape(N, -1).topk(k=n_beams, dim=-1)
                beam, token = torch.div(indexes, n_classes, rounding_mode="floor"), indexes % n_classes
                # create the property of the new beams
                I = token.reshape(N*n_beams, 1)
                intermediate = [torch.gather(inter.reshape(N, predicted.shape[1], -1, D),
                                             1,
                                             beam.reshape(N, n_beams, 1, 1).expand(-1, -1, inter.shape[1], D)
                                             ).reshape(N*n_beams, -1, D)
                                for inter in intermediate]
                predicted = torch.gather(predicted, 1, beam.unsqueeze(-1).expand(-1, -1, predicted.shape[-1]))
                predicted = torch.cat([predicted, token.unsqueeze(-1)], dim=-1)
                if all_log_likelyhoods.shape[1] != n_beams:
                    all_log_likelyhoods = all_log_likelyhoods.expand(-1, n_beams, -1)
                log_likelyhood = torch.gather(all_log_likelyhoods, -1, indexes.unsqueeze(-1)).squeeze(-1)
                n_predicted_tokens = torch.gather(n_predicted_tokens, 1, beam)
                encoded = encoded_expanded
            # get best final beam
            predicted = predicted[:, 0, :]
            translations = [self.tokenizer_output.decode(p.cpu().tolist()) for p in predicted]
            return translations

    @property
    def device(self) -> torch.device:
        return self.head.weight.device

    def _x_to_tensor(self, x: List[str],
                     device: Optional[torch.device] = None):
        return sentences_to_tensor(x, self.tokenizer_input, device,
                                    max_sequence_length=self.input_sequence_length,
                                    add_start_end_tokens=False)

    def _y_to_tensor(self, y: List[str],
                     device: Optional[torch.device] = None) -> torch.Tensor:
        return sentences_to_tensor(y, self.tokenizer_output, device,
                                   max_sequence_length=self.output_sequence_length,
                                   add_start_end_tokens=True)

    def _tensor_to_y(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor_to_sentences(tensor, self.tokenizer_output)
