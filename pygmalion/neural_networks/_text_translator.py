import torch
import numpy as np
from typing import Union, List, Tuple, Sequence, Optional
from itertools import count
from warnings import warn
from .layers.transformers import TransformerEncoder, TransformerDecoder, ATTENTION_TYPE, ScaledDotProductAttention
from .layers.positional_encoding import SinusoidalPositionalEncoding, POSITIONAL_ENCODING_TYPE
from .layers import Dropout, beam_search
from ._conversions import strings_to_tensor, tensor_to_strings
from ._conversions import floats_to_tensor
from ._neural_network import NeuralNetwork
from ._loss_functions import cross_entropy
from pygmalion.tokenizers._utilities import Tokenizer


class TextTranslator(NeuralNetwork):

    def __init__(self, tokenizer_input: Tokenizer, tokenizer_output: Tokenizer,
                 n_stages: int, projection_dim: int, n_heads: int,
                 activation: str = "relu",
                 dropout: Union[float, None] = None,
                 mask_padding: bool = True,
                 gradient_checkpointing: bool = True,
                 label_smoothing: float = 0.,
                 positional_encoding_type: Optional[POSITIONAL_ENCODING_TYPE] = SinusoidalPositionalEncoding,
                 input_positional_encoding_kwargs: dict={},
                 output_positional_encoding_kwargs: dict={},
                 attention_type: ATTENTION_TYPE = ScaledDotProductAttention,
                 attention_kwargs: dict = {}):
        """
        Parameters
        ----------
        
        tokenizer_input : Tokenizer
            tokenizer of the input sentences
        tokenizer_output : Tokenizer
            tokenizer of the output (target/predicted) sentences
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
        label_smoothing : float
            label smoothing level used in cross entropy loss
        positional_encoding_type : POSITIONAL_ENCODING_TYPE or None
            type of absolute positional encoding
        input_positional_encoding_kwargs : dict
            additional kwargs passed to positional_encoding_type initializer
        output_positional_encoding_kwargs : dict
            additional kwargs passed to positional_encoding_type initializer
        attention_type : ATTENTION_TYPE
            type of attention for multi head attention
        attention_kwargs : dict
            additional kwargs passed to attention_type initializer
        """
        super().__init__()
        self.mask_padding = mask_padding
        self.label_smoothing = label_smoothing
        embedding_dim = projection_dim*n_heads
        self.tokenizer_input = tokenizer_input
        self.tokenizer_output = tokenizer_output
        self.embedding_input = torch.nn.Embedding(self.tokenizer_input.n_tokens,
                                                  embedding_dim)
        self.embedding_output = torch.nn.Embedding(self.tokenizer_output.n_tokens,
                                                embedding_dim)
        self.dropout_input = Dropout(dropout)
        self.dropout_output = Dropout(dropout)
        if positional_encoding_type is None:
            self.positional_encoding_input = None
            self.positional_encoding_output = None
        else:
            self.positional_encoding_input = positional_encoding_type(embedding_dim, **input_positional_encoding_kwargs)
            self.positional_encoding_output = positional_encoding_type(embedding_dim, **output_positional_encoding_kwargs)
        self.transformer_encoder = TransformerEncoder(n_stages, projection_dim, n_heads,
                                                      dropout=dropout, activation=activation,
                                                      attention_type=attention_type,
                                                      gradient_checkpointing=gradient_checkpointing,
                                                      **attention_kwargs)
        self.transformer_decoder = TransformerDecoder(n_stages, projection_dim, n_heads,
                                                      dropout=dropout, activation=activation,
                                                      attention_type=attention_type,
                                                      gradient_checkpointing=gradient_checkpointing,
                                                      **attention_kwargs)
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
        if self.positional_encoding_input is not None:
            X = self.positional_encoding_input(X)
        X = self.dropout_input(X.reshape(N*L, -1)).reshape(N, L, -1)
        X = self.transformer_encoder(X, padding_mask)
        return X

    def decode(self, Y: torch.Tensor, encoded: torch.Tensor, encoded_padding_mask: Optional[torch.Tensor],
               history: Optional[Tuple[dict]]=None):
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
        if self.positional_encoding_output is not None:
            offset = history[0].get("query_offset", 0) if history is not None else 0
            Y = self.positional_encoding_output(Y, offset=offset)
        Y = self.dropout_output(Y.reshape(N*L, -1)).reshape(N, L, -1)
        Y = self.transformer_decoder(Y, encoded, encoded_padding_mask, history)
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
                             weights, class_weights, label_smoothing=self.label_smoothing)

    # def predict(self, sequences: List[str], max_tokens: Optional[int] = None,
    #             n_beams: int = 1) -> List[str]:
    #     """
    #     Predict a translation for the given sequences using beam search,
    #     outputing at most 'max_tokens' tokens.
    #     If 'n_beams' is 1, this is equivalent to predicting the single token
    #     with the highest likelyhood at each step.
    #     """
    #     if isinstance(sequences, str):
    #         sequences = [sequences]
    #     self.eval()
    #     with torch.no_grad():
    #         X = self._x_to_tensor(sequences, self.device, raise_on_longer_sequences=True)
    #         START = self.tokenizer_output.START
    #         END = self.tokenizer_output.END
    #         PAD = self.tokenizer_input.PAD
    #         n_classes = self.tokenizer_output.n_tokens
    #         encoded_padding_mask = (X == PAD) if self.mask_padding else None
    #         encoded = self(X, encoded_padding_mask)
    #         N, _, D = encoded.shape
    #         encoded_expanded = encoded.unsqueeze(1).repeat(1, n_beams, 1, 1).reshape(N*n_beams, -1, D)
    #         if self.mask_padding:
    #             encoded_padding_mask_expanded = encoded_padding_mask.unsqueeze(1).expand(-1, n_beams, -1)
    #         else:
    #             encoded_padding_mask_expanded = None
    #         predicted = torch.zeros((N, 1, 0), device=X.device, dtype=torch.long)  # index in vocabulary of predicted tokens (N, n_beams, L)
    #         log_likelyhood = torch.zeros((N, 1), device=X.device, dtype=torch.float)  # sum of negative log likelyhood of rpedicted tokens (N, n_beams)
    #         n_predicted_tokens = torch.zeros((N, 1), device=X.device, dtype=torch.long)  # number of predicted tokens before <END> (N, n_beams)
    #         intermediate = [torch.zeros((N, 0, D), device=X.device)
    #                         for _ in self.transformer_encoder.stages]  # list of intermediate representations (N, L, D)
    #         I = torch.full([N, 1], START,
    #                        dtype=torch.long, device=X.device)  # Index of previously predicted tokens in the vocabulary (N*n_beams, 1)
    #         counter = range(max_tokens) if max_tokens is not None else count(0)
    #         for i in counter:
    #             stop = (predicted == END).any(dim=-1)
    #             if stop.all():
    #                 break
    #             Q = self.embedding_output(I)
    #             if self.positional_encoding_output is not None:
    #                 Q = self.positional_encoding_output(Q, offset=i)
    #             intermediate, Q = self.transformer_decoder.predict(
    #                 intermediate, Q, encoded, encoded_padding_mask)
    #             # lookup the beam/token that lead to highest mean log likelyhood
    #             log_p = torch.log(torch.softmax(self.head(Q.reshape(N, -1, D)), dim=-1))
    #             all_log_likelyhoods = log_likelyhood.unsqueeze(-1) + torch.masked_fill(log_p, stop.unsqueeze(-1), 0.)
    #             n_predicted_tokens = n_predicted_tokens + (~stop)
    #             mean_log_likelyhoods = all_log_likelyhoods / n_predicted_tokens.unsqueeze(-1)
    #             _, indexes = mean_log_likelyhoods.reshape(N, -1).topk(k=n_beams, dim=-1)
    #             beam, token = torch.div(indexes, n_classes, rounding_mode="floor"), indexes % n_classes
    #             # create the property of the new beams
    #             I = token.reshape(N*n_beams, 1)
    #             intermediate = [torch.gather(inter.reshape(N, predicted.shape[1], -1, D),
    #                                          1,
    #                                          beam.reshape(N, n_beams, 1, 1).expand(-1, -1, inter.shape[1], D)
    #                                          ).reshape(N*n_beams, -1, D)
    #                             for inter in intermediate]
    #             predicted = torch.gather(predicted, 1, beam.unsqueeze(-1).expand(-1, -1, predicted.shape[-1]))
    #             predicted = torch.cat([predicted, token.unsqueeze(-1)], dim=-1)
    #             log_likelyhood = torch.gather(all_log_likelyhoods.reshape(N, -1), -1, indexes).reshape(N, n_beams)
    #             n_predicted_tokens = torch.gather(n_predicted_tokens, 1, beam)
    #             encoded = encoded_expanded
    #             encoded_padding_mask = encoded_padding_mask_expanded
    #         # get best final beam
    #         predicted = predicted[:, 0, :]
    #         translations = [self.tokenizer_output.decode(p.cpu().tolist()) for p in predicted]
    #         return translations

    def predict(self, string: str, max_tokens: Optional[int] = None, n_beams: int = 1) -> List[str]:
        """
        Predict a translation for the given sequence using beam search,
        outputing at most 'max_tokens' tokens.
        """
        self.eval()
        with torch.no_grad():
            # encode input
            X = self._x_to_tensor([string], self.device, raise_on_longer_sequences=True)
            START = self.tokenizer_output.START
            END = self.tokenizer_output.END
            PAD = self.tokenizer_input.PAD
            encoded_padding_mask = (X == PAD) if self.mask_padding else None
            encoded = self(X, encoded_padding_mask)
            # decode encoded input
            sequences = [[START]]
            histories = [tuple({} for _ in self.transformer_decoder.stages)]
            sum_likelyhoods = [0.]
            counter = range(max_tokens) if max_tokens is not None else count(0)
            for _ in counter:
                predicted_likelyhoods = [torch.log(torch.softmax(self.decode(torch.tensor(sequence[-1:], dtype=torch.long, device=self.device).unsqueeze(0),
                                                   encoded, encoded_padding_mask, history), dim=-1))
                                         for sequence, history in zip(sequences, histories)]
                beam_search(n_beams, sequences, histories, sum_likelyhoods, predicted_likelyhoods)
                if all(sequence[-1] == END for sequence in sequences):
                    break
            # get final beams
            return [self.tokenizer_output.decode(sequence[1:-1]) for sequence in sequences]


    def _predict_naive(self, sequences: List[str], max_tokens: Optional[int] = None) -> List[str]:
        """
        For comparison sake, this should output the same result as predict with n_beams=1
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        self.eval()
        with torch.no_grad():
            X = self._x_to_tensor(sequences, self.device, raise_on_longer_sequences=True)
            START = self.tokenizer_output.START
            END = self.tokenizer_output.END
            PAD = self.tokenizer_input.PAD
            encoded_padding_mask = (X == PAD) if self.mask_padding else None
            encoded = self(X, encoded_padding_mask)
            N, _, D = encoded.shape
            predicted = torch.full([N, 1], START, dtype=torch.long, device=X.device)
            counter = range(max_tokens) if max_tokens is not None else count(0)
            for _ in counter:
                stop = (predicted == END).any(dim=-1)
                if stop.all():
                    break
                Q = self.embedding_output(predicted)
                if self.positional_encoding_output is not None:
                    Q = self.positional_encoding_output(Q)
                Q = self.transformer_decoder(Q, encoded, encoded_padding_mask)
                p = torch.softmax(self.head(Q), dim=-1)
                token = p.max(dim=-1).indices[:, -1:]
                predicted = torch.cat([predicted, token], dim=-1)
            else:
                warn(f"Prediction stoped because {max_tokens} where generated")
            translations = [self.tokenizer_output.decode(p.cpu().tolist()) for p in predicted]
            return translations


    @property
    def device(self) -> torch.device:
        return self.head.weight.device
    
    def data_to_tensor(self, x: object, y: object,
                       weights: Optional[Sequence[float]] = None,
                       device: Optional[torch.device] = None,
                       max_input_sequence_length: Optional[int] = None,
                       max_output_sequence_length: Optional[int] = None,
                       **kwargs) -> tuple:
        X = self._x_to_tensor(x, device, max_input_sequence_length, **kwargs)
        Y = self._y_to_tensor(y, device, max_output_sequence_length, **kwargs)
        # skiping observations where input or target was too long
        mask = (X[:, 0] != self.tokenizer_input.PAD) & (Y[:, 0] != self.tokenizer_output.PAD)
        x, y = X[mask, ...], Y[mask, ...]
        if weights is not None:
            w = floats_to_tensor(weights, device)
            data = (x, y, w/w.mean())
        else:
            data = (x, y)
        return data

    def _x_to_tensor(self, x: List[str],
                     device: Optional[torch.device] = None,
                     max_input_sequence_length: Optional[int] = None,
                     raise_on_longer_sequences: bool = False,
                     progress_bar: bool = False):
        return strings_to_tensor(x, self.tokenizer_input, device,
                                 max_sequence_length=max_input_sequence_length,
                                 raise_on_longer_sequences=raise_on_longer_sequences,
                                 add_start_end_tokens=False,
                                 progress_bar=progress_bar)

    def _y_to_tensor(self, y: List[str],
                     device: Optional[torch.device] = None,
                     max_output_sequence_length: Optional[int] = None,
                     raise_on_longer_sequences: bool = False,
                     progress_bar: bool = False) -> torch.Tensor:
        return strings_to_tensor(y, self.tokenizer_output, device,
                                 max_sequence_length=max_output_sequence_length,
                                 raise_on_longer_sequences=raise_on_longer_sequences,
                                 add_start_end_tokens=True,
                                 progress_bar=progress_bar)

    def _tensor_to_y(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor_to_strings(tensor, self.tokenizer_output)
