import torch
import math
from ._weighting import Linear
from ._functional import mask_chronological
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 linear_complexity: bool = True):
        """
        Parameters
        ----------
        projection_dim : int
            the dimension of the projection space for the feature vectors
        n_heads : int
            the number of different projection at each stage of the transformer
        linear_complexity : bool
            if True, a variant of the attention mechanism is used that has
            linear coplexity with the sequence length of tokens
        """
        super().__init__()
        self.n_heads = n_heads
        self.projection_dim = projection_dim
        self.linear_complexity = linear_complexity
        dim = projection_dim*n_heads
        self.query = Linear(dim, dim, bias=False)
        self.key = Linear(dim, dim, bias=False)
        self.value = Linear(dim, dim, bias=False)

    @classmethod
    def from_dump(cls, dump: dict) -> 'MultiHeadAttention':
        assert dump["type"] == cls.__name__
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.n_heads = dump["n heads"]
        obj.projection_dim = dump["projection dim"]
        obj.linear_complexity = dump["linear complexity"]
        obj.query = Linear.from_dump(dump["query"])
        obj.key = Linear.from_dump(dump["key"])
        obj.value = Linear.from_dump(dump["value"])
        return obj

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        Forward pass of the multihead attention module.
        Apply masked attention, followed by dropout, and batch normalization

        Parameters
        ----------
        query : torch.Tensor
            tensor of shape (N, Lq, D) with
            * N the number of sentences to treat
            * Lq the sequence length of the query
            * D the embedding dimension
        key : torch.Tensor
            tensor of shape (N, Lk, D) with
            * N the number of sentences to treat
            * Lk the sequence length of the key
            * D the embedding dimension
        mask : torch.Tensor or None
            the mask, tensor of booleans of shape (Lq, Lk), where attention
            is set to -infinity

        Returns
        -------
        torch.Tensor :
            tensor of shape (N, Lq, D)
        """
        if self.linear_complexity:
            return self._multihead_attention(query, key, (mask is not None))
        else:
            # pytorch's stock implementation is more memory efficient
            return self._multihead_attention_stock(query, key, mask)

    def _multihead_attention(self, query: torch.Tensor, key: torch.Tensor,
                             mask: Union[Optional[torch.Tensor], bool]
                             ) -> torch.Tensor:
        """
        Apply multihead attention.
        Same inputs/outputs types/shapes as the forward pass
        """
        N, Lq, _ = query.shape
        N, Lk, _ = key.shape
        # project into 'n_heads' different subspaces
        q = self.query(query).reshape(N, Lq, self.n_heads, self.projection_dim)
        k = self.key(key).reshape(N, Lk, self.n_heads, self.projection_dim)
        v = self.value(key).reshape(N, Lk, self.n_heads, self.projection_dim)
        # compute attention dot product
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.linear_complexity:
            attention, _ = self._linear_complexity_attention(q, k, v, mask)
        else:
            attention, _ = self._scaled_dot_product_attention(q, k, v, mask)
        attention = attention.transpose(2, 1).reshape(N, Lq, -1)
        return attention

    def _linear_complexity_attention(self, q: torch.Tensor, k: torch.Tensor,
                                     v: torch.Tensor, mask: bool = False,
                                     RPE: Optional[torch.Tensor] = None
                                     ) -> Tuple[torch.Tensor, None]:
        """
        This is an alternative attention function.
        'softmax(Q*Kt)*V' becomes 'softmax(Q)*softmax(K)*V'
        This allow to chose operation order for the 3 matrices multiplications
        and have linear (instead of squared) complexity with sequence length or
        projection dim depending on the situation.


        This implementation reproduces this paper :
            'Efficient Attention: Attention with Linear Complexities'
            https://arxiv.org/pdf/1812.01243.pdf
        The masking for causal-attention is inspired from the paper :
            'Rethinking Attention with Performers', Annexe B.1
            https://arxiv.org/pdf/2009.14794.pdf

        Parameters
        ----------
        q : torch.Tensor
            query tensor of shape (N, H, Lq, D)
        k : torch.Tensor
            key tensor of shape (N, H, Lk, D)
        v : torch.Tensor
            value tensor of shape (N, H, Lk, D)
        mask : bool
            If True, returns unidirectional attention

        Returns
        -------
        tuple of (torch.tensor, None) :
            the attention score cannot always be calculated,
            so None is returned instead
        """
        if not isinstance(mask, bool):
            raise ValueError("Custom masking is not supported by linear "
                             "complexity attention. 'mask' must be True for "
                             "unidirectional, and False for bidirectional "
                             "attention.")
        _, _, Lq, D = q.shape
        _, _, Lk, _ = k.shape
        q = torch.softmax(q, dim=-1)
        k = torch.softmax(k, dim=-1)
        if mask:
            left_cost = Lq * Lk
            right_cost = D**2 * Lk
            if left_cost <= right_cost:
                left = torch.matmul(q, k.transpose(-2, -1))
                mask = mask_chronological(Lq, Lk, left.device)
                left = left.masked_fill(mask, 0.)
                result = torch.matmul(left, v)
            else:
                right = torch.cumsum(
                    torch.einsum("...ki, ...kj -> ...kij", k, v), dim=-3)
                result = torch.matmul(q.unsqueeze(-2), right).squeeze(-2)
                return result, None
        else:
            # slower than 2 matrix products, but less intermediate memory used
            result = torch.einsum("...aq, ...cq, ...cb -> ...ab", q, k, v)
            # # check the most costly operation order between
            # # (q*kt)*v and q*(kt*v)
            # kt = k.transpose(-1, -2)
            # left_cost = Lq*Lk
            # right_cost = D**2
            # if left_cost <= right_cost:
            #     left = torch.matmul(q, kt)
            #     return torch.matmul(left, v), None
            # else:
            #     right = torch.matmul(kt, v)
            #     return torch.matmul(q, right), None
        return result, None

    def _linear_RPE(self, q: torch.Tensor, RPE_matrix: torch.Tensor,
                    v: torch.Tensor, masked: bool = False) -> torch.Tensor:
        """
        Performs the 'Relative Positional Encoding' part of the attention
        as described in:

        'Self-Attention with Relative Position Representations'
        https://arxiv.org/pdf/1803.02155.pdf


        implementation is adapted from the second strategy described in:
        'Translational Equivariance in Kernelizable Attention'
        https://arxiv.org/pdf/2102.07680.pdf

        Parameters
        ----------
        q : torch.Tensor
            tensor of shape (N, H, Lq, D)
        RPE_matrix : torch.Tensor
            tensor of shape (2*R+1, D)
        v : torch.Tensor
            tensor of shape (N, H, Lk, D)

        Returns
        -------
        torch.Tensor :
            tensor of shape (N, H, Lq, 2*R+1)
        """
        N, H, Lq, D = q.shape
        E, D = RPE_matrix.shape
        N, H, Lk, D = v.shape
        R = (E-1)//2
        assert E % 2 == 1 and R > 0
        # compute the dot product between each query and each RPE embedding
        # 'scores' is of shape (N, H, Lq, 2*R+1)
        scores = torch.matmul(q.reshape(N, H, Lq, 1, 1, D),
                              RPE_matrix.reshape(1, 1, 1, E, D, 1)
                              ).squeeze(-2).squeeze(-1)
        # split the scores
        before = scores[..., 0:1]  # the score of distance -R and lower
        after = scores[..., -1:]  # the score of the distance R and upper
        if masked:
            # The score of the distances -(R-1) to 0
            window = scores[..., 1:R+2].unsqueeze(-2)
        else:
            # The scores of the distances -(R-1) to (R-1)
            window = scores[..., 1:-1].unsqueeze(-2)
        # sum of the value vectors on the left of the RPE horizon
        head = torch.zeros((N, H, R, D), device=q.device)
        core = v[..., :min(Lq-R, Lk), :].cumsum(dim=-2)
        content = [head, core]
        if Lq-Lk-R > 0:
            bottom = core[..., -1:, :].repeat(1, 1, max(0, Lq-Lk-R), 1)
            content.append(bottom)
        left = torch.cat(content, dim=-2)
        if masked:
            # weighted sum of the values in the first half of the window
            slider = torch.cat([torch.zeros((N, H, max(0, R-1), D),
                                            device=q.device),
                                v, torch.zeros((N, H, max(0, Lq-Lk), D),
                                               device=q.device)],
                               dim=-2)
            L = slider.shape[-2]
            slider = slider.as_strided((N, H, Lq, R+1, D),
                                       (H*L*D, L*D, D, D, 1))
            center = torch.matmul(window, slider).squeeze(-2)
            # returns the weighted sum of value vectors
            return before*left + center
        else:
            # sum of the value vectors on the right of the RPE horizon
            stop = min(Lq+R, Lk)
            core = (v[..., R-1:stop, :].sum(dim=-2).unsqueeze(-2)
                    - v[..., R-1:stop-1, :].cumsum(dim=-2))
            bottom = torch.zeros((N, H, max(0, Lq-(Lk-R)), D), device=q.device)
            right = torch.cat([core, bottom], dim=-2)
            # weighted sum of the values in the horizon
            slider = torch.cat([torch.zeros((N, H, max(0, R-1), D),
                                            device=q.device),
                                v, torch.zeros((N, H, max(0, Lq-(Lk-R)), D),
                                               device=q.device)],
                               dim=-2)
            L = slider.shape[-2]
            slider = slider.as_strided((N, H, Lq, E-2, D),
                                       (H*L*D, L*D, D, D, 1))
            center = torch.matmul(window, slider).squeeze(-2)
            # returns the weighted sum of value vectors
            return before*left + center + after*right

    def _naive_RPE(self, q: torch.Tensor, RPE_matrix: torch.Tensor,
                   v: torch.Tensor, masked: bool = False) -> torch.Tensor:
        """
        naive implementation of _linear_RPE
        for testing only
        """
        N, H, Lq, D = q.shape
        E, D = RPE_matrix.shape
        N, H, Lk, D = v.shape
        R = (E-1)//2
        assert R >= 0
        # compute the dot product between each query and each RPE embedding
        # 'scores' is of shape (N, H, Lq, 2*R+1)
        scores = torch.matmul(q.reshape(N, H, Lq, 1, 1, D),
                              RPE_matrix.reshape(1, 1, 1, E, D, 1)
                              ).squeeze(-2).squeeze(-1)
        # expands score matrix of shape (N, H, Lq, Lk)
        indexes = torch.tensor([[R-i+j for j in range(Lk)] for i in range(Lq)],
                               dtype=torch.long, device=q.device).clip(0, 2*R)
        indexes = indexes.reshape(1, 1, Lq, Lk).expand(N, H, Lq, Lk)
        scores = torch.gather(scores, -1, indexes)
        # masks the top right corner of the matrix if needed
        if masked:
            mask = mask_chronological(Lq, Lk, q.device)
            scores = scores.masked_fill(mask, 0.)
        # returns the weighted sum of value vectors
        return torch.matmul(scores, v)

    def _scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor,
                                      v: torch.Tensor,
                                      mask: Optional[torch.Tensor]
                                      ) -> Tuple[torch.Tensor]:
        """
        Apply scaled dot product attention to a batch of 'N' sentences pairs,
        with 'H' the number of heads, and 'D' the projection dimension.
        The query is a sequence of length 'Lq', and the key is
        a sequence of length 'Lk'.

        This is the original attention mechanism described in the 2017 paper:
            'Attention is all you need'
            https://arxiv.org/pdf/1706.03762.pdf

        Parameters
        ----------
        q : torch.Tensor
            query tensor of shape (N, H, Lq, D)
        k : torch.Tensor
            key tensor of shape (N, H, Lk, D)
        v : torch.Tensor
            value tensor of shape (N, H, Lk, D)
        mask : torch.Tensor or None
            tensor of booleans of shape (Lq, Lk)

        Returns
        -------
        tuple of torch.Tensors:
            a tuple of (attention, score)
        """
        Lk = k.shape[-2]
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Lk)
        if mask is not None:
            score = score.masked_fill(mask, -1.0E10)
        score = torch.softmax(score, dim=-1)
        attention = torch.matmul(score, v)
        return attention, score

    def _multihead_attention_stock(self, query: torch.Tensor,
                                   key: torch.Tensor,
                                   mask: Optional[torch.Tensor]):
        """
        Apply multihead attention.
        Same inputs/outputs types/shapes as the forward pass
        """
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = key
        embed_dim = self.projection_dim*self.n_heads
        in_proj_weight = None
        in_proj_bias = None
        out_proj_weight = torch.eye(embed_dim, dtype=torch.float,
                                    device=query.device)
        out_proj_bias = torch.zeros(embed_dim, dtype=torch.float,
                                    device=query.device)
        dropout = 0.
        return F.multi_head_attention_forward(
                query, key, value, embed_dim, self.n_heads,
                in_proj_weight, in_proj_bias,
                None, None, False,
                dropout, out_proj_weight, out_proj_bias,
                training=self.training,
                key_padding_mask=None, need_weights=False,
                attn_mask=mask, use_separate_proj_weight=True,
                q_proj_weight=self.query.weight,
                k_proj_weight=self.key.weight,
                v_proj_weight=self.value.weight)[0].transpose(1, 0)

    @property
    def in_features(self):
        return self.query.in_features

    @property
    def out_features(self):
        return self.value.out_features

    @property
    def dump(self) -> dict:
        return {"type": type(self).__name__,
                "n heads": self.n_heads,
                "projection dim": self.projection_dim,
                "linear complexity": self.linear_complexity,
                "query": self.query.dump,
                "key": self.key.dump,
                "value": self.value.dump}
