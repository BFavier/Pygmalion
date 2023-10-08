import torch
from pygmalion.neural_networks.layers.transformers import TransformerDecoder, TransformerEncoder
from pygmalion.neural_networks.layers.transformers.multihead_attention import ScaledDotProductAttention, KernelizedAttention, FourrierKernelAttention

attention_types = [ScaledDotProductAttention, FourrierKernelAttention]  # , KernelizedAttention]


def test_encoder():
    N, Lq, D = 1, 2, 4*3
    for attention_type in attention_types:
        encoder = TransformerEncoder(n_stages=1, projection_dim=4, n_heads=3, mask_future=True, attention_type=attention_type)
        encoder.eval()
        Y = torch.rand(N, Lq, D)
        with torch.no_grad():
            R = encoder(Y)
            histories = tuple(dict() for _ in range(len(encoder.stages)))
            results = []
            for i in range(Lq):
                results.append(encoder(Y[:, i:i+1, :], histories=histories))
            Q = torch.cat(results, dim=1)
        assert torch.allclose(R, Q, atol=1.0E-6, rtol=1.0E-5)


def test_decoder():
    N, Lq, Lk, D = 10, 100, 110, 4*3
    for attention_type in attention_types:
        decoder = TransformerDecoder(n_stages=2, projection_dim=4, n_heads=3, attention_type=attention_type)
        decoder.eval()
        encoded = torch.rand((N, Lk, D))
        encoded_padding_mask = (torch.rand((N, Lk)) > 0.5)
        Y = torch.rand(N, Lq, D)
        with torch.no_grad():
            R = decoder(Y, encoded, encoded_padding_mask)
            histories = tuple(dict() for _ in range(len(decoder.stages)))
            results = []
            for i in range(Lq):
                results.append(decoder(Y[:, i:i+1, :], encoded, encoded_padding_mask, histories))
            Q = torch.cat(results, dim=1)
        assert torch.allclose(R, Q, atol=1.0E-6, rtol=1.0E-5)


if __name__ == "__main__":
    test_encoder()
    test_decoder()
