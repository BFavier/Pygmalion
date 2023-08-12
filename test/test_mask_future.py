import torch
from pygmalion.neural_networks.layers.transformers import TransformerDecoder, TransformerEncoder
from pygmalion.neural_networks.layers.transformers.multihead_attention import KernelizedAttention, ScaledDotProductAttention

N, L, n, h, d = 1, 10, 1, 4, 16
ATTENTIONS = [KernelizedAttention, ScaledDotProductAttention]


def test_encoder_mask_future():
    for attention_type in ATTENTIONS:
        encoder = TransformerEncoder(n, d, h, mask_future=True, gradient_checkpointing=False, attention_type=attention_type)
        encoder.eval()
        X = torch.rand(N, L, h*d)
        y0 = encoder(X)[:, :L//2, :]
        y1 = encoder(X[:, :L//2, :])
        assert torch.allclose(y0, y1, atol=1.0E-6, rtol=0.)


def test_decoder_mask_future():
    for attention_type in ATTENTIONS:
        decoder = TransformerDecoder(n, d, h, gradient_checkpointing=False, attention_type=attention_type)
        decoder.eval()
        encoded = torch.rand(N, L+10, h*d)
        Y = torch.rand(N, L, h*d)
        y0 = decoder(Y, encoded)[:, :L//2, :]
        y1 = decoder(Y[:, :L//2, :], encoded)
        assert torch.allclose(y0, y1, atol=1.0E-6, rtol=0.)


if __name__ == "__main__":
    test_encoder_mask_future()
    test_decoder_mask_future()
    import IPython
    IPython.embed()
