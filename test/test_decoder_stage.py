import torch
from pygmalion.neural_networks.layers.transformers._stages import TransformerDecoderStage


def test_equality():
    N, Lq, Lk, D = 10, 100, 110, 4*3
    decoder = TransformerDecoderStage(projection_dim=4, n_heads=3)
    decoder.eval()
    encoded = torch.rand((N, Lk, D))
    encoded_padding_mask = (torch.rand((N, Lk)) > 0.5)
    Y = torch.rand(N, Lq, D)
    R = decoder.forward(Y, encoded, encoded_padding_mask)
    Q = decoder.predict(Y, encoded, encoded_padding_mask)
    assert torch.allclose(R[:, -1:, :], Q)


if __name__ == "__main__":
    test_equality()
