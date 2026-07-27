import unittest

import numpy as np
import torch

from analysis._shared import cascade_pre_post


class AddBlock(torch.nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.register_buffer("delta", torch.tensor(delta, dtype=torch.float32))

    def forward(self, hidden):
        return hidden + self.delta


class DummyBackbone(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, input_ids, use_cache=False):
        hidden = torch.stack(
            (input_ids.float(), input_ids.float() * 2),
            dim=-1,
        )
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class DummyModel:
    device = torch.device("cpu")

    def __init__(self, layers):
        self.model = DummyBackbone(layers)


class CascadePrePostTest(unittest.TestCase):
    def test_pre_is_target_output_before_projection_not_target_input(self):
        layers = [
            AddBlock([1.0, 2.0]),
            AddBlock([10.0, 20.0]),
        ]
        model = DummyModel(layers)
        probes = {
            1: {
                "w": torch.tensor([1.0, 0.0]),
                "b": torch.tensor(0.0),
                "margin": 0.0,
            }
        }

        pre, post = cascade_pre_post(
            model=model,
            layer_modules=layers,
            lower_layers=[],
            target_layer=1,
            probes=probes,
            beta=1.0,
            target_margin=0.0,
            ids_list=[torch.tensor([[3]])],
        )

        # Target-layer input is [4, 8], while its unprojected output is [14, 28].
        np.testing.assert_allclose(pre, [[14.0, 28.0]])
        np.testing.assert_allclose(post, [[0.0, 28.0]])
        self.assertEqual(len(layers[1]._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
