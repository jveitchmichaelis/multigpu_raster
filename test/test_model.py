import torch

from multi_gpu_raster.model import ResNet50


def test_model_inference():
    model = ResNet50()
    x = torch.randn((1, 3, 224, 224))
    output = model(x)
    assert output.shape[1] == 1000, "ResNet50 output dimension mismatch"
