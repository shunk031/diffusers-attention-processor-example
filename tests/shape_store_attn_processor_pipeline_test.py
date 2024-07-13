import pytest
import torch
from PIL.Image import Image as PilImage

from my_pipelines import ShapeStoreAttnProcessorPipeline


@pytest.fixture
def model_id() -> str:
    return "runwayml/stable-diffusion-v1-5"


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def prompt() -> str:
    return "A photo of an astronaut riding on a horse."


def test_shape_store_attn_processor_pipeline(
    model_id: str, device: torch.device, prompt: str
):
    pipeline = ShapeStoreAttnProcessorPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    output = pipeline(prompt=prompt)
    assert isinstance(output.images[0], PilImage)
