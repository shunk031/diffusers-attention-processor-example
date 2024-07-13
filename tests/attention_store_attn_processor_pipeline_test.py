import logging

import pytest
import torch
from PIL.Image import Image as PilImage

from my_pipelines import AttentionStoreAttnProcessorPipeline

logger = logging.getLogger(__name__)


@pytest.fixture
def model_id() -> str:
    return "runwayml/stable-diffusion-v1-5"


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def prompt() -> str:
    return "A photo of an astronaut riding on a horse."


@pytest.fixture
def resolution() -> int:
    return 16


def test_shape_store_attn_processor_pipeline(
    model_id: str, device: torch.device, prompt: str, resolution: int
):
    pipeline = AttentionStoreAttnProcessorPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    output = pipeline(prompt=prompt, attn_res=(resolution, resolution))
    (generated_image,) = output.images

    visualizations = pipeline.show_cross_attention(
        generated_image=generated_image, prompt=prompt, res=resolution
    )
    for i, vis in enumerate(visualizations):
        filepath = f"{i}_{vis['text']}.png"
        logger.info(f"Saving visualization to {filepath}")

        vis["image"].save(filepath)
