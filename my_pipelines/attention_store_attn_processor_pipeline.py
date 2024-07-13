import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, TypedDict, Union

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from PIL import Image
from PIL.Image import Image as PilImage

logger = logging.getLogger(__name__)

BlockName = Literal["down", "mid", "up"]


class AttentionStoreDict(TypedDict):
    down: List[torch.Tensor]
    mid: List[torch.Tensor]
    up: List[torch.Tensor]


@dataclass
class AttentionStore(object):
    attn_res: Tuple[int, int]

    num_att_layers: int = -1

    cur_att_layer: int = 0
    curr_step_index: int = 0

    _attn_store: Optional[AttentionStoreDict] = None
    _step_store: Optional[AttentionStoreDict] = None

    def __post_init__(self):
        self._step_store = self.get_empty_store()

    @property
    def attn_store(self) -> AttentionStoreDict:
        assert self._attn_store is not None
        return self._attn_store

    @property
    def step_store(self) -> AttentionStoreDict:
        assert self._step_store is not None
        return self._step_store

    def get_empty_store(self) -> AttentionStoreDict:
        return {"down": [], "mid": [], "up": []}

    def reset(self) -> None:
        self.cur_att_layer = 0
        self._step_store = self.get_empty_store()
        self._attn_store = None

    def between_steps(self) -> None:
        self._attn_store = self.step_store
        self._step_store = self.get_empty_store()

    def get_average_attention(self) -> AttentionStoreDict:
        average_attention = self.attn_store
        return average_attention

    def aggregate_attention(self, from_where: List[BlockName]) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps[location]:
                cross_maps = item.reshape(
                    -1, self.attn_res[0], self.attn_res[1], item.shape[-1]
                )
                out.append(cross_maps)
        out_th = torch.cat(out, dim=0)
        out_th = out_th.sum(0) / out_th.shape[0]
        return out_th

    def __call__(self, attn, is_cross: bool, place_in_unet: BlockName) -> None:
        if self.cur_att_layer >= 0 and is_cross:
            if attn.shape[1] == np.prod(self.attn_res):
                self.step_store[place_in_unet].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()


class AttentionStoreProcessor(AttnProcessor):
    def __init__(self, attn_store: AttentionStore, place_in_unet: BlockName) -> None:
        super().__init__()
        self.attn_store = attn_store
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask,  # type: ignore
            sequence_length,
            batch_size,
        )

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        self.attn_store(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class AttentionStoreAttnProcessorPipeline(StableDiffusionPipeline):
    unet: UNet2DConditionModel
    vae_scale_factor: int

    _attention_store: Optional[AttentionStore]

    @property
    def attention_store(self) -> AttentionStore:
        assert self._attention_store is not None
        return self._attention_store

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttentionStoreProcessor(
                attn_store=self.attention_store, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    def show_image_relevance(
        self, image_relevance: torch.Tensor, image: PilImage, relevnace_res: int = 16
    ) -> PilImage:
        # create heatmap from mask on image
        def show_cam_on_image(img, mask):
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img)
            cam = cam / np.max(cam)
            return cam

        image = image.resize((relevnace_res**2, relevnace_res**2))
        image_np = np.array(image)

        image_relevance = image_relevance.reshape(
            1, 1, image_relevance.shape[-1], image_relevance.shape[-1]
        )
        image_relevance = (
            image_relevance.cuda()
        )  # because float16 precision interpolation is not supported on cpu
        image_relevance = torch.nn.functional.interpolate(
            image_relevance, size=relevnace_res**2, mode="bilinear"
        )
        image_relevance = image_relevance.cpu()  # send it back to cpu
        image_relevance = (image_relevance - image_relevance.min()) / (
            image_relevance.max() - image_relevance.min()
        )
        image_relevance = image_relevance.reshape(relevnace_res**2, relevnace_res**2)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        vis = show_cam_on_image(image_np, image_relevance)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

        vis = vis.astype(np.uint8)
        return Image.fromarray(vis)

    def show_cross_attention(
        self, generated_image: PilImage, prompt: str, res: int
    ) -> List[Dict[str, Union[str, PilImage]]]:
        attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid")
        )

        tokens = self.tokenizer.encode(prompt)
        logger.info([self.tokenizer.decode(t) for t in tokens])

        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = self.show_image_relevance(
                image_relevance=image, image=generated_image, relevnace_res=res
            )
            image = image.resize((res**2, res**2))
            images.append({"text": self.tokenizer.decode(tokens[i]), "image": image})
        return images

    @torch.no_grad()
    def __call__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        attn_res: Optional[Tuple[int, int]] = (16, 16),
        *args,
        **kwargs,
    ) -> StableDiffusionPipelineOutput:
        height = height or self.unet.config.sample_size * self.vae_scale_factor  # type: ignore
        width = width or self.unet.config.sample_size * self.vae_scale_factor  # type: ignore

        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))

        self._attention_store = AttentionStore(attn_res)
        self.register_attention_control()

        return super().__call__(width=width, height=height, *args, **kwargs)  # type: ignore
