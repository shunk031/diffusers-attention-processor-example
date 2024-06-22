from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention, AttnProcessor


@dataclass
class ShapeStore:
    """shapeを保存しておく用のクラス"""

    q: torch.Size  # query
    k: torch.Size  # key
    v: torch.Size  # value
    attn: torch.Size  # attention score/probs


class NewAttnProcessor(AttnProcessor):
    def __init__(self):
        super().__init__()
        # Self/Cross Attention の保存先を追加
        self.self_attentions = []
        self.cross_attentions = []

    def get_attention_scores(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        オレオレ `get_attention_scores`
        """
        dtype = query.dtype
        if attn.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=attn.scale,
        )
        del baddbmm_input

        if attn.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = self.get_attention_scores(attn, query, key, attention_mask)

        is_cross_attn = encoder_hidden_states is not None
        if is_cross_attn:
            self.cross_attentions.append(
                ShapeStore(
                    q=query.shape,
                    k=key.shape,
                    v=value.shape,
                    attn=attention_probs.shape,
                )
            )
        else:
            self.self_attentions.append(
                ShapeStore(
                    q=query.shape,
                    k=key.shape,
                    v=value.shape,
                    attn=attention_probs.shape,
                )
            )

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class MyPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        self.unet.set_attn_processor(NewAttnProcessor())
        return super().__call__(*args, **kwargs)


def main():
    pipeline = MyPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline = pipeline.to("cuda")

    output = pipeline(prompt="A photo of an astronaut riding on a horse.")
    (image,) = output.images

    image.save("generated-image.png")


if __name__ == "__main__":
    main()
