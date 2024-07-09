from my_pipelines import (
    AttentionStoreAttnProcessorPipeline,
    ShapeStoreAttnProcessorPipeline,
)


def run_shape_store_attn_processor_pipeline():
    pipeline = ShapeStoreAttnProcessorPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    )
    pipeline = pipeline.to("cuda")

    output = pipeline(prompt="A photo of an astronaut riding on a horse.")
    (image,) = output.images

    image.save("generated-image.png")

    for name, attn_processor in pipeline.unet.attn_processors.items():
        print(name, attn_processor)


def run_attention_store_attn_processor_pipeline():
    pipeline = AttentionStoreAttnProcessorPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    )
    pipeline = pipeline.to("cuda")

    output = pipeline(prompt="A photo of an astronaut riding on a horse.")
    (image,) = output.images

    image.save("generated-image.png")

    for name, attn_processor in pipeline.unet.attn_processors.items():
        print(name, attn_processor)

    attention_maps = pipeline.attention_store.aggregate_attention(
        from_where=("up", "down", "mid")
    )


def main():
    # run_shape_store_attn_processor_pipeline()
    run_attention_store_attn_processor_pipeline()


if __name__ == "__main__":
    main()
