from my_pipelines import ShapeStoreAttnProcessorPipeline


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


def main():
    run_shape_store_attn_processor_pipeline()


if __name__ == "__main__":
    main()
