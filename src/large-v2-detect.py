from typing import Tuple

import torch
from transformers import pipeline
from optimum.bettertransformer import BetterTransformer

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.cloud import compute_requirements, environments, pipelines
from pipeline.objects import File
from pipeline.objects.graph import InputField, InputSchema

# Getting environment variables
import os

# Initializing environment variables
login   = os.environ["USERNAME"]
pl      = os.environ["PIPELINE"]
env     = os.environ["ENVIRONMENT"]

# Parameters
class ModelKwargs(InputSchema):
    batch_size: int | None = InputField(
        default=24,
        ge=1,
        le=64,
        title="Batch Size",
    )

    return_timestamps: bool | None = InputField(
        default=False,
        title="Return Timestamps",
    )

@entity
class WhisperModel:
    def __init__(self):
        ...

    @pipe(on_startup=True, run_once=True)
    def load(self):
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.device = torch.device(0 if torch.cuda.is_available() else "cpu")
        self.pipe = pipeline(
            generate_kwargs={
                "task": "transcribe",
            },
            model="openai/whisper-large-v2",
            torch_dtype=self.torch_dtype,
            chunk_length_s=30,
            stride_length_s=5,
            device=self.device,
        )
                
    @pipe
    def predict(self, audio_file: File, kwargs: ModelKwargs) -> Tuple[str, list | None]:
        prediction = self.pipe(
            str(audio_file.path),
            batch_size=kwargs.batch_size,
            return_timestamps=kwargs.return_timestamps,
        )

        full_text: str = prediction["text"]
        timestamps: list = prediction["chunks"] if kwargs.return_timestamps else None

        return (full_text, timestamps)

# Creating the pipeline

with Pipeline() as builder:
    audio_file = Variable(File)
    kwargs = Variable(ModelKwargs)

    model = WhisperModel()

    model.load()

    full_text, timestamps = model.predict(audio_file, kwargs)

    builder.output(full_text, timestamps)

# Building the pipeline

my_pl = builder.get_pipeline()

# To run locally:

# output = my_pl.run(
#     File(path="output.mp3", allow_out_of_context_creation=True), ModelKwargs()
# )
# print(output)

my_pl_name = f"{login}/{pl}"
my_env_name = f"{login}/{env}"
# print(my_pl_name, my_env_name)

# Creating the environment

environments.create_environment(
    my_env_name,
    python_requirements=[
        "torch>=2.1.1",
        "transformers>=4.36.1",
        "optimum>=1.16.0",
        "accelerate>=0.25.0",
    ],
)
 
# environments

# # Uploading the pipeline


pipelines.upload_pipeline(
    my_pl,
    my_pl_name,
    environment_id_or_name=my_env_name,
    required_gpu_vram_mb=10_000,
    accelerators=[
        compute_requirements.Accelerator.nvidia_l4,
    ],
)