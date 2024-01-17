from typing import Tuple

import torch
from transformers import pipeline

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.objects import File
from pipeline.objects.graph import InputField, InputSchema

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
    audio_file = Variable(
        File,
        title = "Audio file"
        )

    kwargs = Variable(ModelKwargs)

    model = WhisperModel()

    model.load()

    full_text, timestamps = model.predict(audio_file, kwargs)

    builder.output(full_text, timestamps)

# Building the pipeline

my_pl = builder.get_pipeline()
