import logging
from copy import deepcopy

import mlflow

from PIL import Image
from mlflow.models import infer_signature
from model_helpers.data_loader.azure.azure_image_loader import ImageAzureLoader
from model_helpers.data_loader.image.data_frame import DataFrameLoader
from model_helpers.post_process.common import BasePostProcessor
from model_helpers.utils.codecs import pil_to_dataframe

from {{cookiecutter.project_slug}}.model_builder import get_requirements
from {{cookiecutter.project_slug}}.model_wrapper import ModelWrapper

logger = logging.getLogger(__name__)


def upload_artifact(artifact_path, model_class: str, model_freeze: bool):

    for loader_class, artifact_name in [(DataFrameLoader, "wrapped_model"), (ImageAzureLoader, "wrapped_model_azure")]:

        logger.info(f"Loading model with loader class: {loader_class}")

        wrapper = ModelWrapper(
            model_class=model_class,
            loader_class=loader_class,
            post_processors_class=BasePostProcessor,
            freeze=model_freeze
        )

        model = deepcopy(wrapper)
        model.load_model(artifact_path)

        model_input_signature = loader_class.get_data_example()

        image_path = './img_1.png'
        image = Image.open(image_path).convert("RGB")
        model_input = pil_to_dataframe(image)

        prediction = model.run_prediction(model_input)
        model_output_df = model.post_process_elem(
            model_input,
            prediction
        )

        if model_input_signature is None:
            model_input_signature = model_input

        model_input = model_input.head()
        signature = infer_signature(model_input_signature, model_output_df)  # model input to be modified

        mlflow.pyfunc.log_model(
            python_model=wrapper,
            input_example=model_input,
            signature=signature,
            artifact_path=artifact_name,
            artifacts={
                "checkpoint": artifact_path
            },
            code_path=["{{cookiecutter.project_slug}}/"],
            pip_requirements=get_requirements()
        )