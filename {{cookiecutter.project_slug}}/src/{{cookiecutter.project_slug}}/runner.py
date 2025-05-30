import os
import time
from copy import deepcopy
import logging

import click
import mlflow
import pandas as pd
from PIL import Image
from PIL import ImageFile
from mlflow.models import infer_signature
from model_helpers.data_loader.image.data_frame import DataFrameLoader
from model_helpers.post_process.common import BasePostProcessor
from model_helpers.utils.codecs import pil_to_dataframe
from types import SimpleNamespace

from {{cookiecutter.project_slug}}.model_builder import get_requirements
from {{cookiecutter.project_slug}}.model_wrapper import ModelWrapper
from {{cookiecutter.project_slug}}.utils.uploader import upload_artifact

ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
import numpy as np

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@click.command()
@click.option("--number_of_epoch", help="Number of training epochs", default=10)
@click.option("--device", default='cuda:0', help="Device to use for training")
@click.option("--dataset_path", default=default_dataset_path, help="Path to the dataset directory")
@click.option("--split_path", default=default_split_path, help="Path to split configuration")
@click.option("--run_name", default='{{cookiecutter.project_slug}}', help="Name for the current run")
@click.option("--only_list", is_flag=True, help="Only list tasks without executing")
@click.option("--clean_run", default=True, help="Start fresh without loading previous weights")
@click.option("--save-weights", default=True, help="Save model weights during training")
@click.option("--save-scores", default=True, help="Save evaluation scores during training")
@click.option("--phase", multiple=True, default=['test'], help="Phases to execute (train/test)")
@click.option("--resize_prob", type=float, default=0.2, help="Probability for random resize crop")
@click.option("--resize_size", type=int, default=512, help="Output size for random resize crop")
@click.option("--resize_scale", nargs=2, type=float, default=[0.2, 1.0], help="Scale range for random resize")
@click.option("--resize_ratio", nargs=2, type=float, default=[0.75, 1.3333333333333333], help="Aspect ratio range for resize")
@click.option("--jpeg_prob", type=float, default=0.2, help="Probability for JPEG compression")
@click.option("--jpeg_qual", nargs=2, type=int, default=[30, 100], help="JPEG quality range")
@click.option("--blur_prob", type=float, default=0.2, help="Probability for Gaussian blur")
@click.option("--blur_sigma", nargs=2, type=float, default=[1e-6, 3.0], help="Sigma range for Gaussian blur")
@click.option("--patch_size", type=int, default=96, help="Crop size after augmentation")
@click.option("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
@click.option("--learning_dropoff", type=int, default=3, help="Epochs before learning rate reduction")
@click.option("--dropout", type=float, default=0.0, help="Dropout probability")
@click.option("--model_flag", default='nodown', help="Model architecture flag")
@click.option("--model_freeze", type=bool, default=False, help="Freeze base model weights")
@click.option("--features", type=bool, default=False, help="Extract features before linear layer")
@click.option("--batch_size", type=int, default=16, help="Training batch size")
@click.option("--min_vram", type=int, default=16000, help="Minimum VRAM requirement in MB")
@click.option("--dry_run", type=bool, default=False, help="Simulate training without actual execution")
def runner(
    number_of_epoch,
    device,
    dataset_path,
    split_path,
    run_name,
    only_list,
    dry_run,
    clean_run,
    save_weights,
    save_scores,
    phase,
    resize_prob,
    resize_size,
    resize_scale,
    resize_ratio,
    jpeg_prob,
    jpeg_qual,
    blur_prob,
    blur_sigma,
    patch_size,
    learning_rate,
    learning_dropoff,
    dropout,
    model_flag,
    model_freeze,
    features,
    batch_size,
    min_vram,
):
    print("Current working directory:", os.getcwd())

    model = MyModel()

    # inti the mlflow run for storing metrics and artifacts
    with mlflow.start_run(run_name=run_name) as run: 

        model.train()

        for epoch in range(number_of_epoch):
            with tqdm(train_loader, unit='batch', mininterval=0.5) as tepoch:
                
                
                tepoch.set_description(f'Epoch {epoch}', refresh=False)

                model.fit(...)

                os.makedirs(os.path.join("./tmp_dir/", 'checkpoints'), exist_ok=True)
                artifact_path = os.path.join("./tmp_dir/", 'checkpoints', f'{epoch}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                }, artifact_path)

                # saving checkpoint on tracking server
                mlflow.log_artifact(artifact_path)

        # Save the best model

        best_check_point_path = os.path.join("./tmp_dir/", 'checkpoints', 'best.pt')

        torch.save({
            'model_state_dict': model.state_dict(),
        }, best_check_point_path)


        wrapper = ModelWrapper(
            loader_class=DataFrameLoader,
            post_processors_class=BasePostProcessor   
        )

        # TODO load sample file
        # image_path = './img_1.png'
        # image = Image.open(image_path).convert("RGB")
        # model_input = pil_to_dataframe(image)

        # TODO run predction
        # prediction = model.run_prediction(model_input)
        # model_output_df = model.post_process_elem(
        #     model_input,
        #     prediction
        # )

        # infer signature

        upload_artifact(
            artifact_path=best_model_path,
            model_class=model_flag,
            model_freeze=model_freeze
        )
        # Logging model metrics on traking server
        mlflow.log_metric("accuracy_tot", output_values[0])
        mlflow.log_metric("accuracy_fake", output_values[1])
        mlflow.log_metric("accuracy_real", output_values[2])

if __name__ == "__main__":
    runner()