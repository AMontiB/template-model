import logging
from typing import Type

from PIL.Image import Image
from pandas import DataFrame
from tqdm import tqdm

from model_helpers.data_loader.common import AbstractDataLoader
from model_helpers.post_process.common import AbstractPostProcessor
from model_helpers.utils.codecs import dataframe_to_pil
from model_helpers.utils.device import get_device
from model_helpers.wrapper import ModelWrapper as BaseWrapper


logger = logging.getLogger(__name__)


class ModelWrapper(BaseWrapper):

    def __init__(self,
                 loader_class: Type[AbstractDataLoader],
                 post_processors_class: Type[AbstractPostProcessor],
                 **kwargs
                 ):
        super().__init__(
            loader_class=loader_class,
            post_processors_class=post_processors_class,
            **kwargs
        )
        self.model = None
        self.device = None
        self.transform = None

        # DO NOT initialize the model in __init__ method, the model should be initialized in load_context method (https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel.load_context)

    def load_context(self, context):
        super().load_context(context)

        # TODO properly load the model for prediction in model server. look at the example below
        
        # logger.info(f"Calling load_context with context: {context}")

        # logger.info(f"artifacts: {context.artifacts}")

        # artifact_path = context.artifacts["checkpoint"]

        # self.load_model(artifact_path)


    # def load_model(self, artifact_path):
    #     self.device = get_device(None)
    #     logger.info(f"device {self.device}")

    #     self.model = get_network(
    #         model=self._model_class,
    #         dropout=None,
    #         task=self.task,
    #         freeze=self.freeze  
    #     )
    #     self.transform = build_transform()


    #     logger.info(f"Loading model from: {artifact_path}")
    #     checkpoint = torch.load(artifact_path, map_location=self.device)

    #     self.model.load_state_dict(checkpoint['model_state_dict'])
    #     self.model.to(self.device)
    #     self.model.eval()

    def run_prediction(self, image: Image | DataFrame):

        # TODO run prediction using the loaded model, look at the example below

        # if not isinstance(image, Image):
        #     image = dataframe_to_pil(image, "RGB")

        # image = self.transform(image)
        # test_loader = DataLoader(dataset=image[None,], batch_size=1, shuffle=False, num_workers=0)
        # with torch.no_grad():
        #     with tqdm(test_loader, unit='batch', mininterval=0.5) as tbatch:
        #         tbatch.set_description(f'Test')
        #         for data in tbatch:
        #             data = data.to(self.device)
        #             scores = self.model(data)

        # # postprocess result
        # if type(scores) == tuple:
        #     _, scores = scores
        # print('prob: ', torch.sigmoid(scores))
        # pred = torch.where((scores) > -2.38, torch.tensor(1), torch.tensor(0)).to(torch.int)
        # print('prediction: ', pred)

        # return pred.cpu().numpy()[0][0]
