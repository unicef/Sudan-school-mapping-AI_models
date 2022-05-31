from src.tf.models.common import BaseModel
from src.tf.models.resnet18 import ResNet18
from src.tf.models.resnet50 import ResNet50
from src.tf.models.resnet101 import ResNet101
from src.tf.models.object_localizer import ObjectLocalizer
from src.tf.models.object_classifier import ObjectClassifier
from src.tf.models.transformer import TransformerBinaryClassifier


def get_model_constructor(model_path: str) -> BaseModel:
    """
    Returns a specific class that knows how to load a saved model from the specifed path.

    :param model_path: a path to the saved model that is either saved in the HDF5 or in the SavedModel format.
    :return: a class that will be used for loading the saved model architecture and running the prediction.
    """
    available_model_constructors = {
        "cnn": ObjectLocalizer,
        "vgg19": ObjectClassifier,
        "resnet18": ResNet18,
        "resnet50": ResNet50,
        "resnet101": ResNet101,
        "trans": TransformerBinaryClassifier
    }

    for keyword, model_ctr in available_model_constructors.items():
        if keyword in model_path:
            return model_ctr

    raise Exception(f"Did not find a model constructor for the given path: {model_path}")