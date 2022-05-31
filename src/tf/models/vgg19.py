from src.tf.models.common import BaseModel


class VGG19BinaryClassifier(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        pass



