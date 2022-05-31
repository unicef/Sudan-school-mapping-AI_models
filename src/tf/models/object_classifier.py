from src.tf.models.common import BaseModel


class ObjectClassifier(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def train(self, *args, **kwargs):
        pass


    def build(self):
        pass


    def get_predicted_probability(self,  prediction_result):
        return prediction_result[0][0]



