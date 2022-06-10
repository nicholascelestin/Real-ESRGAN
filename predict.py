from cog import BasePredictor, Input, Path
import torch
import inference_realesrgan
import os
import re

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
#         self.model = torch.load("./weights.pth")

    # The arguments and types the model takes as input
    def predict(self,
          image: Path = Input(description="Grayscale input image")
    ) -> Path:
        response = os.popen(f"python ./inference_realesrgan.py -i {image} -o ./ --suffix out --face_enhance").read().strip()
        response = os.path.join('./',response)
        response_search = re.search('RESPONSE(.*)RESPONSE', response, re.IGNORECASE)
        if response_search:
            response = response_search.group(1)
            print(f'Response - {response}')
            return Path(response)
