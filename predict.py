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
          image: Path = Input(description="Original input image"),
          face_enhance: bool = Input(description="Whether or not to enable face enhancement", default=True)
    ) -> Path:
        if(face_enhance):
            print("Enhancing faces")
            face_enhance_string = "--face_enhance"
        else:
            face_enhance_string = ""
            print("Not enhancing faces")
        response = os.popen(f"python ./inference_realesrgan.py -i {image} -o ./ --suffix out --face_enhance {face_enhance_string}").read().strip()
        response = os.path.join('./',response)
        response_search = re.search('RESPONSE(.*)RESPONSE', response, re.IGNORECASE)
        if response_search:
            response = response_search.group(1)
            print(f'Response - {response}')
            return Path(response)
