from cog import BasePredictor, Input, Path
import torch
import os
import re


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

    #         self.model = torch.load("./weights.pth")

    # The arguments and types the model takes as input
    def predict(self,
                image: Path = Input(description="Original input image"),
                face_enhance: bool = Input(description="Whether or not to enable face enhancement", default=False),
                model: str = Input(description="Which model to use", default="RealESRGAN_x4plus", choices=["RealESRGAN_x4plus","RealESRGAN_x4plus_anime_6B"]),
                scale: int = Input(description="How much to upscale by?", default=4)
                ) -> Path:
        if (face_enhance):
            print("Enhancing faces")
            face_enhance_string = "--face_enhance"
        else:
            face_enhance_string = ""
            print("Not enhancing faces")

        scaling_string = f'-s {scale}'

        print(f"python ./inference_realesrgan.py -n {model} -i {image} -o ./ --suffix out {face_enhance_string} {scaling_string}")
        response = os.popen(
            f"python ./inference_realesrgan.py -n {model} -i {image} -o ./ --suffix out {face_enhance_string} {scaling_string}").read().strip()
        response = os.path.join('./', response)
        response_search = re.search('RESPONSE(.*)RESPONSE', response, re.IGNORECASE)
        if response_search:
            response = response_search.group(1)
            print(f'Response - {response}')
            print(os.popen("ls").read())
            return Path(response)
