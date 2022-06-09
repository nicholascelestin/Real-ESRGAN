from cog import BasePredictor, Input, Path
import torch
import inference_realesrgan
import os

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
#         self.model = torch.load("./weights.pth")

    # The arguments and types the model takes as input
    def predict(self,
          image: Path = Input(description="Grayscale input image")
    ) -> Path:
        response = os.popen(f"python ./inference_realesrgan.py -i {image} -o ./ --suffix out").read().strip()
        response = os.path.join('./',response)
        print(f'Response - {response}')
        return Path(response)
#         inference_realesrgan.main()