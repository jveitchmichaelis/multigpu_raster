import lightning.pytorch as pl
import torchvision.models as models


# PyTorch Lightning model
class ResNet50Prediction(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        self.model.eval()

    def forward(self, x):
        return self.model(x)

class ObjectDetector(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def forward(self, x):
        return self.model(x)
    