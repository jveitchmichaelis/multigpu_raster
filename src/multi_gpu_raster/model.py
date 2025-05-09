import lightning.pytorch as pl
import torch
import torchvision.models as models


# PyTorch Lightning model
class ResNet50Prediction(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")
        self.model.eval()

    def forward(self, x):
        return self.model(x)


class ObjectDetector(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.eval()
        self.gpu_stats = {"max_memory": 0, "total_memory": 0, "iterations": 0}

    def forward(self, x):
        return self.model(x)

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(device)
            max_memory_allocated = torch.cuda.max_memory_allocated(device)

            # Update GPU stats
            self.gpu_stats["max_memory"] = max(
                self.gpu_stats.get("max_memory", 0), max_memory_allocated
            )
            self.gpu_stats["total_memory"] = (
                self.gpu_stats.get("total_memory", 0) + memory_allocated
            )
            self.gpu_stats["iterations"] = self.gpu_stats.get("iterations", 0) + 1

            # Optionally log the stats
            self.log(
                "gpu_max_memory",
                self.gpu_stats["max_memory"],
                prog_bar=True,
                on_epoch=True,
            )
            self.log(
                "gpu_avg_memory",
                self.gpu_stats["total_memory"] / self.gpu_stats["iterations"],
                prog_bar=True,
                on_epoch=True,
            )

            # Reset peak memory for the next batch
            torch.cuda.reset_peak_memory_stats(device)

        return super().on_predict_batch_end(outputs, batch, batch_idx, dataloader_idx)
