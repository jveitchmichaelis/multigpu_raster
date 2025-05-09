import lightning.pytorch as pl
import numpy as np
import torch
import torchvision.models as models
from torch.distributed import all_gather, get_rank, get_world_size


class LoggingModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            # Initialize the GPU stats dictionary if not already done
            if not hasattr(self, "gpu_stats"):
                self.gpu_stats = {
                    device: {"max_memory": 0, "total_memory": 0, "iterations": 0}
                    for device in range(torch.cuda.device_count())
                }

            device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(device)
            max_memory_allocated = torch.cuda.max_memory_allocated(device)

            # Ensure GPU stats are initialized for each device
            for device in self.gpu_stats:
                self.gpu_stats[device]["max_memory"] = max(
                    self.gpu_stats[device]["max_memory"], max_memory_allocated
                )
                self.gpu_stats[device]["total_memory"] += memory_allocated
                self.gpu_stats[device]["iterations"] += 1

        return super().on_predict_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_predict_end(self):
        if torch.cuda.is_available():
            # Gather stats from all processes
            world_size = get_world_size()
            rank = get_rank()

            # Collect stats from all ranks
            gathered_stats = [None] * world_size
            stats = self.gpu_stats[torch.cuda.current_device()]

            # Gather all GPU stats from each process
            all_gather(gathered_stats, stats)

            if rank == 0:  # Print only from rank 0 to avoid duplicate logs
                for i, stat in enumerate(gathered_stats):
                    avg_memory = stat["total_memory"] / stat["iterations"]
                    total_memory = torch.cuda.get_device_properties(i).total_memory

                    avg_memory_gb = avg_memory / (1024**3)
                    max_memory_gb = stat["max_memory"] / (1024**3)

                    avg_memory_percent = (avg_memory / total_memory) * 100
                    max_memory_percent = (stat["max_memory"] / total_memory) * 100

                    print(f"GPU {i}:")
                    print(
                        f"  Average Memory Usage per Batch: {avg_memory_gb:.2f} GB ({avg_memory_percent:.2f}%)"
                    )
                    print(
                        f"  Peak Memory Usage: {max_memory_gb:.2f} GB ({max_memory_percent:.2f}%)"
                    )


# PyTorch Lightning model
class ResNet50Prediction(LoggingModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")
        self.model.eval()

    def forward(self, x):
        return self.model(x)


class ObjectDetector(LoggingModule):
    def __init__(self):
        super().__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.eval()
        self.gpu_stats = {
            "max_memory": 0,
            "memory_per_batch": [],
            "avg_memory": 0,
            "iterations": 0,
        }

    def forward(self, x):
        return self.model(x)
