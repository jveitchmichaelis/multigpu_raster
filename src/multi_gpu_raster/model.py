import lightning.pytorch as pl
import numpy as np
import torch
import torchvision.models as models
from torch.distributed import all_gather, get_rank, get_world_size


class LoggingModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.gpu_stats = {
            "max_memory": 0,
            "total_memory": 0,
            "iterations": 0,
            "utilization": [],
        }

    def sample_memory(self):
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(device)
            max_memory_allocated = torch.cuda.max_memory_allocated(device)

            utilization = (
                memory_allocated / torch.cuda.get_device_properties(device).total_memory
            )

            self.gpu_stats["max_memory"] = max(
                self.gpu_stats["max_memory"], max_memory_allocated
            )
            self.gpu_stats["total_memory"] += memory_allocated
            self.gpu_stats["iterations"] += 1
            self.gpu_stats["utilization"].append(utilization)

    def on_predict_end(self):
        if torch.cuda.is_available():
            world_size = get_world_size()
            rank = get_rank()

            avg_utilization = sum(self.gpu_stats["utilization"]) / len(
                self.gpu_stats["utilization"]
            )

            stats_tensor = torch.tensor(
                [
                    self.gpu_stats["max_memory"],
                    self.gpu_stats["total_memory"],
                    self.gpu_stats["iterations"],
                    avg_utilization,
                ],
                device=self.device,
                dtype=torch.float64,
            )

            gathered_stats = [torch.zeros_like(stats_tensor) for _ in range(world_size)]
            all_gather(gathered_stats, stats_tensor)

            if rank == 0:
                print("GPU Usage Summary:")
                for i, stat_tensor in enumerate(gathered_stats):
                    max_memory, total_memory, iterations, avg_util = (
                        stat_tensor.tolist()
                    )

                    avg_memory = total_memory / iterations
                    total_gpu_memory = torch.cuda.get_device_properties(i).total_memory

                    avg_memory_gb = avg_memory / (1024**3)
                    max_memory_gb = max_memory / (1024**3)

                    avg_memory_percent = (avg_memory / total_gpu_memory) * 100
                    max_memory_percent = (max_memory / total_gpu_memory) * 100
                    avg_util_percent = avg_util * 100

                    print(f"GPU {i}:")
                    print(
                        f"  Average Memory Usage per Batch: {avg_memory_gb:.2f} GB ({avg_memory_percent:.2f}%)"
                    )
                    print(
                        f"  Peak Memory Usage: {max_memory_gb:.2f} GB ({max_memory_percent:.2f}%)"
                    )
                    print(f"  Average Utilization per Batch: {avg_util_percent:.2f}%")


# PyTorch Lightning model
class ResNet50Prediction(LoggingModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")
        self.model.eval()

    def forward(self, x):
        res = self.model(x)
        self.sample_memory()
        return res


class ObjectDetector(LoggingModule):
    def __init__(self):
        super().__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.eval()

    def forward(self, x):
        res = self.model(x)
        self.sample_memory()
        return res
