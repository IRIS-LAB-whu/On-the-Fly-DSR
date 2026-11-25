import os
import sys
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None


class SupSceneFeatureExtractor:
    """SupScene DINOv2-SCPP global descriptor extractor.

    The C++ caller expects an Extract(imagePath) method returning a numpy array.
    Keep the returned dtype as float64 because the existing bridge reads it as
    a double buffer before converting to float.
    """

    def __init__(self, weights_path=None, device="cuda"):
        self.repo_dir = Path(__file__).resolve().parent
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        if weights_path is None or str(weights_path).strip() == "" or str(weights_path).lower() == "auto":
            weights_path = self.repo_dir / "checkpoints" / "dinov2_scpp_supscene_1536.pth"
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"SupScene weights not found: {self.weights_path}")

        requested_device = str(device or "cuda").lower()
        if requested_device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(requested_device)
        else:
            self.device = torch.device("cpu")

        self.model = torch.hub.load(
            str(self.repo_dir),
            "dinov2_scpp_supscene_1536",
            pretrained=True,
            weights=str(self.weights_path),
            source="local",
            map_location=str(self.device),
        )
        self.model.eval().to(self.device)

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def Extract(self, imagePath):
        image_path = Path(str(imagePath).replace("\\", "/"))
        if not image_path.exists():
            raise FileNotFoundError(f"Image does not exist: {image_path}")

        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            descriptor = self.model(tensor)
            if isinstance(descriptor, dict):
                descriptor = descriptor.get(
                    "global_descriptor",
                    descriptor.get("desc", next(iter(descriptor.values()))),
                )
            descriptor = descriptor.detach().cpu().numpy()

        descriptor = np.asarray(descriptor, dtype=np.float64)
        descriptor = descriptor.reshape(1, -1)
        if descriptor.shape[1] != 1536:
            raise RuntimeError(f"Unexpected SupScene descriptor dimension: {descriptor.shape}")
        return np.ascontiguousarray(descriptor)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract a SupScene global descriptor.")
    parser.add_argument("image")
    parser.add_argument("--weights", default="auto")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    extractor = SupSceneFeatureExtractor(args.weights, args.device)
    feature = extractor.Extract(args.image)
    print(feature.shape)
