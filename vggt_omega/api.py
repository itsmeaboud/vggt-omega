import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from jaxtyping import Float32

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.geometry import unproject_depth_map_to_point_map
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass
class VGGTOutput:
    points: Float32[np.ndarray, "S H W 3"]
    images: Float32[np.ndarray, "S H W 3"]
    depth: Float32[np.ndarray, "S H W 1"]
    confidence: Float32[np.ndarray, "S H W"]
    intrinsic: Float32[np.ndarray, "S 3 3"]
    extrinsic: Float32[np.ndarray, "S 3 4"]
    shape: tuple[int, int]
    frames: int


def load_model(
    checkpoint_path: PathLike,
    device: str | torch.device | None = None,
) -> VGGTOmega:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model to %s", device)

    model = VGGTOmega().to(device).eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    return model


def run_vggt(
    images_path: Sequence[PathLike],
    checkpoint_path: PathLike | None = None,
    model: VGGTOmega | None = None,
    device: str | torch.device | None = None,
    image_resolution: int = 512,
) -> VGGTOutput:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when model is not provided")
        model = load_model(checkpoint_path=checkpoint_path, device=device)
    else:
        model = model.to(device).eval()

    images = load_and_preprocess_images(
        images_path,
        image_resolution=image_resolution,
    ).to(device)
    frames = images.size(0)
    height, width = images.shape[-2:]

    with torch.inference_mode():
        logger.info("Processing %d images", frames)
        predictions = model(images)

        extrinsic, intrinsic = encoding_to_camera(
            predictions["pose_enc"],
            (height, width),
        )

    depth = _to_numpy(predictions["depth"])
    confidence = _to_numpy(predictions["depth_conf"])
    extrinsic_np = _to_numpy(extrinsic)
    intrinsic_np = _to_numpy(intrinsic)
    images_np = images.detach().cpu().permute(0, 2, 3, 1).numpy()

    points = unproject_depth_map_to_point_map(
        depth,
        extrinsic_np,
        intrinsic_np,
    )

    return VGGTOutput(
        points=points,
        images=images_np,
        depth=depth,
        confidence=confidence,
        intrinsic=intrinsic_np,
        extrinsic=extrinsic_np,
        shape=(height, width),
        frames=frames,
    )


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    if array.shape[0] == 1:
        array = array[0]
    return array


