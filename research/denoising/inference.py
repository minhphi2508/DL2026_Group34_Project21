"""Denoising API: RGB uint8 HxWx3 -> RGB uint8 HxWx3.

No ground truth, severity label, or dataset metadata is used at inference.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
WEIGHTS = {
    "ffdnet_gray": "ffdnet_gray_clip.pth",
    "dncnn_gray": "dncnn_gray_blind.pth",
    "ffdnet_color": "ffdnet_color_clip.pth",
}
CANDIDATES = (*WEIGHTS, "nlm_luma", "identity")


def _validate(image):
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be numpy.ndarray")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be RGB uint8 HxWx3")
    if min(image.shape[:2]) < 1:
        raise ValueError("image dimensions must be positive")
    return np.ascontiguousarray(image)


def luminance(image):
    return image.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def estimate_sigma(image):
    """Robust diagonal Haar MAD estimate, in uint8 intensity units.

    A difference of four neighbors, divided by two, has unit Gaussian-noise
    variance. MAD limits the influence of edges/scratches and sparse specks.
    Native photo grain can inflate this estimate; this is discussed in the notes.
    """
    y = luminance(image)
    if min(y.shape) < 2:
        return 2.5
    detail = (y[:-1, :-1] - y[:-1, 1:] - y[1:, :-1] + y[1:, 1:]) / 2
    value = np.median(np.abs(detail - np.median(detail))) / 0.67448975
    return float(np.clip(value, 2.5, 20.0))


def load_config(path=None):
    import yaml
    with open(path or MODULE_DIR / "config.yaml", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class Denoiser:
    def __init__(self, candidate="ffdnet_gray", device="cpu", weights_dir=None,
                 sigma_scale=1.0, blend=1.0, tile_size=0, threads=2, **_):
        if candidate not in CANDIDATES:
            raise ValueError(f"Unknown candidate {candidate!r}; choose {CANDIDATES}")
        if not 0 <= blend <= 1 or sigma_scale <= 0:
            raise ValueError("blend must be in [0,1] and sigma_scale must be positive")
        self.candidate = candidate
        self.sigma_scale, self.blend = float(sigma_scale), float(blend)
        self.tile_size, self.device = int(tile_size), device
        self.model = None
        self.checkpoint = None
        if candidate in WEIGHTS:
            import torch
            from .networks import FFDNet, DnCNN
            if threads:
                torch.set_num_threads(int(threads))
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; use --device cpu")
            self.device = device
            self.checkpoint = Path(weights_dir or MODULE_DIR / "weights") / WEIGHTS[candidate]
            if not self.checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint missing: {self.checkpoint}. "
                                        "Run models/denoising/download_weights.py first.")
            self.model = (DnCNN(1) if candidate == "dncnn_gray" else
                          FFDNet(3 if candidate == "ffdnet_color" else 1))
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)
            self.model.to(device=device, memory_format=torch.channels_last).eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def _forward(self, tensor, sigma):
        if self.candidate == "dncnn_gray":
            return self.model(tensor)
        return self.model(tensor, sigma)

    def _network(self, image, noise_sigma):
        import torch
        array = (image.astype(np.float32) / 255).transpose(2, 0, 1)
        tensor = torch.from_numpy(array.copy()).unsqueeze(0).to(
            device=self.device, memory_format=torch.channels_last)
        sigma = tensor.new_full((1, 1, 1, 1), noise_sigma / 255)
        with torch.inference_mode():
            tile = self.tile_size
            if tile <= 0 or max(image.shape[:2]) <= tile:
                output = self._forward(tensor, sigma)
            else:
                # CNN receptive-field halo protects central tile predictions.
                # Keep starts even to align FFDNet's 2x2 subimage grid.
                tile = max(32, tile // 2 * 2)
                halo = 32 if self.candidate != "dncnn_gray" else 24
                output = torch.empty_like(tensor)
                h, w = image.shape[:2]
                for top in range(0, h, tile):
                    for left in range(0, w, tile):
                        bottom, right = min(top + tile, h), min(left + tile, w)
                        y0, x0 = max(0, top-halo), max(0, left-halo)
                        y1, x1 = min(h, bottom+halo), min(w, right+halo)
                        patch = self._forward(tensor[..., y0:y1, x0:x1], sigma)
                        output[..., top:bottom, left:right] = patch[..., top-y0:bottom-y0, left-x0:right-x0]
            if not torch.isfinite(output).all():
                raise RuntimeError("Denoiser produced non-finite pixels")
            return output[0].cpu().numpy().transpose(1, 2, 0) * 255

    def __call__(self, image):
        image = _validate(image)
        if self.candidate == "identity":
            return image.copy()
        sigma = estimate_sigma(image) * self.sigma_scale
        if self.candidate == "ffdnet_color":
            restored = self._network(image, sigma)
        else:
            y = luminance(image)
            if self.candidate == "nlm_luma":
                import cv2
                denoised_y = cv2.fastNlMeansDenoising(
                    np.rint(y).astype(np.uint8), None, h=max(1.0, 0.8*sigma),
                    templateWindowSize=7, searchWindowSize=15).astype(np.float32)
            else:
                denoised_y = self._network(y[..., None], sigma)[..., 0]
            # An equal-channel correction preserves RGB color differences.
            restored = image.astype(np.float32) + (denoised_y-y)[..., None]
        restored = (1-self.blend)*image.astype(np.float32) + self.blend*restored
        return np.rint(np.clip(restored, 0, 255)).astype(np.uint8)


def get_denoiser(candidate=None, config_path=None, **options):
    config = load_config(config_path)
    candidate = candidate or config["selected_candidate"]
    params = {**config.get("defaults", {}), **config.get("candidates", {}).get(candidate, {})}
    params.update(options)
    return Denoiser(candidate=candidate, **params)


@lru_cache(maxsize=1)
def _default():
    return get_denoiser()


def denoise(image):
    return _default()(image)
