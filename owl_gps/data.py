"""Aligned concept-band GeoTIFF dataset with sparse and pseudo-label masks."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def _raster(path: Path) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("Reading geospatial data requires pip install 'owl-gps[geo]'") from exc
    with rasterio.open(path) as src:
        return src.read(1)


class PFASBandDataset(Dataset):
    """Reads band_1/...band_K TIFFs and masks with matching basenames.

    root/
      band_1/<id>_merged.tif ... band_K/<id>_merged.tif
      masks/<id>_mask.tif                  pseudo/expanded training masks
      updated_masks_sparse/<id>_mask.tif  observed ground-truth pixels

    Labels: 0/1 observed binary classes; 2 ignored. At acquisition time
    call features(index), which NEVER reads either target mask. Call
    query(index) only after selecting that index.
    """
    def __init__(self, root: str | Path, num_bands: int, *,
                 means=None, stds=None, pseudo_dir="masks",
                 measured_dir="updated_masks_sparse"):
        if num_bands < 1:
            raise ValueError("num_bands must be positive")
        self.root = Path(root).expanduser()
        self.num_bands = num_bands
        self.pseudo_dir = pseudo_dir
        self.measured_dir = measured_dir
        self.files = sorted((self.root / "band_1").glob("*.tif"))
        if not self.files:
            raise FileNotFoundError(f"No TIFFs in {self.root / 'band_1'}")
        self.means = np.asarray(means if means is not None else [0.] * num_bands, np.float32)
        self.stds = np.asarray(stds if stds is not None else [1.] * num_bands, np.float32)
        if self.means.shape != (num_bands,) or self.stds.shape != (num_bands,):
            raise ValueError("means/stds must each contain num_bands entries")
        if np.any(self.stds <= 0):
            raise ValueError("stds must all be positive")

    def __len__(self):
        return len(self.files)

    def _label_name(self, index):
        return self.files[index].name.replace("_merged.tif", "_mask.tif")

    def features(self, index: int) -> torch.Tensor:
        """Return inputs only: K x 1 x H x W (no access to withheld labels)."""
        base = self.files[index].name
        bands = []
        for n in range(self.num_bands):
            path = self.root / f"band_{n+1}" / base
            if not path.is_file():
                raise FileNotFoundError(path)
            arr = (_raster(path).astype(np.float32) - self.means[n]) / self.stds[n]
            bands.append(torch.from_numpy(arr).unsqueeze(0))
        if len({tuple(x.shape) for x in bands}) != 1:
            raise ValueError(f"Mismatched dimensions for {base}")
        return torch.stack(bands, dim=0)

    def query(self, index: int):
        """Read labels only AFTER an observation has been selected by acquisition."""
        name = self._label_name(index)
        pseudo = self.root / self.pseudo_dir / name
        measured = self.root / self.measured_dir / name
        if not pseudo.is_file() or not measured.is_file():
            raise FileNotFoundError(f"Missing paired masks: {pseudo} or {measured}")
        return (torch.from_numpy(_raster(pseudo).astype(np.int64)),
                torch.from_numpy(_raster(measured).astype(np.int64)))

    def __getitem__(self, index: int):
        pseudo, measured = self.query(index)
        return {
            "images": self.features(index),
            "label": pseudo,
            "observed_label": measured,
            "id": self.files[index].stem,
        }
