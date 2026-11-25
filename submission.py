"""BatchBench submission: hand-crafted image features, classical batch correction, LDA + gradient boosting.

No neural network and no GPU. Needs only the unlabeled test images and their
(source, batch, plate) ids. Negative controls, if passed, are ignored.

Validation, see evaluate.py.
Training, see train.py.

docstrings are generated.
"""

from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np

from evaluate import predict, profiles
from features import extract_array

NUM_CLASSES = 2000
MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.joblib"


def _ids(metadata):
    """Source and plate of every row. Rows are (source, batch, plate) tuples or dicts."""
    rows = [
        (
            (r["Metadata_Source"], r["Metadata_Plate"])
            if isinstance(r, Mapping)
            else (r[0], r[2])
        )
        for r in metadata
    ]

    ids = np.array(rows, dtype=str).reshape(-1, 2)

    return ids[:, 0], ids[:, 1]


def _fingerprint(images):
    return (
        len(images),
        int(np.asarray(images[0]).sum()),
        int(np.asarray(images[-1]).sum()),
    )


class Submission:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self._features = None  # (fingerprint of the image stack, its features)

    def _measure(self, images):
        key = _fingerprint(images)

        if self._features is None or self._features[0] != key:
            self._features = key, extract_array(images)

        return self._features[1]

    def adapt(self, test_images, test_metadata, *args, **kwargs) -> None:
        """Compute the features once. The normalisation happens in predict()."""
        if len(test_images):
            self._measure(test_images)

    def predict(self, test_images, test_metadata, *args, **kwargs) -> np.ndarray:
        """(N, 2000) float32 class probabilities, rows in the order of test_images."""
        if len(test_images) != len(test_metadata):
            raise ValueError("test_images and test_metadata must have equal length.")

        if not len(test_images):
            return np.zeros((0, NUM_CLASSES), np.float32)

        source, plate = _ids(test_metadata)
        Z, W, tgt = profiles(self._measure(test_images), source, plate)
        out = predict(self.model, Z, W, tgt, source, plate, NUM_CLASSES)

        # no exact zeros
        out += 1e-6

        return out / out.sum(1, keepdims=True)
