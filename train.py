import argparse
from pathlib import Path

import joblib
import numpy as np

from evaluate import TEST_MIX, accuracy, hgb, lda, load, predict

if __name__ == "__main__":
    # Some arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="feats/train500.npy")
    ap.add_argument(
        "--validation-source",
        default="source_2",
        help='"" trains on every public source',
    )
    ap.add_argument("--out", default="models/model.joblib")
    a = ap.parse_args()

    d, Z, W = load(a.feats)
    val, tgt, y = (
        (d.source == a.validation_source).values,
        d.tgt_plate.values,
        d.label.values,
    )

    source, plate = d.source.values.astype(str), d.Metadata_Plate.values.astype(str)

    # compound plates: the 8 positive controls
    controls = ~val & ~tgt & (d.group == "poscon").values

    # target plates: 293 target compounds + the 8 controls
    targets = ~val & tgt
    model = {
        "cmp_lda": lda().fit(W[controls], y[controls]),
        "cmp_hgb": hgb().fit(Z[controls], y[controls]),
        "tgt_lda": lda().fit(W[targets], y[targets]),
    }

    assert (model["cmp_lda"].classes_ == model["cmp_hgb"].classes_).all()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, a.out, compress=3)

    print(
        f"{a.out}: trained on {sorted(d.source[~val].unique())}; {controls.sum()} control images / "
        f"{len(model['cmp_lda'].classes_)} classes, {targets.sum()} target-plate images / "
        f"{len(model['tgt_lda'].classes_)} classes"
    )

    if val.any():
        pred = np.full(len(d), -1)
        pred[val] = predict(
            model, Z[val], W[val], tgt[val], source[val], plate[val]
        ).argmax(1)
        acc = accuracy(d, pred, val)
        print(
            f"validation on {a.validation_source}:",
            {g: round(float(v), 4) for g, v in acc.items()},
            "| overall %.4f" % (pred == y)[val].mean(),
            "| overall at the test mix %.4f"
            % sum(acc[g] * w for g, w in TEST_MIX.items()),
        )
