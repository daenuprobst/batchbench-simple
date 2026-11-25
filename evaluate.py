"""Lazy, generated docstrings again..."""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

TEST_MIX = {"poscon": 16691 / 46150, "target": 25352 / 46150, "rest": 4107 / 46150}
TGT_MIN_IMAGES = 250
UNSEEN_COUNTS = range(203, 314)
MAX_CONTROL_SHARE = 0.2


def _groups(keys):
    return pd.Series(np.arange(len(keys))).groupby(keys).indices.values()


def normalise(F, plates):
    """Robust z-score of every feature within its plate."""
    Z = np.zeros_like(F)

    for idx in _groups(plates):
        med = np.median(F[idx], 0)
        mad = 1.4826 * np.median(np.abs(F[idx] - med), 0)
        Z[idx] = (F[idx] - med) / (mad + 1e-3 * np.abs(med) + 1e-6)

    return np.clip(Z, -20, 20)


def whiten(Z, groups, shrink=0.2):
    """Whiten each group with its own shrunk covariance."""
    W = np.zeros_like(Z)
    for idx in _groups(groups):
        # a single image has no covariance, leave it at zero
        if len(idx) < 2:
            continue

        X = Z[idx] - Z[idx].mean(0)
        C = np.cov(X.T)
        C = (1 - shrink) * C + shrink * np.eye(len(C)) * np.trace(C) / len(C)
        w, V = np.linalg.eigh(C)
        W[idx] = X @ (V / np.sqrt(w)) @ V.T

    return W


def _plate_key(source, plate):
    # plate ids repeat across sources
    return np.char.add(np.char.add(source.astype(str), "/"), plate.astype(str))


def control_share(P, ctrl, prior):
    """Estimate the share of control images on one plate with EM.

    P are the target model's class probabilities for the plate's images, ctrl marks
    the control classes, and prior is the share of controls in the training data.
    """
    a, b = P[:, ctrl].sum(1) / prior, P[:, ~ctrl].sum(1) / (1 - prior)
    share = 0.5

    for _ in range(200):
        share = np.mean(share * a / (share * a + (1 - share) * b + 1e-300))

    return share


def recheck(tgt_model, controls, Z, W, tgt, source, plate):
    """Send large plates that are mostly controls to the compound model.

    Returns the updated W and tgt. Runs until no more plates are moved, since the
    whitening changes each time.
    """
    ctrl = np.isin(tgt_model.classes_, controls)
    prior, key = tgt_model.priors_[ctrl].sum(), _plate_key(source, plate)

    while tgt.any():
        rows, P = np.flatnonzero(tgt), tgt_model.predict_proba(W[tgt])
        out = [
            rows[i]
            for i in _groups(key[tgt])
            if control_share(P[i], ctrl, prior) > MAX_CONTROL_SHARE
        ]

        if not out:
            break

        print(
            f"NOTE: {len(out)} large plates are mostly controls, sent to the compound model",
            file=sys.stderr,
        )
        tgt = tgt.copy()
        tgt[np.concatenate(out)] = False
        W = whiten(Z, np.char.add(source.astype(str), tgt.astype(str)))

    return W, tgt


def profiles(F, source, plate, route=True):
    """Turn features into z-scores Z, whitened profiles W, and a target-plate flag.

    F has shape (images, quadrants, F) or (images, F). No labels are used.
    """
    F = F[:, None] if F.ndim == 2 else F
    N, T, nf = F.shape
    plate = _plate_key(source, plate)
    Z = normalise(F.reshape(-1, nf), np.repeat(plate, T)).reshape(N, T, nf).mean(1)
    _, inverse, counts = np.unique(plate, return_inverse=True, return_counts=True)
    tgt = counts[inverse] >= TGT_MIN_IMAGES
    odd = np.isin(counts, UNSEEN_COUNTS).sum()

    if odd:
        print(
            f"WARNING: {odd} plates have an image count never seen in the public data, "
            "their layout is a guess",
            file=sys.stderr,
        )
    return (
        Z,
        whiten(
            Z, np.char.add(source.astype(str), tgt.astype(str)) if route else source
        ),
        tgt,
    )


def load(feats, route=True):
    """Load the public metadata and features. Returns (table, Z, W)."""
    d = pd.read_csv("data/2k/200x200/train_domain_data.csv").rename(
        columns={"Metadata_Source": "source"}
    )

    F = np.load(feats)
    assert len(F) == len(
        d
    ), "features and metadata are paired by row position and must match in length"

    Z, W, d["tgt_plate"] = profiles(
        F, d.source.values.astype(str), d.Metadata_Plate.values.astype(str), route
    )

    return d, Z, W


def proba(clf, X, y, a, b):
    clf.fit(X[a], y[a])
    return clf.classes_, clf.predict_proba(X[b])


def lda():
    return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")


def hgb():
    return HistGradientBoostingClassifier(
        max_iter=300, early_stopping=True, random_state=0
    )


def predict(model, Z, W, tgt, source, plate, n_classes=2000):
    """Class probabilities, shape (N, n_classes), from the models that train.py fits."""
    W, tgt = recheck(
        model["tgt_lda"], model["cmp_lda"].classes_, Z, W, tgt, source, plate
    )
    out = np.zeros((len(Z), n_classes), np.float32)

    if (~tgt).any():  # compound plates: the 8 positive controls
        P = model["cmp_lda"].predict_proba(W[~tgt]) + model["cmp_hgb"].predict_proba(
            Z[~tgt]
        )
        out[np.ix_(~tgt, model["cmp_lda"].classes_)] = P / 2

    if tgt.any():  # target plates: 293 target compounds + the 8 controls
        out[np.ix_(tgt, model["tgt_lda"].classes_)] = model["tgt_lda"].predict_proba(
            W[tgt]
        )

    return out


def accuracy(d, pred, rows):
    """Accuracy per evaluation subset over the given rows."""
    ok = pred == d.label.values
    return {
        g: (
            ok[rows & (d.group == g).values].mean()
            if (rows & (d.group == g).values).any()
            else np.nan
        )
        for g in TEST_MIX
    }


def run(d, Z, W, held, kind, route=True):
    tr, te = (d.source != held).values, (d.source == held).values
    tgt, pos, y = d.tgt_plate.values, (d.group == "poscon").values, d.label.values

    # the guard on large plates, as submitted
    if route and kind != "rf" and (tr & tgt).any() and (te & tgt).any():
        W, tgt = W.copy(), tgt.copy()
        W[te], tgt[te] = recheck(
            lda().fit(W[tr & tgt], y[tr & tgt]),
            np.unique(y[pos]),
            Z[te],
            W[te],
            tgt[te],
            d.source.values[te].astype(str),
            d.Metadata_Plate.values[te].astype(str),
        )

    pred = np.full(len(d), -1)

    # (layout mask, training rows, forest leaf size, 300-class leaf tables must fit in RAM, at least on my humble setup)
    jobs = [(~tgt, tr & ~tgt & pos, 1), (tgt, tr & tgt, 2)]

    # one model for all plates, 301 classes
    if not route:
        jobs = [(np.ones(len(d), bool), tr & (d.group != "rest").values, 2)]

    for layout, a, leaf in jobs:
        b = te & layout
        if not b.any():
            continue

        if kind == "rf":
            cls, P = proba(
                RandomForestClassifier(
                    300, min_samples_leaf=leaf, n_jobs=-1, random_state=0
                ),
                Z,
                y,
                a,
                b,
            )
        else:
            cls, P = proba(lda(), W, y, a, b)
            # boosting 300 classes costs 300 trees per round (controls only)
            if kind == "ens" and leaf == 1:
                P = P + proba(hgb(), Z, y, a, b)[1]

        pred[b] = cls[P.argmax(1)]

    return accuracy(d, pred, te)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="feats/train500.npy")
    ap.add_argument(
        "--clf",
        default="lda",
        choices=["lda", "ens", "rf"],
        help="lda: whitened LDA; ens: LDA + gradient boosting on controls; rf: random forest only",
    )
    ap.add_argument("--sources", default="")
    ap.add_argument(
        "--no-route",
        action="store_true",
        help="one model for all plates, no layout assumption",
    )
    a = ap.parse_args()

    # heads-up, a plate that is 30% confident controls is estimated as 30% controls
    _P = np.full((400, 4), 0.005)
    _P[:120, 0], _P[120:, 2] = 0.985, 0.985
    assert (
        abs(control_share(_P, np.array([True, True, False, False]), 0.5) - 0.3) < 0.01
    )

    d, Z, W = load(a.feats, route=not a.no_route)

    rows = {}
    for s in a.sources.split(",") if a.sources else sorted(d.source.unique()):
        rows[s] = run(d, Z, W, s, a.clf, route=not a.no_route)
        print(s, {k: round(float(v), 3) for k, v in rows[s].items()}, flush=True)

    r = pd.DataFrame(rows).T

    r["overall@testmix"] = sum(
        r[g].fillna(r[g].mean()) * w for g, w in TEST_MIX.items()
    )

    print(r.round(3).to_string())
    print("MEAN", r.mean().round(3).to_dict())
