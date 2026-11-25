import io
import sys
import time
import urllib.request

import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_otsu
from skimage.measure import regionprops_table

GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation")
SHAPE_PROPS = ("area", "eccentricity", "solidity")

# px from the nearest nuclear edge, tuned for 200 px crops
RINGS = (
    0,
    3,
    7,
    12,
    20,
    np.inf,
)

PER_CHANNEL = 53

# between Otsu's two classes, in background standard deviations
MIN_SEPARATION = 5.0

# index of the nucleus count in the output vector
N_NUCLEI = 5 * PER_CHANNEL + 20


def _otsu(a):
    return a > threshold_otsu(a) if a.max() > a.min() else np.zeros(a.shape, bool)


def _nuclei(a):
    """Otsu's split of the DNA stain, kept only if its two classes are really apart."""
    if a.max() <= a.min():
        return np.zeros(a.shape, bool)

    m = a > threshold_otsu(a)
    lo = a[~m]
    if a[m].mean() - lo.mean() < MIN_SEPARATION * (lo.std() + 1e-6):
        return np.zeros(a.shape, bool)

    return m


def _mean(a, mask):
    return a[mask].mean() if mask.any() else 0.0


def feats(img):
    """Feature vector for one uint8 (H, W, 5) image."""
    x = img.astype(np.float32)
    sm = ndi.gaussian_filter(x, (1, 1, 0))

    nuc = ndi.binary_fill_holes(ndi.binary_opening(_nuclei(sm[..., 0])))
    lab, n = ndi.label(nuc)
    cell = _otsu(sm[..., 1:].sum(-1)) | nuc

    # empty field, use the whole frame
    if cell.sum() < 100:
        cell = np.ones_like(cell)

    cyto = cell & ~nuc
    d_out = ndi.distance_transform_edt(~nuc) if n else np.full(nuc.shape, np.inf)
    d_in = ndi.distance_transform_edt(nuc)
    rings = [cell & (d_out > RINGS[i]) & (d_out <= RINGS[i + 1]) for i in range(5)]

    f = []
    for c in range(5):
        raw = x[..., c]
        bg = np.median(raw[~cell]) if (~cell).sum() > 100 else np.percentile(raw, 1)
        a = raw - bg
        ac = a[cell]
        mc = max(ac.mean(), 1.0)
        z = (ac - ac.mean()) / (ac.std() + 1e-6)
        f += [a.mean(), mc, bg, ac.std() / mc, (z**3).mean(), (z**4).mean()]
        f += list(np.percentile(ac, [10, 50, 90, 99]) / mc)

        # where the stain sits. Nucleus vs cytoplasm, nuclear interior vs rim, rings around nuclei
        f += [
            _mean(a, nuc) / mc,
            _mean(a, cyto) / mc,
            _mean(a, nuc & (d_in > 3)) / (abs(_mean(a, nuc)) + 1.0),
        ]
        f += [_mean(a, r) / mc for r in rings]

        # band-pass energies inside cells. Something along the lines of the CellProfiler granularity spectra
        g = [a] + [ndi.gaussian_filter(a, s) for s in (1, 2, 4, 8)]
        f += [(g[i] - g[i + 1])[cell].std() / mc for i in range(4)]
        f.append(ndi.gaussian_gradient_magnitude(a, 1)[cell].mean() / mc)

        # spots (nucleoli, granules, mitochondrial puncta). The bright local maxima of a band-pass
        dog = g[1] - g[2]
        thr = 3 * 1.4826 * np.median(np.abs(dog[cell] - np.median(dog[cell]))) + 1e-6
        peaks = (dog == ndi.maximum_filter(dog, 5)) & (dog > thr)
        f += [
            1e3 * (peaks & cell).sum() / cell.sum(),
            1e3 * (peaks & nuc).sum() / (nuc.sum() + 1.0),
        ]

        # LBP histograms. Should be invariant to any monotonic intensity change, such as gain and exposure
        for r in (1, 3):
            codes = local_binary_pattern(img[..., c], 8, r, "uniform")[cell].astype(int)
            f += list(np.bincount(codes, minlength=10) / cell.sum())

        # Haralick-style texture on the cells' own intensity range
        # background-background bin dropped
        lo, hi = np.percentile(ac, [1, 99])
        q = np.clip((a - lo) / (hi - lo + 1e-6) * 14 + 1, 1, 15).astype(np.uint8) * cell
        glcm = graycomatrix(
            q, (1, 3), (0, np.pi / 2), levels=16, symmetric=True
        ).astype(np.float64)
        glcm[0, 0] = 0
        glcm /= glcm.sum((0, 1), keepdims=True) + 1e-9

        for p in GLCM_PROPS:
            f += list(graycoprops(glcm, p).mean(axis=1))

    # a blank channel gives NaN, zeroed at the end
    with np.errstate(invalid="ignore", divide="ignore"):
        f += list(np.corrcoef(x.reshape(-1, 5).T)[np.triu_indices(5, 1)])
        f += list(np.corrcoef(x[cell].T)[np.triu_indices(5, 1)])

    f += [n, nuc.mean(), cell.mean(), cell.sum() / max(n, 1), nuc.sum() / cell.sum()]

    if n:
        rp = regionprops_table(lab, properties=SHAPE_PROPS)
        for p in SHAPE_PROPS:
            f += [rp[p].mean(), rp[p].std(), np.median(rp[p])]

        f.append(rp["area"].max())
        idx = np.arange(1, n + 1)
        f.append(
            (
                ndi.standard_deviation(x[..., 0], lab, idx)
                / (ndi.mean(x[..., 0], lab, idx) + 1e-6)
            ).mean()
        )

        for c in range(5):
            om = ndi.mean(x[..., c], lab, idx)
            f.append(om.std() / (abs(om.mean()) + 1e-6))

        if n > 1:
            cen = np.array(ndi.center_of_mass(nuc, lab, idx))
            d = np.sqrt(((cen[:, None] - cen[None]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            f += [d.min(1).mean(), d.min(1).std()]
        else:
            f += [0.0, 0.0]
    else:
        f += [0.0] * (3 * len(SHAPE_PROPS) + 1 + 1 + 5 + 2)

    return np.nan_to_num(np.asarray(f, np.float32), posinf=0.0, neginf=0.0)


def tiles(img):
    """The four quadrants of an image."""
    h, w = img.shape[0] // 2, img.shape[1] // 2

    return [img[:h, :w], img[:h, w:], img[h:, :w], img[h:, w:]]


def _fetch(url, lo, hi):
    """Bytes [lo, hi) of a remote file."""
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi - 1}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                b = r.read()
            if len(b) == hi - lo:
                return b
        except OSError:
            pass

        time.sleep(2**attempt)

    raise OSError(f"could not fetch bytes {lo}-{hi} of {url}")


def _header(src):
    """Shape and data offset of a .npy file, local or behind a URL."""
    f = io.BytesIO(_fetch(src, 0, 4096)) if src.startswith("http") else open(src, "rb")
    with f:
        np.lib.format.read_magic(f)
        shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        assert dtype == np.uint8, dtype
        return shape, f.tell()


def _measure(imgs, tiled):
    return np.stack(
        [np.stack([feats(t) for t in (tiles(im) if tiled else [im])]) for im in imgs]
    )


def _chunk(src, shape, off, lo, hi, tiled):
    # the 500 px set is 119 GB. So we stream it, keeping only features
    if src.startswith("http"):
        n = int(np.prod(shape[1:]))
        imgs = np.frombuffer(_fetch(src, off + lo * n, off + hi * n), np.uint8).reshape(
            -1, *shape[1:]
        )
    else:
        imgs = np.load(src, mmap_mode="r")[lo:hi]

    return _measure(imgs, tiled)


def extract_array(images, tiled=True, n_jobs=-1, chunk=32):
    """Features for an image stack in memory or memory-mapped. Used by submission.py."""
    parts = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(_measure)(np.ascontiguousarray(images[i : i + chunk]), tiled)
        for i in range(0, len(images), chunk)
    )

    out = np.concatenate(parts)

    return out if tiled else out[:, 0]


def extract(src, tiled=False, n_jobs=-1, chunk=64, limit=None):
    """Features for every image, (N, F) or (N, 4, F) with tiled=True."""
    shape, off = _header(src)
    n = limit or shape[0]

    parts = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_chunk)(src, shape, off, i, min(i + chunk, n), tiled)
        for i in range(0, n, chunk)
    )

    out = np.concatenate(parts)

    return out if tiled else out[:, 0]


if __name__ == "__main__":
    # self-check on synthetic blobs. Finite, deterministic, and it counts nuclei, yay
    if len(sys.argv) >= 3:
        np.save(sys.argv[2], extract(sys.argv[1], tiled="--tiles" in sys.argv))
    else:
        rng = np.random.default_rng(0)
        im = np.zeros((200, 200, 5), np.float32)

        for r, c in [(40, 40), (40, 150), (150, 40), (150, 150), (100, 100)]:
            im[r, c] = 1

        im = ndi.gaussian_filter(im, (4, 4, 0))
        im = (255 * im / im.max() * 0.8 + rng.uniform(0, 20, im.shape)).astype(np.uint8)
        v = feats(im)

        assert np.isfinite(v).all() and (v == feats(im)).all()
        assert v[N_NUCLEI] == 5, v[N_NUCLEI]

        # a missing channel (source 15 has no RNA), let's not mess things up
        im[..., 1] = 0
        assert np.isfinite(feats(im)).all()

        # blank frame
        assert np.isfinite(feats(np.zeros((200, 200, 5), np.uint8))).all()

        # no nuclei here, plain Otsu would still segment this
        noise = rng.integers(8, 14, (200, 200, 5)).astype(np.uint8)
        assert feats(noise)[N_NUCLEI] == 0

        print("ok,", len(v), "features")
