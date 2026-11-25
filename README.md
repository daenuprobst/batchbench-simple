# batchbench-simple

A submission for the [BatchBench challenge](https://huggingface.co/spaces/ai-lab-jku/jumpcp-challenge) that uses no neural network. It computes classic image features, normalises them per plate, and classifies with LDA and gradient boosting. It runs on CPU.

Forked from [ml-jku/batchbench-baselines](https://github.com/ml-jku/batchbench-baselines) for the submission interface. The baseline code is removed.

![Architecture](architecture.png)

## Method

Each 500x500 image is cut into four 250x250 tiles. Nuclei are segmented with Otsu on the DNA channel, cells with Otsu on the sum of the other four channels. If the two Otsu classes of the DNA channel are less than 5 background standard deviations apart, the tile is treated as having no nuclei.

308 features are computed per tile. For each channel these are the intensity inside cells, nuclei, cytoplasm, and rings around the nuclei, plus band-pass energies, spot counts, LBP histograms, and GLCM texture. On top of that come correlations between channels and the count, size, shape, and spacing of nuclei. See `features.py`.

Features are z-scored within each plate (median and MAD), averaged over the four tiles, and whitened within each source. This needs no labels, so the test images go through the same steps.

The data has two kinds of plates. Compound plates hold the 8 positive controls in 4 wells each and a few rare compounds. Target plates hold about 300 compounds once each. Plates with fewer than 250 images go to a model for the 8 controls (LDA and gradient boosting). Plates with 250 or more images go to a model with 301 classes (LDA). If that model estimates that more than 20% of a large plate is controls, the plate goes to the 8-control model instead.

## Results

Trained on JUMPCP-2k-500 only. No pretraining, no external data, and nothing from sources 4, 6, 9, or 13.

Validation on source 2 after training on the other eight public sources, as the baselines do (`python train.py`):

| | POSCON | TARGET | REST | Overall | Test mix |
|---|---|---|---|---|---|
| ARM-BN ResNet18 | 0.843 | 0.063 | 0.006 | 0.444 | 0.340 |
| CS-ARM-BN ResNet18 | 0.839 | 0.052 | 0.005 | 0.439 | 0.332 |
| This repo | 0.815 | 0.195 | 0.000 | 0.458 | 0.402 |

The baseline rows are the mean over three seeds, read from the validation metrics stored in their released checkpoints. "Test mix" weights the three subsets as in the private test set (36.2% POSCON, 54.9% TARGET, and 8.9% REST).

Scores on source 2 are much higher than on the private sources. The leaderboard on 25 November 2026:

| | POSCON | TARGET | REST | Overall |
|---|---|---|---|---|
| CS-ARM-BN-Ens-TTA | 0.694 | 0.033 | 0.003 | 0.269 |
| CS-ARM-BN | 0.663 | 0.035 | 0.005 | 0.259 |
| ARM-BN | 0.662 | 0.034 | 0.003 | 0.258 |

Leaving out one public source at a time (`python evaluate.py --clf ...`), mean over the nine sources. TARGET is the mean over the seven sources that have target plates.

| | POSCON | TARGET | Test mix |
|---|---|---|---|
| Random forest (`--clf rf`) | 0.639 | 0.077 | 0.273 |
| LDA, one model for all plates (`--clf lda --no-route`) | 0.631 | 0.096 | 0.281 |
| LDA (`--clf lda`) | 0.678 | 0.129 | 0.316 |
| LDA and gradient boosting (`--clf ens`), submitted | 0.698 | 0.129 | 0.323 |

The submitted variant ranges from 0.252 (source 11) to 0.402 (source 2).

## Known problems

- The 250-image rule comes from the public plates, where compound plates have at most 202 images and target plates at least 314. Plates in the private sources have other sizes. A plate sent to the wrong model loses most of its accuracy. Plates with 203 to 313 images are reported on stderr.
- The check on large plates was only tested on simulated plates, made by pooling ten public compound plates into one. Control accuracy on those went from 0.19-0.26 to 0.58-0.65, with 0.73 as the best possible, and none of the 76 public target plates were moved. Large plates of mostly rare compounds are not caught, and neither are target plates with fewer than 250 images.
- The two kinds of plates cannot be told apart from image content alone with these features.
- Rare compounds are never predicted on compound plates, so REST is 0 there.
- Features, pooling, whitening, classifiers, and the 250-image rule were all chosen on the public sources, including source 2.
- Images with a blank channel get no special treatment.

## Reproduce

Python 3.13.

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r requirements.txt

# metadata
hf download ai-lab-jku/jumpcp-challenge-datasets --repo-type dataset \
  --include "2k/200x200/train_domain_data.csv" --local-dir data

# features. Streams the 119 GB image file and keeps 470 MB, about 70 min on 24 threads
.venv/bin/python features.py \
  https://huggingface.co/datasets/ai-lab-jku/jumpcp-challenge-datasets/resolve/main/2k/500x500/train_data_images.npy \
  feats/train500.npy --tiles

.venv/bin/python evaluate.py --clf ens   # leave one source out
.venv/bin/python train.py                # writes models/model.joblib, prints the source 2 validation
docker build -t batchbench-simple .
```

`python features.py` without arguments runs a small self-check.

## Submission

`submission.py` has the `Submission` class with `adapt()` and `predict()`. `adapt()` computes the features and `predict()` returns an `(N, 2000)` float32 array of class probabilities in input order. Negative controls are accepted and ignored. Feature extraction takes about 49 ms per image on 24 threads, so about 40 minutes for the 46,150 test images.
