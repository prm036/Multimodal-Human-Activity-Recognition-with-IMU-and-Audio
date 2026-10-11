# Multimodal Human Activity Recognition

Recognizing everyday activities from smartwatch **IMU and audio** recordings using unimodal CNNs, late fusion, and bidirectional cross-attention.

**[Open the notebook →](Multimodal_Human_Activity_Recognition.ipynb)**

The notebook contains the complete experiment: data exploration, preprocessing, model definitions, training, evaluation, plots, and recorded results. It compares four neural models and a random forest activity detector across 15 sessions and 10 target activities plus background.

## Models

| Model | Approach |
|---|---|
| Audio-only | MobileNetV2 on log-mel spectrograms |
| IMU-only | 1D CNN on accelerometer and gyroscope windows |
| Late fusion | Average the audio and IMU model logits |
| Cross-attention fusion | Bidirectional attention between temporal audio and IMU features |
| Random forest | Binary activity detection from 48 statistical IMU features |

Inputs use one-second windows with 0.5-second hops: six IMU channels at approximately 50 Hz and 64 × 96 audio mel spectrograms.

## Recorded results

| Model | Window detection F1 | Segment detection F1 | Known-segment classification macro F1 |
|---|---:|---:|---:|
| Random forest | 0.5092 | 0.6667 | — |
| Audio-only | 0.3858 | 0.3871 | 0.7771 |
| IMU-only | 0.5252 | **0.7333** | 0.7631 |
| Late fusion | 0.4827 | 0.4308 | 0.8614 |
| Cross-attention | **0.5699** | 0.5833 | **0.8853** |

These are saved notebook outputs, not a new training run. Known-segment classification uses ground-truth activity boundaries and excludes `Other` when selecting a segment label; **0.8853 macro F1 is not 88.53% continuous-stream accuracy**. Detection and classification use different evaluation populations. The notebook discusses the single-run results, smoothing differences, and other limitations.

## Run the notebook

### Google Colab

1. Open `Multimodal_Human_Activity_Recognition.ipynb` in Google Colab.
2. Select a GPU runtime, such as T4.
3. Run the cells in order. The setup cell mounts Drive and creates `MyDrive/multimodal_har`.
4. The data cell downloads the original archive and prepares `sensor_data/`. Existing data in that directory is reused.

Dataset access is required; raw recordings are not included. The original download's availability has not been verified during this cleanup.

### Local Jupyter

With Python 3.10–3.13:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab Multimodal_Human_Activity_Recognition.ipynb
```

Skip the Google Drive setup cell when running locally. Open the notebook from the repository directory. Training the audio and fusion models benefits from a GPU. MobileNetV2 initialization downloads pretrained weights on its first run.

### Data layout

```text
sensor_data/
├── 011/
│   ├── 011.csv       # Headerless six-channel IMU data
│   ├── 011.wav       # Audio at 22,050 Hz
│   └── 011ant.csv    # Tab-separated activity annotations
└── ...
```

Session directories nested under `train/` or `test/` are also supported. Annotations use `Start (ms)`, `End (ms)`, `Duration (ms)`, and `Label`.

| Split | Session IDs |
|---|---|
| Train | 011, 021, 031, 041, 051, 061, 071, 081 |
| Validation | 091, 101 |
| Test | 111, 121, 131, 141, 151 |

The validation split is declared in the notebook but is not used for checkpoint selection in its original training loops. Model definitions, training settings, evaluation logic, and saved result outputs are retained; setup paths and presentation were cleaned up, and the missing test-loader initialization was added.

## Repository contents

- [Multimodal_Human_Activity_Recognition.ipynb](Multimodal_Human_Activity_Recognition.ipynb) — experiment and recorded results.
- [requirements.txt](requirements.txt) — notebook dependencies.
- `.gitignore` — excludes local environments, sensor data, caches, and model artifacts.

All rights reserved. Dataset and third-party rights remain separate.
