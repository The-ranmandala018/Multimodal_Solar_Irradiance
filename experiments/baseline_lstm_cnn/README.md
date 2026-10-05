# Baseline CNN + LSTM Experiment

This folder stores outputs from the baseline multimodal solar irradiance training run.

## Architecture

- Folsom visual input: 10 x 180 x 720
- CNN visual encoder: 256-D
- Weather history: 40 x 7
- LSTM temporal encoder: 128-D
- Fusion: 384-D
- Regression head: 1 future k-index value
- Forecast horizon: 10 minutes

## Runtime outputs

Training creates these folders automatically:

```text
baseline_lstm_cnn/
├── checkpoints/
│   └── best_model.pt
├── logs/
│   └── training_history.csv
└── plots/
    └── training_validation_loss.png
```

The generated checkpoint, logs, and plots are experiment artifacts and should normally remain outside the source/calibration code.

The original Folsom calibration source is not modified by this experiment.
