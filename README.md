# Handwritten Digit Recogniser with a GUI

Draw a digit with the mouse and the app tells you what it thinks you drew. A small classifier trained on MNIST sits behind a Tkinter canvas, so the loop from freehand input to a prediction is visible end to end instead of hidden in a notebook cell.

## What it does

- A drawing canvas with a **Clear** button to wipe it and a **Save** button to keep the image
- Real-time prediction of the digit on the canvas
- A saved model, so it predicts without retraining

## Files

| File | What it does |
|---|---|
| `GUI_digit_predictor.py` | The Tkinter app: canvas, buttons, preprocessing and prediction |
| `Handwritten Digit Recog.ipynb` | Training and evaluation of the classifier |
| `digits/` | Sample digit images used while testing the preprocessing |

## Requirements

- Python 3.x
- `tkinter` (bundled with most Python installs; on Linux you may need `python3-tk`)
- NumPy, Matplotlib, scikit-learn

## Running it

```bash
pip install numpy matplotlib scikit-learn
python GUI_digit_predictor.py
```

## Notes

The hard part of this project is not the classifier, it is the preprocessing. MNIST digits are centred, size-normalised and anti-aliased in a particular way, and a mouse-drawn digit is none of those things, so what happens between the canvas and the model decides whether the prediction is any good.

The model is trained on the MNIST dataset, which is the standard benchmark for this task and also an easy one.
