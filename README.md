# Handwritten Digit Recognition (PyTorch + Flask)

A handwritten digit recognition dashboard and inference engine powered by a Convolutional Neural Network (CNN) trained on the MNIST dataset (~99.15% test accuracy).

---

## 1. Project Architecture & Components

```text
hand_digit_recognization/
│
├── backend/
│   ├── app.py             # Flask API server, MNIST preprocessing, PyTorch inference
│   └── mnist_cnn.pth      # Original pretrained CNN weights (Untouched, SHA256 verified)
│
├── frontend/
│   ├── index.html         # Modern AI/ML dashboard UI
│   ├── style.css          # Dark-theme styles, responsive layout, animations
│   └── script.js          # Canvas drawing engine, smoothing, API integration, history
│
├── test_app.py            # Automated validation & test suite
├── pyproject.toml         # uv project configuration and dependencies
└── uv.lock                # Dependency lockfile
```

---

## 2. CNN Model Architecture (Strictly Preserved)

The original weights and biases from `mnist_cnn.pth` are loaded strictly via `model.load_state_dict()` without alteration:

```python
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
```

- **Parameters:** 421,642 float32 parameters
- **Input shape:** `[1, 1, 28, 28]`
- **Scaling:** `[0.0, 1.0]` (ToTensor scaling only, no extraneous standardization)
- **Model Hash (SHA-256):** `0dc588ed26836cca21854762f22ec5e49d96009d3ea3547f81a3cd4c3f49c726`

---

## 3. Preprocessing Pipeline (LeCun / MNIST Alignment)

The core accuracy failure of the original application was a catastrophic inversion bug in image preprocessing that converted black background to white and fed black-on-white inverted frames into a model trained on white-on-black digits.

The new preprocessing pipeline adheres strictly to MNIST conventions:
1. **Grayscale conversion:** Ensures single-channel input (`PIL.Image.convert("L")`).
2. **Polarity validation:** Verifies outer boundary pixel statistics; guarantees white digit on black background (`arr = 255 - arr` if drawn on white).
3. **Digit segmentation:** Detects the active bounding box (`mask.getbbox()`) to eliminate unconstrained empty canvas space.
4. **Safety & noise gating:** Safely catches empty canvases and rejects accidental sub-pixel clicks (< 3px).
5. **Aspect ratio preservation:** Rescales the digit's longest dimension to 20 pixels (`scale = 20.0 / max(w, h)`).
6. **Centering by Center of Mass (CoM):**
   - The original MNIST dataset centered NIST digits by translating the 28×28 frame to place the digit's center of mass at coordinate `(14.0, 14.0)`.
   - The application computes $c_x = \frac{\sum x \cdot I(x, y)}{\sum I(x, y)}$ and $c_y = \frac{\sum y \cdot I(x, y)}{\sum I(x, y)}$, then performs a subpixel affine translation (`PIL.Image.Transform.AFFINE`).
7. **Tensor conversion:** Returns `torch.float32` tensor with shape `[1, 1, 28, 28]` normalized to $[0.0, 1.0]$.
8. **Visual inspector base64 output:** Returns the exact 28×28 image as a PNG data URL for visual inspection in the UI.

---

## 4. How to Run the Application

Start the Flask server using `uv`:

```bash
uv run python backend/app.py
```

The application will start on `http://127.0.0.1:5000`. Open this URL in any modern web browser.

---

## 5. How to Test the Application

Run the automated test suite covering model integrity, edge cases, and all digits (0–9):

```bash
uv run python test_app.py
```
