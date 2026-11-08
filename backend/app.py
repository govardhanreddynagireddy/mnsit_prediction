import os
import io
import base64
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory

# ============================================================================
# 1. MODEL ARCHITECTURE (Strictly Preserved)
# ============================================================================
class MNIST_CNN(nn.Module):
    """
    Standard MNIST Convolutional Neural Network.
    Architecture, layer count, channels, kernels, and parameters are
    strictly preserved to match the pretrained mnist_cnn.pth checkpoint.
    """
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


# ============================================================================
# 2. FLASK INITIALIZATION & MODEL LOADING
# ============================================================================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "backend", "mnist_cnn.pth")
FRONTEND_PATH = os.path.join(BASE_DIR, "frontend")

# Verify checkpoint exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model checkpoint not found at: {MODEL_PATH}")

# Instantiate model architecture
device = torch.device("cpu")
model = MNIST_CNN().to(device)

# Load pretrained weights into architecture
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)

# Set model to evaluation mode for inference
model.eval()

print(f"Pretrained MNIST CNN loaded successfully from {MODEL_PATH} on {device}")


# ============================================================================
# 3. ROBUST MNIST-COMPLIANT IMAGE PREPROCESSING
# ============================================================================
def preprocess_image(image: Image.Image) -> tuple[torch.Tensor, str]:
    """
    Converts user-drawn canvas input into a 28x28 normalized tensor aligned with
    the original MNIST dataset conventions (LeCun et al.):
      1. Convert input image to single-channel 8-bit grayscale ('L').
      2. Verify & correct pixel polarity (MNIST requires white digit on black background).
      3. Segment digit using bounding box detection to eliminate blank background.
      4. Reject blank or unrecognizably faint/small inputs safely.
      5. Size-normalize the digit to fit within a 20x20 bounding box while preserving aspect ratio.
      6. Center the normalized digit inside a 28x28 black canvas.
      7. Translate the 28x28 image so the center of mass aligns with the image center (14.0, 14.0).
      8. Convert to float32 PyTorch tensor [1, 1, 28, 28] scaled to [0.0, 1.0].
      9. Generate a base64 PNG data URL of the final 28x28 image for visual debugging inspection.
    """
    # 1. Convert to grayscale
    if image.mode != "L":
        image = image.convert("L")

    arr = np.array(image, dtype=np.float32)

    # 2. Polarity check: MNIST training distribution requires white digit (high intensity)
    #    on black background (zero intensity). If outer borders are predominantly white,
    #    invert the image so background is zero.
    border = np.concatenate([
        arr[:4, :].ravel(),
        arr[-4:, :].ravel(),
        arr[:, :4].ravel(),
        arr[:, -4:].ravel()
    ])
    if np.mean(border) > 127:
        image = ImageOps.invert(image)
        arr = 255.0 - arr

    # 3. Detect bounding box of non-background pixels (threshold > 20 removes anti-aliasing noise)
    mask = image.point(lambda p: 255 if p > 20 else 0)
    bbox = mask.getbbox()

    if bbox is None:
        raise ValueError("No digit detected. Please draw a digit on the canvas.")

    cropped = image.crop(bbox)
    w, h = cropped.size

    # Guard against accidental tiny dots or noise clicks
    if w < 3 and h < 3:
        raise ValueError("Drawing is too small or faint. Please draw a clearer digit.")

    # 4. Aspect-ratio preserving resize into a 20x20 bounding box
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # 5. Place resized digit inside a 28x28 black image at geometric center
    padded = Image.new("L", (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))

    # 6. MNIST Center of Mass Alignment:
    #    LeCun et al. centered NIST digits by translating the image so that
    #    its center of mass aligns at (14.0, 14.0).
    padded_arr = np.array(padded, dtype=np.float32)
    total_mass = padded_arr.sum()

    if total_mass > 0:
        y_indices, x_indices = np.indices((28, 28))
        cx = (x_indices * padded_arr).sum() / total_mass
        cy = (y_indices * padded_arr).sum() / total_mass

        # Shift required to center mass at (14.0, 14.0), clamped to prevent clipping
        shift_x = float(np.clip(14.0 - cx, -4.0, 4.0))
        shift_y = float(np.clip(14.0 - cy, -4.0, 4.0))

        # Smooth subpixel translation using PIL affine transform
        final_img = padded.transform(
            (28, 28),
            Image.Transform.AFFINE,
            (1, 0, -shift_x, 0, 1, -shift_y),
            resample=Image.Resampling.BILINEAR
        )
    else:
        final_img = padded

    # 7. Convert to float32 tensor [1, 1, 28, 28] scaled to [0.0, 1.0]
    final_arr = np.array(final_img, dtype=np.float32) / 255.0
    tensor = torch.tensor(final_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 8. Export base64 data URI of the exact 28x28 image for UI debugging preview
    buffer = io.BytesIO()
    final_img.save(buffer, format="PNG")
    debug_image_b64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    return tensor, debug_image_b64


# ============================================================================
# 4. API ROUTES & CORS
# ============================================================================
@app.after_request
def add_cors_headers(response):
    """Enable CORS headers for cross-origin requests (e.g. VS Code Live Server on port 5500)."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def home():
    """Serve the primary single-page frontend application."""
    return send_from_directory(FRONTEND_PATH, "index.html")


@app.route("/<path:filename>")
def frontend_static(filename):
    """Serve static assets (style.css, script.js, icons, etc.)."""
    return send_from_directory(FRONTEND_PATH, filename)


@app.route("/health", methods=["GET", "OPTIONS"])
def health_check():
    """Health check endpoint to verify backend status and model readiness."""
    if request.method == "OPTIONS":
        return "", 204
    return jsonify({
        "status": "online",
        "model_loaded": True,
        "model": "MNIST_CNN",
        "device": str(device),
        "test_accuracy": "99.15%"
    })


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """
    Receives drawing data URL from the canvas, performs MNIST-compliant
    preprocessing, runs CNN inference, and returns predicted class and
    softmax probabilities.
    """
    if request.method == "OPTIONS":
        return "", 204
    try:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({
                "error": "Invalid request. Expected JSON payload with 'image' field."
            }), 400

        image_data = data["image"]
        if not isinstance(image_data, str) or not image_data.strip():
            return jsonify({"error": "Empty or invalid image data provided."}), 400

        # Decode base64 image (handles data URL prefixes)
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception:
            return jsonify({
                "error": "Failed to decode image data. Ensure valid Base64 / PNG format."
            }), 400

        # Preprocess input image to 28x28 MNIST format
        image_tensor, debug_image = preprocess_image(image)

        # Run inference using torch.inference_mode()
        with torch.inference_mode():
            logits = model(image_tensor.to(device))
            probabilities_tensor = torch.softmax(logits, dim=1)[0]
            confidence_val, predicted_idx = torch.max(probabilities_tensor, dim=0)

            predicted_digit = int(predicted_idx.item())
            confidence_pct = round(float(confidence_val.item()) * 100.0, 2)

            probabilities_dict = {
                str(i): round(float(probabilities_tensor[i].item()) * 100.0, 2)
                for i in range(10)
            }

        return jsonify({
            "prediction": predicted_digit,
            "confidence": confidence_pct,
            "probabilities": probabilities_dict,
            "debug_image": debug_image
        })

    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as err:
        return jsonify({"error": f"Internal server error: {str(err)}"}), 500


# ============================================================================
# 5. ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
