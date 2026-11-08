"""
Comprehensive Validation & Test Suite for Handwritten Digit Recognition
Validates model preservation, preprocessing accuracy, API routes, and edge cases.
"""

import io
import base64
import hashlib
import numpy as np
from PIL import Image, ImageDraw
import torch

from backend.app import app, model, preprocess_image, MNIST_CNN, MODEL_PATH

EXPECTED_SHA256 = "0dc588ed26836cca21854762f22ec5e49d96009d3ea3547f81a3cd4c3f49c726"

def test_model_preservation():
    print("\n--- 1. Testing Model Preservation Rules ---")
    with open(MODEL_PATH, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    assert actual_hash == EXPECTED_SHA256, f"Model hash changed! Expected {EXPECTED_SHA256}, got {actual_hash}"
    print(f"[PASS] Model file mnist_cnn.pth SHA256 verified: {actual_hash}")

    # Verify architecture
    assert isinstance(model, MNIST_CNN), "Model is not an instance of MNIST_CNN"
    assert hasattr(model, "conv1") and model.conv1.in_channels == 1 and model.conv1.out_channels == 32
    assert hasattr(model, "conv2") and model.conv2.in_channels == 32 and model.conv2.out_channels == 64
    assert hasattr(model, "fc1") and model.fc1.in_features == 64 * 7 * 7 and model.fc1.out_features == 128
    assert hasattr(model, "fc2") and model.fc2.in_features == 128 and model.fc2.out_features == 10
    print("[PASS] CNN layer specifications, channels, and parameter shapes match exactly.")

    # Verify device and eval mode
    assert not model.training, "Model should be in eval() mode"
    print("[PASS] Model is in eval() mode.")


def test_preprocessing_edge_cases():
    print("\n--- 2. Testing Preprocessing & Edge Cases ---")

    # Empty canvas
    blank_canvas = Image.new("RGB", (280, 280), (0, 0, 0))
    try:
        preprocess_image(blank_canvas)
        assert False, "Should have raised ValueError on blank canvas"
    except ValueError as e:
        print(f"[PASS] Blank canvas correctly rejected: {e}")

    # Single tiny dot
    dot_canvas = Image.new("RGB", (280, 280), (0, 0, 0))
    dot_canvas.putpixel((140, 140), (255, 255, 255))
    try:
        preprocess_image(dot_canvas)
        assert False, "Should have raised ValueError on tiny dot"
    except ValueError as e:
        print(f"[PASS] Tiny dot correctly rejected: {e}")

    # Polarity invariance (white canvas with black digit)
    white_bg_canvas = Image.new("RGB", (280, 280), (255, 255, 255))
    draw = ImageDraw.Draw(white_bg_canvas)
    draw.line([(140, 50), (140, 230)], fill=(0, 0, 0), width=22)
    tensor, dbg = preprocess_image(white_bg_canvas)
    assert tensor.shape == (1, 1, 28, 28), f"Expected tensor shape (1, 1, 28, 28), got {tensor.shape}"
    assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"
    assert 0.0 <= tensor.min().item() and tensor.max().item() <= 1.0, "Tensor not in [0, 1] range"
    print("[PASS] White-background polarity inversion handled correctly, tensor range [0.0, 1.0].")


def test_api_endpoints():
    print("\n--- 3. Testing Flask API Endpoints & CORS ---")
    client = app.test_client()

    # GET /
    res = client.get("/")
    assert res.status_code == 200, f"GET / returned {res.status_code}"
    assert b"Handwritten Digit Recognition" in res.data
    print("[PASS] GET / serves frontend index.html")

    # GET /health with simulated Live Server Origin
    live_server_origin = "http://127.0.0.1:5500"
    res = client.get("/health", headers={"Origin": live_server_origin})
    assert res.status_code == 200
    assert res.headers.get("Access-Control-Allow-Origin") == "*"
    data = res.get_json()
    assert data["status"] == "online" and data["model_loaded"] is True
    print(f"[PASS] GET /health with CORS verified: {data}")

    # OPTIONS preflight /predict with simulated Live Server Origin
    res_opt = client.options("/predict", headers={
        "Origin": live_server_origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type"
    })
    assert res_opt.status_code == 204
    assert res_opt.headers.get("Access-Control-Allow-Origin") == "*"
    print("[PASS] OPTIONS /predict preflight request returns 204 with CORS headers.")

    # POST /predict with missing body
    res = client.post("/predict", json={})
    assert res.status_code == 400
    assert res.headers.get("Access-Control-Allow-Origin") == "*"
    print("[PASS] POST /predict with empty payload rejected with 400.")

    # POST /predict with invalid data
    res = client.post("/predict", json={"image": "not_an_image"})
    assert res.status_code == 400
    print("[PASS] POST /predict with invalid base64 rejected with 400.")


def test_end_to_end_digits():
    print("\n--- 4. Testing End-to-End Predictions on Digits 0 to 9 ---")
    client = app.test_client()

    def make_canvas(draw_func):
        c = Image.new("RGB", (280, 280), (0, 0, 0))
        d = ImageDraw.Draw(c)
        draw_func(d)
        buf = io.BytesIO()
        c.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    test_drawings = {
        0: lambda d: d.ellipse([80, 50, 200, 230], outline=(255, 255, 255), width=22),
        1: lambda d: d.line([(140, 50), (140, 230)], fill=(255, 255, 255), width=22),
        2: lambda d: d.line([(85, 95), (140, 55), (195, 95), (85, 230), (195, 230)], fill=(255, 255, 255), width=22, joint="curve"),
        3: lambda d: d.line([(90, 65), (190, 65), (140, 135), (190, 205), (90, 205)], fill=(255, 255, 255), width=22, joint="curve"),
        4: lambda d: [
            d.line([(175, 50), (175, 230)], fill=(255, 255, 255), width=22),
            d.line([(175, 50), (75, 170), (205, 170)], fill=(255, 255, 255), width=22)
        ],
        5: lambda d: d.line([(190, 65), (95, 65), (90, 135), (190, 145), (190, 215), (90, 215)], fill=(255, 255, 255), width=22, joint="curve"),
        6: lambda d: [
            d.line([(175, 65), (105, 155)], fill=(255, 255, 255), width=22),
            d.ellipse([90, 120, 190, 225], outline=(255, 255, 255), width=22)
        ],
        7: lambda d: d.line([(85, 65), (195, 65), (115, 225)], fill=(255, 255, 255), width=22),
        8: lambda d: [
            d.ellipse([100, 55, 180, 135], outline=(255, 255, 255), width=20),
            d.ellipse([90, 125, 190, 225], outline=(255, 255, 255), width=22)
        ],
        9: lambda d: [
            d.ellipse([90, 55, 190, 155], outline=(255, 255, 255), width=22),
            d.line([(180, 75), (175, 225), (120, 225)], fill=(255, 255, 255), width=22, joint="curve")
        ]
    }

    all_passed = True
    for expected_digit, draw_fn in test_drawings.items():
        payload = {"image": make_canvas(draw_fn)}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200, f"Prediction for {expected_digit} failed with {res.status_code}"
        body = res.get_json()

        pred = body["prediction"]
        conf = body["confidence"]
        probs = body["probabilities"]
        dbg = body["debug_image"]

        # Probability assertions
        assert len(probs) == 10, f"Expected 10 class probabilities, got {len(probs)}"
        prob_sum = sum(probs.values())
        assert 99.0 <= prob_sum <= 101.0, f"Probabilities should sum to ~100%, got {prob_sum:.2f}%"
        assert dbg.startswith("data:image/png;base64,"), "Debug image format incorrect"

        # Check prediction correctness
        is_correct = (pred == expected_digit)
        if not is_correct:
            all_passed = False
        print(f"Digit {expected_digit}: Predicted {pred} (Conf: {conf:.2f}%) - Sum: {prob_sum:.2f}% - {'CORRECT' if is_correct else 'MISMATCH'}")

    assert all_passed, "Some digit predictions did not match expected"
    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")


if __name__ == "__main__":
    test_model_preservation()
    test_preprocessing_edge_cases()
    test_api_endpoints()
    test_end_to_end_digits()
