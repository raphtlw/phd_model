"""
Export Wav2Vec2 phoneme recognition model to ONNX format for web deployment.

This script converts the PyTorch model to ONNX format with optimizations
for running in the browser using ONNX Runtime Web.
"""

import json
import os

import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from model.wav2vec2 import Wav2Vec2


def export_to_onnx(
    model_name="pklumpp/Wav2Vec2_CommonPhone",
    output_path="wav2vec2_phone.onnx",
    opset_version=18,
    sample_rate=16000,
    audio_duration=5.0,
):
    """
    Export Wav2Vec2 model to ONNX format.

    Args:
        model_name: Huggingface model identifier
        output_path: Path to save ONNX model
        opset_version: ONNX opset version (14+ recommended for web)
        sample_rate: Audio sample rate in Hz
        audio_duration: Duration of sample audio for tracing
    """
    print(f"Loading model from {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    wav2vec2 = Wav2Vec2.from_pretrained(model_name)
    wav2vec2.to(device)
    wav2vec2.eval()

    # Create dummy input with specified duration
    num_samples = int(sample_rate * audio_duration)
    dummy_input = torch.randn(1, num_samples, device=device)

    # Standardize input (as required by the model)
    mean = dummy_input.mean()
    std = dummy_input.std()
    dummy_input = (dummy_input - mean) / (std + 1e-9)

    print(f"Exporting to ONNX format (opset {opset_version})...")
    print(f"Input shape: {dummy_input.shape}")

    # Export to ONNX
    torch.onnx.export(
        wav2vec2,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["audio"],
        output_names=["logits", "encoder_features", "cnn_features"],
        dynamic_axes={
            "audio": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "time_steps"},
            "encoder_features": {0: "batch_size", 1: "time_steps"},
            "cnn_features": {0: "batch_size", 1: "time_steps"},
        },
    )

    print(f"Model exported to {output_path}")

    # Verify the exported model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    # Print model size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")

    return output_path


def quantize_model(input_path, output_path):
    """
    Quantize ONNX model to reduce size for web deployment.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save quantized model
    """
    import onnxruntime as ort

    print(f"\nQuantizing model from {input_path}...")

    try:
        # Use ONNX Runtime to optimize the graph (constant folding, etc.)
        # This converts dynamic ops to use constant initializers
        print("Running ONNX Runtime graph optimization...")
        optimized_path = input_path.replace(".onnx", "_optimized.onnx")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath = optimized_path

        # Create session to trigger optimization and save
        _ = ort.InferenceSession(input_path, sess_options, providers=["CPUExecutionProvider"])

        # Quantize the optimized model
        print("Quantizing optimized model...")
        quantize_dynamic(
            optimized_path,
            output_path,
            weight_type=QuantType.QUInt8,
        )

        # Clean up optimized file
        if os.path.exists(optimized_path):
            os.remove(optimized_path)

        # Print size comparison
        original_size = os.path.getsize(f"{input_path}.data") / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)

        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")

        return output_path
    except Exception as e:
        print(f"Quantization failed: {e}")
        print("Continuing with non-quantized model...")
        return input_path


def test_onnx_model(model_path, sample_rate=16000):
    """
    Test the exported ONNX model with random input.

    Args:
        model_path: Path to ONNX model
        sample_rate: Audio sample rate
    """
    import onnxruntime as ort

    print(f"\nTesting ONNX model: {model_path}")

    # Create session
    session = ort.InferenceSession(model_path)

    # Print input/output info
    print("\nModel Inputs:")
    for inp in session.get_inputs():
        print(f"  - {inp.name}: {inp.shape} ({inp.type})")

    print("\nModel Outputs:")
    for out in session.get_outputs():
        print(f"  - {out.name}: {out.shape} ({out.type})")

    # Create test input
    test_audio = np.random.randn(1, sample_rate * 2).astype(np.float32)

    # Standardize
    mean = test_audio.mean()
    std = test_audio.std()
    test_audio = (test_audio - mean) / (std + 1e-9)

    # Run inference
    print("\nRunning inference...")
    outputs = session.run(None, {"audio": test_audio})

    print(f"Logits shape: {outputs[0].shape}")
    print(f"Encoder features shape: {outputs[1].shape}")
    print(f"CNN features shape: {outputs[2].shape}")

    return outputs


def export_vocabulary(output_path="vocabulary.json"):
    """
    Export IPA vocabulary to JSON for use in web application.

    Args:
        output_path: Path to save vocabulary JSON
    """
    from phonetics.ipa import symbol_to_descriptor, to_symbol

    print(f"\nExporting vocabulary to {output_path}...")

    vocab = {}
    vocab_list = []

    try:
        # Try to get all symbols
        idx = 0
        while idx < 200:  # Safety limit
            try:
                symbol = to_symbol(idx)
                descriptor = symbol_to_descriptor(symbol) if symbol else "unknown"
                vocab[str(idx)] = {"symbol": symbol, "descriptor": descriptor}
                vocab_list.append(symbol)
                idx += 1
            except:
                break
    except Exception as e:
        print(f"Note: Extracted {len(vocab)} phonemes. Error: {e}")

    # Export both formats
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    with open("vocabulary_list.json", "w", encoding="utf-8") as f:
        json.dump(vocab_list, f, ensure_ascii=False)

    print(f"Exported {len(vocab)} phonemes to {output_path}")
    print(f"Exported vocabulary list to vocabulary_list.json")

    return vocab


def main():
    """Main export pipeline."""

    # Step 1: Export to ONNX
    onnx_path = export_to_onnx(
        model_name="pklumpp/Wav2Vec2_CommonPhone",
        output_path="wav2vec2_phone.onnx",
        opset_version=18,
        audio_duration=5.0,
    )

    # Step 2: Quantize for smaller size
    quantized_path = "wav2vec2_phone_quantized.onnx"
    actual_quantized_path = quantize_model(onnx_path, quantized_path)

    # Step 3: Test the models
    print("\n" + "=" * 50)
    print("Testing original ONNX model:")
    print("=" * 50)
    test_onnx_model(onnx_path)

    # Only test quantized model if it was successfully created
    if actual_quantized_path != onnx_path and os.path.exists(actual_quantized_path):
        print("\n" + "=" * 50)
        print("Testing quantized ONNX model:")
        print("=" * 50)
        test_onnx_model(actual_quantized_path)
    else:
        print("\n" + "=" * 50)
        print("Skipping quantized model test (quantization was not successful)")
        print("=" * 50)

    # Step 4: Export vocabulary
    export_vocabulary("vocabulary.json")

    print("\n" + "=" * 50)
    print("Export complete!")
    print("=" * 50)
    print("\nFiles created:")
    print(f"  - {onnx_path} (full precision)")
    print(f"  - {quantized_path} (quantized, recommended for web)")
    print(f"  - vocabulary.json (IPA phoneme mapping)")
    print(f"  - vocabulary_list.json (phoneme list)")
    print("\nNext steps:")
    print("  1. Copy these files to your web project")
    print("  2. Use the TypeScript wrapper to load and run inference")
    print("  3. Process audio from microphone or file uploads")


if __name__ == "__main__":
    main()
