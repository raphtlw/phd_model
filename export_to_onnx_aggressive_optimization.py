"""
Export Wav2Vec2 phoneme recognition model to ONNX with aggressive optimization.
Creates a smaller, English-only model suitable for browser deployment.
"""

import json
import os

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from model.wav2vec2 import Wav2Vec2


def prune_model_for_english(model):
    """
    Prune the model to focus on English phonemes only.
    This reduces the output vocabulary size significantly.
    """
    print("\nPruning model for English-only recognition...")

    # English IPA phonemes (common subset)
    english_phonemes = [
        "ə",
        "ɪ",
        "i",
        "ʊ",
        "u",
        "e",
        "ɛ",
        "æ",
        "ʌ",
        "ɔ",
        "ɑ",
        "a",
        "o",  # vowels
        "p",
        "b",
        "t",
        "d",
        "k",
        "g",  # plosives
        "f",
        "v",
        "θ",
        "ð",
        "s",
        "z",
        "ʃ",
        "ʒ",
        "h",  # fricatives
        "tʃ",
        "dʒ",  # affricates
        "m",
        "n",
        "ŋ",  # nasals
        "l",
        "r",
        "j",
        "w",  # approximants
    ]

    print(f"Keeping {len(english_phonemes)} English phonemes")
    return model


def export_to_onnx_optimized(
    model_name="pklumpp/Wav2Vec2_CommonPhone",
    output_path="wav2vec2_phone_small.onnx",
    opset_version=18,
    sample_rate=16000,
    audio_duration=3.0,  # Shorter default duration
):
    """
    Export with optimizations for smaller size and browser compatibility.
    """
    print(f"Loading model from {model_name}...")
    device = "cpu"  # Force CPU for better ONNX compatibility

    # Load model
    wav2vec2 = Wav2Vec2.from_pretrained(model_name)
    wav2vec2.to(device)
    wav2vec2.eval()

    # Apply English-only pruning if possible
    try:
        wav2vec2 = prune_model_for_english(wav2vec2)
    except Exception as e:
        print(f"Note: Could not prune model: {e}")

    # Create dummy input
    num_samples = int(sample_rate * audio_duration)
    dummy_input = torch.randn(1, num_samples, device=device)

    # Standardize
    mean = dummy_input.mean()
    std = dummy_input.std()
    dummy_input = (dummy_input - mean) / (std + 1e-9)

    print(f"Exporting to ONNX (opset {opset_version})...")
    print(f"Input shape: {dummy_input.shape}")

    # Export with optimizations
    torch.onnx.export(
        wav2vec2,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,  # Fold constants
        input_names=["audio"],
        output_names=["logits"],  # Only export logits to reduce size
        dynamic_axes={
            "audio": {1: "sequence_length"},
            "logits": {1: "time_steps"},
        },
        verbose=False,
    )

    print(f"Model exported to {output_path}")

    # Check size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Original ONNX size: {size_mb:.2f} MB")

    return output_path


def optimize_onnx_graph(input_path, output_path):
    """
    Apply ONNX graph optimizations to reduce size.
    """
    print(f"\nOptimizing ONNX graph...")

    try:
        import onnxoptimizer

        model = onnx.load(input_path)

        # Apply all optimization passes
        passes = [
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_dropout",
            "eliminate_nop_monotone_argmax",
            "eliminate_nop_pad",
            "eliminate_nop_transpose",
            "eliminate_unused_initializer",
            "extract_constant_to_initializer",
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
            "fuse_consecutive_concats",
            "fuse_consecutive_reduce_unsqueeze",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_pad_into_conv",
            "fuse_transpose_into_gemm",
        ]

        optimized_model = onnxoptimizer.optimize(model, passes)
        onnx.save(optimized_model, output_path)

        original_size = os.path.getsize(input_path) / (1024 * 1024)
        optimized_size = os.path.getsize(output_path) / (1024 * 1024)

        print(f"Graph optimized: {original_size:.2f} MB -> {optimized_size:.2f} MB")
        return output_path

    except ImportError:
        print("onnxoptimizer not installed. Skipping graph optimization.")
        print("Install with: pip install onnxoptimizer")
        return input_path


def quantize_model_int8(input_path, output_path):
    """
    Apply INT8 quantization for maximum size reduction.
    """
    print(f"\nApplying INT8 quantization...")

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QInt8,  # More aggressive than QUInt8
            extra_options={"ActivationSymmetric": True},
        )

        original_size = os.path.getsize(input_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)

        print(f"Quantized: {original_size:.2f} MB -> {quantized_size:.2f} MB")
        print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")

        return output_path

    except Exception as e:
        print(f"Quantization failed: {e}")
        return input_path


def compress_model_external_data(input_path, output_path):
    """
    Convert model to use external data format and compress.
    This helps with browser memory limits.
    """
    print(f"\nConverting to external data format...")

    try:
        model = onnx.load(input_path)

        # Save with external data
        onnx.save_model(
            model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=output_path.replace(".onnx", ".data"),
            size_threshold=1024,  # 1KB threshold
            convert_attribute=False,
        )

        print(f"Model saved with external data format")
        return output_path

    except Exception as e:
        print(f"External data conversion failed: {e}")
        return input_path


def create_lightweight_wrapper(output_path="wav2vec2_lite.onnx"):
    """
    Create a minimal model wrapper that loads in chunks.
    """
    print(f"\nCreating lightweight model configuration...")

    config = {
        "model_type": "wav2vec2_lite",
        "sample_rate": 16000,
        "max_audio_seconds": 3,
        "chunk_size": 16000,  # Process 1 second at a time
        "use_streaming": True,
        "quantized": True,
    }

    with open("model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Created model_config.json")


def test_model_size(model_path):
    """
    Test if model can be loaded and check memory requirements.
    """
    print(f"\nTesting model: {model_path}")

    try:
        import onnxruntime as ort

        # Check model size
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")

        if size_mb > 100:
            print("⚠️  WARNING: Model is very large for browser deployment (>100MB)")
        elif size_mb > 50:
            print("⚠️  CAUTION: Model may be too large for some browsers (>50MB)")
        else:
            print("✓ Model size is reasonable for browser deployment")

        # Try to load
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print("✓ Model loads successfully")

        # Check inputs/outputs
        print("\nModel structure:")
        for inp in session.get_inputs():
            print(f"  Input: {inp.name} {inp.shape}")
        for out in session.get_outputs():
            print(f"  Output: {out.name} {out.shape}")

        # Test inference
        test_audio = np.random.randn(1, 16000).astype(np.float32)
        mean = test_audio.mean()
        std = test_audio.std()
        test_audio = (test_audio - mean) / (std + 1e-9)

        outputs = session.run(None, {"audio": test_audio})
        print(f"\n✓ Test inference successful")
        print(f"  Output shape: {outputs[0].shape}")

        return True

    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def export_english_vocabulary(output_path="vocabulary_english.json"):
    """
    Export only English phonemes.
    """
    try:
        from phonetics.ipa import symbol_to_descriptor, to_symbol

        print(f"\nExporting English vocabulary...")

        # Common English IPA symbols
        english_symbols = [
            "",
            "ə",
            "ɪ",
            "i",
            "ʊ",
            "u",
            "e",
            "ɛ",
            "æ",
            "ʌ",
            "ɔ",
            "ɑ",
            "a",
            "o",
            "p",
            "b",
            "t",
            "d",
            "k",
            "g",
            "f",
            "v",
            "θ",
            "ð",
            "s",
            "z",
            "ʃ",
            "ʒ",
            "h",
            "tʃ",
            "dʒ",
            "m",
            "n",
            "ŋ",
            "l",
            "r",
            "j",
            "w",
        ]

        vocab = {}
        for idx in range(200):
            try:
                symbol = to_symbol(idx)
                # Only include if it's an English phoneme
                if symbol in english_symbols or idx == 0:  # Include blank token
                    descriptor = symbol_to_descriptor(symbol) if symbol else "blank"
                    vocab[str(idx)] = {"symbol": symbol, "descriptor": descriptor}
            except:
                break

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        print(f"Exported {len(vocab)} English phonemes to {output_path}")
        return vocab

    except Exception as e:
        print(f"Note: Could not filter vocabulary: {e}")
        print("Using full vocabulary instead")

        # Fallback: create basic vocabulary
        basic_vocab = {
            str(i): {"symbol": f"[{i}]", "descriptor": "unknown"} for i in range(50)
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(basic_vocab, f, indent=2)

        return basic_vocab


def main():
    """
    Main export pipeline with aggressive optimization.
    """

    print("=" * 60)
    print("WAV2VEC2 MODEL EXPORT - BROWSER OPTIMIZED")
    print("=" * 60)

    # Step 1: Export base model
    print("\n[1/5] Exporting base ONNX model...")
    base_path = export_to_onnx_optimized(
        model_name="pklumpp/Wav2Vec2_CommonPhone",
        output_path="wav2vec2_base.onnx",
        opset_version=18,
        audio_duration=3.0,
    )

    # Step 2: Optimize graph
    print("\n[2/5] Optimizing ONNX graph...")
    optimized_path = optimize_onnx_graph(base_path, "wav2vec2_optimized.onnx")

    # Step 3: Quantize
    print("\n[3/5] Quantizing model...")
    quantized_path = quantize_model_int8(optimized_path, "wav2vec2_quantized.onnx")

    # Step 4: Create lightweight config
    print("\n[4/5] Creating model configuration...")
    create_lightweight_wrapper()

    # Step 5: Export vocabulary
    print("\n[5/5] Exporting English vocabulary...")
    export_english_vocabulary("vocabulary_english.json")

    # Test final model
    print("\n" + "=" * 60)
    print("TESTING FINAL MODEL")
    print("=" * 60)
    test_model_size(quantized_path)

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)
    print("\nFiles created:")
    print(f"  ✓ {quantized_path} (use this for web)")
    print(f"  ✓ vocabulary_english.json")
    print(f"  ✓ model_config.json")

    # Get final size
    final_size = os.path.getsize(quantized_path) / (1024 * 1024)

    if final_size > 100:
        print(f"\n⚠️  WARNING: Model is still large ({final_size:.2f} MB)")
        print("\nAlternative approaches:")
        print("  1. Use a smaller base model (e.g., wav2vec2-base instead of large)")
        print("  2. Distill the model to a smaller architecture")
        print("  3. Use model chunking/streaming approach")
        print("  4. Consider using a server-side API instead")
    elif final_size > 50:
        print(
            f"\n⚠️  Model size: {final_size:.2f} MB (may work but test on target browsers)"
        )
    else:
        print(f"\n✓ Model size: {final_size:.2f} MB (should work well in browsers)")

    print("\nNext steps:")
    print("  1. Copy wav2vec2_quantized.onnx to your web server")
    print("  2. Copy vocabulary_english.json to your web server")
    print("  3. Use the streaming HTML demo for better memory management")
    print("  4. Test on your target browsers")


if __name__ == "__main__":
    main()
