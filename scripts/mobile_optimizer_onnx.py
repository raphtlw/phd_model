import argparse
import logging
import os
import sys
from pathlib import Path

import onnx
from onnxoptimizer import optimize
from onnxruntime.quantization import QuantType, quantize_dynamic

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MobileOptimizer")


def optimize_graph(model_path: str, output_path: str):
    """
    Runs ONNX Optimizer passes to simplify the graph structure.
    This fuses layers (like Conv+BN) and removes unused nodes,
    reducing computational overhead before quantization.
    """
    logger.info(f"Step 1: Optimizing graph structure for {model_path}...")

    # Define a robust set of optimization passes suitable for mobile inference.
    # These passes simplify the math without changing the result.
    passes = [
        "eliminate_deadend",
        "eliminate_identity",
        "eliminate_nop_dropout",
        "eliminate_nop_monotone_argmax",
        "eliminate_nop_pad",
        "eliminate_unused_initializer",
        "extract_constant_to_initializer",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_consecutive_concats",
        "fuse_consecutive_log_softmax",
        "fuse_consecutive_reduce_unsqueeze",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_transposes",
        "fuse_matmul_add_bias_into_gemm",
        "fuse_pad_into_conv",
        "fuse_transpose_into_gemm",
    ]

    try:
        model = onnx.load(model_path)
        optimized_model = optimize(model, passes)
        onnx.save(optimized_model, output_path)
        logger.info(f"Graph optimization complete. Saved temp file to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Graph optimization failed: {e}")
        return False


def quantize_model(model_path: str, output_path: str):
    """
    Applies Dynamic Quantization (Int4).
    This reduces model size by ~4x and speeds up inference on mobile CPUs
    by changing weight precision from Float32 to UInt8.
    """
    logger.info(f"Step 2: Quantizing model to Int4 (Dynamic)...")

    try:
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            # QUInt8 is generally safer for ARM/Mobile (Android/iOS) specific kernels than QInt4
            weight_type=QuantType.QUInt4,
        )
        logger.info(f"Quantization complete. Saved to {output_path}")

        # Log size comparison
        orig_size = os.path.getsize(model_path) / (1024 * 1024)
        new_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Size reduction: {orig_size:.2f} MB -> {new_size:.2f} MB")

        return True
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Optimize and Quantize ONNX model for Mobile Deployment",
        epilog="Example: python mobile_optimizer.py my_model.onnx --output my_model_mobile.onnx",
    )
    parser.add_argument(
        "input_model", type=str, help="Path to input .onnx model (Float32)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path for output model (Optional)"
    )

    args = parser.parse_args()

    input_path = Path(args.input_model)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Define paths
    base_name = input_path.stem
    if args.output:
        final_output_path = Path(args.output)
    else:
        final_output_path = input_path.parent / f"{base_name}.mobile.onnx"

    temp_opt_path = input_path.parent / f"{base_name}.opt.temp.onnx"

    print("-" * 50)
    print(f"Starting Mobile Optimization Pipeline for: {input_path.name}")
    print("-" * 50)

    # Step 1: Optimize Graph
    if optimize_graph(str(input_path), str(temp_opt_path)):
        # Step 2: Quantize
        if quantize_model(str(temp_opt_path), str(final_output_path)):
            print("-" * 50)
            logger.info(
                "SUCCESS! You can now deploy the .mobile.onnx file to your device."
            )

            # Cleanup temp file
            if temp_opt_path.exists():
                os.remove(temp_opt_path)
                logger.info("Temporary files cleaned up.")
        else:
            logger.error("Pipeline failed at quantization step.")
            sys.exit(1)
    else:
        logger.error("Pipeline failed at optimization step.")
        sys.exit(1)


if __name__ == "__main__":
    main()
