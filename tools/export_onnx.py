import os
import sys

# 1. Add the project root to sys.path explicitly
# This allows us to import 'model' regardless of where this script is run from
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn

# Now we can safely import from the project structure
try:
    from model.wav2vec2 import Wav2Vec2
except ImportError as e:
    print(f"Error importing model: {e}")
    print(f"Computed project root: {project_root}")
    sys.exit(1)

from onnxruntime.quantization import QuantType, quantize_dynamic


class OnnxWrapper(nn.Module):
    """
    Wrapper to ensure the model only outputs logits during ONNX export.
    This simplifies the web inference graph.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # We only care about the logits (y) for transcription
        y, _, _ = self.model(x)
        return y


def export_model():
    print(f"Loading model from {project_root}...")

    # Load the pretrained model
    try:
        model = Wav2Vec2.from_pretrained("pklumpp/Wav2Vec2_CommonPhone")
        model.eval()
    except Exception as e:
        print(f"Failed to load model from HuggingFace: {e}")
        return

    # Wrap it
    onnx_model = OnnxWrapper(model)

    # Create dummy input (Batch Size 1, 16000 samples = 1 second)
    dummy_input = torch.randn(1, 16000, requires_grad=False)

    # Define output paths (saving to project root for easy access)
    onnx_path = os.path.join(project_root, "wav2vec2_commonphone.onnx")
    quantized_path = os.path.join(project_root, "wav2vec2_commonphone.quant.onnx")

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        onnx_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "time"},
            "logits": {0: "batch_size", 1: "time"},
        },
    )
    print("Export complete.")

    print(f"Quantizing model to {quantized_path} (int8)...")
    quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QUInt8)
    print("Quantization complete.")
    print("-" * 30)
    print(
        f"SUCCESS: You can now use '{os.path.basename(quantized_path)}' in the web app."
    )


if __name__ == "__main__":
    export_model()
