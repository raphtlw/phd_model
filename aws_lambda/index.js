import onnx from "onnxruntime-node";
import { processAudio } from "./lib/audio_processor.js";
import { INDEX_TO_SYMBOL } from "./lib/ipa_tokens.js";
import fs from "node:fs";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";

// Global model session (caching context between warm Lambda invocations)
let session = null;

const MODEL_URL =
  "https://qa.st-visuals.com/assets/models/accent-recognition-2026/wav2vec2_commonphone.quant.onnx";
const LOCAL_MODEL_PATH = "/tmp/wav2vec2_commonphone.quant.onnx";

/**
 * Downloads the model safely.
 * Uses a temporary file to ensure we don't end up with a partial file
 * if the Lambda times out during download.
 */
async function ensureModelExists() {
  if (fs.existsSync(LOCAL_MODEL_PATH)) {
    console.log("Model found in /tmp, skipping download.");
    return;
  }

  console.log(`Downloading model from ${MODEL_URL}...`);

  const tempPath = `${LOCAL_MODEL_PATH}.download`;

  try {
    const response = await fetch(MODEL_URL);

    if (!response.ok) {
      throw new Error(
        `Failed to download model: ${response.status} ${response.statusText}`,
      );
    }

    // Download to a temporary filename first
    const fileStream = fs.createWriteStream(tempPath);
    await pipeline(Readable.fromWeb(response.body), fileStream);

    // Rename to final path only after successful completion (Atomic operation)
    fs.renameSync(tempPath, LOCAL_MODEL_PATH);
    console.log("Download complete.");
  } catch (error) {
    // Clean up temp file if download fails
    if (fs.existsSync(tempPath)) {
      fs.unlinkSync(tempPath);
    }
    throw error;
  }
}

/**
 * CTC Greedy Decoder
 */
function decodeCTC(logits, dims) {
  const [batchSize, timeSteps, vocabSize] = dims;
  const rawData = logits; // Float32Array

  let tokens = [];
  let prevIndex = -1;

  for (let t = 0; t < timeSteps; t++) {
    let maxVal = -Infinity;
    let maxIdx = 0;
    const offset = t * vocabSize;

    for (let v = 0; v < vocabSize; v++) {
      const val = rawData[offset + v];
      if (val > maxVal) {
        maxVal = val;
        maxIdx = v;
      }
    }

    if (maxIdx !== prevIndex) {
      if (maxIdx !== 0) {
        // 0 is blank
        tokens.push(maxIdx);
      }
      prevIndex = maxIdx;
    }
  }

  return tokens.map((idx) => INDEX_TO_SYMBOL[idx] || "").join(" ");
}

export const handler = async (event) => {
  try {
    // 1. Initialize Model (Download + Load)
    if (!session) {
      try {
        await ensureModelExists();
        console.log("Loading ONNX session...");
        session = await onnx.InferenceSession.create(LOCAL_MODEL_PATH);
        console.log("Model loaded successfully.");
      } catch (loadError) {
        console.error("Model initialization failed.", loadError);

        // CRITICAL: If loading fails (e.g. protobuf error), delete the file
        // so the next invocation downloads a fresh copy.
        if (fs.existsSync(LOCAL_MODEL_PATH)) {
          console.warn("Deleting potentially corrupt model file.");
          fs.unlinkSync(LOCAL_MODEL_PATH);
        }

        throw loadError;
      }
    }

    // 2. Handle Input
    let audioBuffer;
    if (event.body) {
      const isBase64 = event.isBase64Encoded;
      audioBuffer = Buffer.from(event.body, isBase64 ? "base64" : "utf8");
    } else if (event.audio_base64) {
      audioBuffer = Buffer.from(event.audio_base64, "base64");
    } else {
      throw new Error("No audio found in event body");
    }

    // 3. Preprocess
    console.log("Processing audio...");
    const float32Audio = await processAudio(audioBuffer);

    // 4. Inference
    console.log("Running inference...");
    const tensor = new onnx.Tensor("float32", float32Audio, [
      1,
      float32Audio.length,
    ]);

    // 'input' must match the input name in your ONNX model
    const feeds = { input: tensor };
    const results = await session.run(feeds);
    const logits = results.logits;

    // 5. Decode
    const ipaString = decodeCTC(logits.data, logits.dims);

    return {
      statusCode: 200,
      body: JSON.stringify({
        phonemes: ipaString,
        length_samples: float32Audio.length,
      }),
    };
  } catch (error) {
    console.error("Inference Error:", error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message }),
    };
  }
};
