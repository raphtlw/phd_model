import wav from "wav-decoder";

/**
 * Preprocesses audio buffer for Wav2Vec2.
 * 1. Decodes WAV
 * 2. Checks Sample Rate (Must be 16000Hz)
 * 3. Normalizes (mean=0, std=1)
 *
 * @param {Buffer} audioBuffer - Raw WAV file buffer
 * @returns {Promise<Float32Array>} - Normalized audio data ready for ONNX
 */
export async function processAudio(audioBuffer) {
  const decoded = await wav.decode(audioBuffer);

  // Wav2Vec2 is strictly trained on 16kHz
  if (decoded.sampleRate !== 16000) {
    throw new Error(
      `Invalid Sample Rate: ${decoded.sampleRate}Hz. Model requires 16000Hz.`,
    );
  }

  // Mix to mono if stereo
  let audioData = decoded.channelData[0];
  if (decoded.channelData.length > 1) {
    // Simple average for mono mixdown
    const left = decoded.channelData[0];
    const right = decoded.channelData[1];
    audioData = new Float32Array(left.length);
    for (let i = 0; i < left.length; i++) {
      audioData[i] = (left[i] + right[i]) / 2;
    }
  }

  // Normalize: (x - mean) / (std + 1e-9)
  let sum = 0;
  for (let i = 0; i < audioData.length; i++) {
    sum += audioData[i];
  }
  const mean = sum / audioData.length;

  let varianceSum = 0;
  for (let i = 0; i < audioData.length; i++) {
    varianceSum += Math.pow(audioData[i] - mean, 2);
  }
  const std = Math.sqrt(varianceSum / audioData.length);

  const normalized = new Float32Array(audioData.length);
  for (let i = 0; i < audioData.length; i++) {
    normalized[i] = (audioData[i] - mean) / (std + 1e-9);
  }

  return normalized;
}
