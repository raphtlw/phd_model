# Multilingual IPA Phone Recognition Model

## What is this?
This is the base source code to utilize a _Wav2Vec2_-based phone recognition / feature extraction model. It was created in the scope of the PhD thesis [Phonetic Transfer Learning from Healthy References for the Analysis of Pathological Speech](https://open.fau.de/items/d0c6b800-e217-4049-ab1f-a746fc9b3966) by [Philipp Klumpp](https://scholar.google.com/citations?user=IWvgno4AAAAJ) to analyze pathological speech signals.. The model parameters are deployed via huggingface. Check out the model card [here](https://huggingface.co/pklumpp/Wav2Vec2_CommonPhone).

## Who wants to use this model?
- You want to recognize IPA phones (for example to evaluate pronunciation)
- You want to extract feature vectors and group them by their respective underlying speech sound
- You want to work with multilingual data
- You need a highly robust phone recognizer

This recognizer is capable of predicting phone symbols following the International Phonetic Alphabet (IPA), even for audio recorded under imperfect conditions, such as reverberation, background noise or poor recording equipment (like a smartphone). The model was trained using the multilingual [**Common Phone**](https://zenodo.org/records/5846137) dataset.

For every recognized phone, the model also emits an associated Softmax probability, as well as two feature vectors. The first comes from the CNN block, the second from the last Transformer block.

## Which IPA symbols does this model understand?
Check out `/phonetics/ipa.py` for the full list of IPA symbols.

## Got any numbers?
Sure, the model was evaluated on the test split of **Common Phone**. The following results represent Phone Error Rates (PER) in percent:

| Language | Test PER |
|:---:|:---:|
| English | 11.0 |
| French | 9.9 |
| German | 9.8 |
| Italian | 9.1 |
| Russian | 6.6 |
| Spanish | 8.8 |
| **Average** | **9.2** |

## Quick start
Creating an instance of the model and downloading the parameters is only a single line of code:
```python
    wav2vec2 = Wav2Vec2.from_pretrained("pklumpp/Wav2Vec2_CommonPhone")
```

The `example.py` script briefly summarizes the core functions of this model and how to use them. **Always standardize** your inputs before feeding them into the model (see the example)!

## Transcription server (FastAPI)
`server.py` exposes the model as an HTTP API that accepts browser-recorded audio (e.g. `audio/webm;codecs=opus`), converts it to 16kHz mono WAV via `ffmpeg`, and returns an IPA phoneme transcription.

### Requirements
- Python deps: see `requirements.txt`
- `ffmpeg` on your `PATH` (used to convert uploaded audio)
- First startup downloads `pklumpp/Wav2Vec2_CommonPhone` from Hugging Face

### Run the server
- Development (auto-reload): `python server.py`
- Explicit uvicorn: `uvicorn server:app --host 0.0.0.0 --port 8000 --reload`

The server enables permissive CORS (`allow_origins=["*"]`), which is convenient for local web apps.

### API overview
Base URL (default): `http://localhost:8000`

#### `GET /health`
Health check endpoint.

Response JSON:
- `status`: always `"healthy"` if process is up
- `model_loaded`: `true|false`
- `device`: one of `"cuda" | "mps" | "cpu"`

Example:
```bash
curl http://localhost:8000/health
```

#### `POST /transcribe`
Transcribe an uploaded audio file to IPA.

Request:
- Content-Type: `multipart/form-data`
- Form field:
  - `file` (required): audio file upload
    - The server infers the container/format from `file.filename` (extension)
    - If `filename` is missing, it assumes `webm`
    - Any format readable by `ffmpeg` should work (e.g. `webm`, `wav`, `mp3`, `m4a`)

Query parameters:
- `segmented` (optional, default `false`):
  - If `true`, also returns pause-delimited chunks under `segments`
- `word_level` (optional, default `false`):
  - If `true`, assumes a fixed prompt and returns one IPA chunk per prompt word under `words`
  - Uses silence segmentation when the number of pause segments matches the number of prompt words; otherwise falls back to an energy-based split into exactly N chunks
- `remove_pauses` (optional, default `false`):
  - If `true`, removes pause tokens like `(...)` from `ipa_text` and `symbols`
  - Also removes `(...)` pause markers from the convenience display strings (`segmented_ipa_text`, `word_ipa_text`)
- `prompt` (optional):
  - Fixed prompt text to use when `word_level=true`
  - If omitted, the server uses `FIXED_PROMPT_TEXT` from the environment
  - If neither is provided, the server returns HTTP 400

Response JSON (always present):
- `ipa_text`: string, space-separated IPA symbols
- `symbols`: list of IPA symbols
- `descriptors`: list of `{ "symbol": string, "descriptor": string }`

Optional fields:
- `segments`: list of pause-delimited chunks (only when `segmented=true`)
  - Each segment has: `start_ms`, `end_ms`, `ipa_text`, `symbols`, `descriptors`
- `segmented_ipa_text`: convenience display string (only when `segmented=true`)
- `words`: list of prompt-aligned chunks (only when `word_level=true`)
  - Each word has: `word`, `start_ms`, `end_ms`, `ipa_text`, `symbols`, `descriptors`
- `word_ipa_text`: convenience display string (only when `word_level=true`)

Common error responses:
- `400`: invalid audio / ffmpeg conversion failure / missing prompt for `word_level=true` / audio too short
- `503`: model not loaded yet
- `500`: unexpected processing error

### Examples
Basic transcription:
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@/path/to/audio.webm"
```

Transcription without pause tokens (`(...)`):
```bash
curl -X POST "http://localhost:8000/transcribe?remove_pauses=true" \
  -F "file=@/path/to/audio.webm"
```

Pause-segmented transcription:
```bash
curl -X POST "http://localhost:8000/transcribe?segmented=true" \
  -F "file=@/path/to/audio.webm"
```

Word-level transcription with explicit prompt:
```bash
curl -X POST "http://localhost:8000/transcribe?word_level=true&prompt=the%20quick%20brown%20fox" \
  -F "file=@/path/to/audio.webm"
```

Word-level transcription using an environment prompt:
```bash
export FIXED_PROMPT_TEXT="the quick brown fox"
python server.py
# then
curl -X POST "http://localhost:8000/transcribe?word_level=true" -F "file=@/path/to/audio.webm"
```

Browser / JavaScript `fetch` example:
```js
const fd = new FormData();
fd.append('file', blob, 'recording.webm');

const params = new URLSearchParams({ segmented: 'true', remove_pauses: 'true' });
const res = await fetch(`http://localhost:8000/transcribe?${params}`, {
  method: 'POST',
  body: fd,
});

if (!res.ok) throw new Error(await res.text());
const data = await res.json();
console.log(data.ipa_text, data.symbols);
```

### Audio preprocessing and segmentation knobs
`server.py` includes some best-effort voice-focused cleanup and segmentation. These are controlled via environment variables:

Noise reduction / bandpass (used for every request):
- `VOICE_NR_ENABLED` (default: `1`)
- `VOICE_NR_BANDPASS_LOW_HZ` (default: `80`)
- `VOICE_NR_BANDPASS_HIGH_HZ` (default: `7800`)
- `VOICE_NR_STRENGTH` (default: `0.8`, range `0..1`)
- `VOICE_NR_NOISE_SEC` (default: `0.25`)

Pause-based segmentation (used when `segmented=true` and as a first attempt for `word_level=true`):
- `VOICE_SEG_SILENCE_MS` (default: `250`)
- `VOICE_SEG_MIN_MS` (default: `120`)
- `VOICE_SEG_PAD_MS` (default: `40`)
- `VOICE_SEG_RMS_MULT` (default: `3.0`)
- `VOICE_SEG_RMS_PCTL` (default: `20`)

Energy-based N-chunk splitting (used when `word_level=true` and pause segments don’t match prompt length):
- `VOICE_WORD_MIN_MS` (default: `120`)
- `VOICE_WORD_SMOOTH_MS` (default: `60`)
- `VOICE_WORD_EDGE_MS` (default: `80`)

## Under what license is this work distributed
Creative Commons Zero 1.0. You can use this model for any purpose, even commercially. See the `LICENSE` for further information.

## How can I reference this work in my publication?

To cite this work, please use the following BibTex snippet:

```
@phdthesis{klumpp2024phdthesis,
  author  = "Philipp Klumpp",
  title   = "Phonetic Transfer Learning from Healthy References for the Analysis of Pathological Speech",
  school  = "Friedrich-Alexander-Universit{\"a}t Erlangen-N{\"u}rnberg",
  address = "Erlangen, Germany",
  year    = 2024,
  month   = may
}
```
