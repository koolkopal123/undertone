// Runs on-device speech-to-text (Whisper tiny, via Transformers.js) in a
// worker so a long transcription never freezes the UI thread. Audio never
// leaves the device -- this only talks to the CDN once, to fetch the model
// weights, which the browser then caches for offline reuse.

import { pipeline } from "https://esm.run/@huggingface/transformers";

let transcriber = null;
let loadingPromise = null;

function ensureLoaded(){
  if (transcriber) return Promise.resolve(transcriber);
  if (!loadingPromise){
    loadingPromise = pipeline("automatic-speech-recognition", "onnx-community/whisper-tiny", {
      dtype: "fp32",
      progress_callback: (p) => {
        if (p && p.file && typeof p.progress === "number"){
          postMessage({ type: "whisper-file-progress", file: p.file, progress: p.progress });
        }
      }
    }).then((t) => { transcriber = t; return t; });
  }
  return loadingPromise;
}

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === "load"){
    try {
      await ensureLoaded();
      postMessage({ type: "whisper-ready" });
    } catch (err){
      postMessage({ type: "whisper-error", error: String((err && err.message) || err) });
    }
    return;
  }

  if (msg.type === "transcribe"){
    let url = null;
    try {
      const t = await ensureLoaded();
      const blob = new Blob([msg.buffer], { type: msg.mimeType || "audio/webm" });
      url = URL.createObjectURL(blob);
      const result = await t(url, { chunk_length_s: 30, stride_length_s: 5 });
      postMessage({ type: "transcribe-result", id: msg.id, text: ((result && result.text) || "").trim() });
    } catch (err){
      console.error("whisper-worker transcribe() threw:", err);
      const errStr = (err && err.message) ? err.message : (err ? String(err) : "(empty error thrown in worker)");
      postMessage({ type: "transcribe-error", id: msg.id, error: errStr });
    } finally {
      if (url) URL.revokeObjectURL(url);
    }
  }
};
