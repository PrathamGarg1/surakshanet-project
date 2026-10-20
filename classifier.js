/**
 * On-device abuse classifier (Transformers.js + ONNX, fully local).
 * Only the service worker imports this file.
 */
import { pipeline, env } from '@xenova/transformers';

const MODEL_NAME = 'custom-macd-model';

// score > FLAG_THRESHOLD => flagged, score >= HIGH_THRESHOLD => "high" severity
const FLAG_THRESHOLD = 0.5;
const HIGH_THRESHOLD = 0.9;

// Local files only, no network.
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = chrome.runtime.getURL('assets/models/');
env.useBrowserCache = false; // CacheStorage rejects chrome-extension:// URLs
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;
env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL('assets/');

let modelPromise = null;

/** Loads the model once; every caller shares the same promise. */
export function loadModel() {
  if (!modelPromise) {
    modelPromise = pipeline('text-classification', MODEL_NAME, { top_k: null }).catch((error) => {
      modelPromise = null; // allow a retry next time
      throw error;
    });
  }
  return modelPromise;
}

/** Returns P(abusive) from the model output, whatever the labels are called. */
function abusiveScore(results) {
  const rows = Array.isArray(results) ? results : [results];
  const find = (...names) => rows.find((r) => names.includes(String(r.label).toLowerCase()));

  const abusive = find('abusive', 'label_0');
  if (abusive) return abusive.score;

  const safe = find('non-abusive', 'label_1');
  if (safe) return 1 - safe.score;

  return 0;
}

/** @returns {Promise<{flagged: boolean, score: number, severity: 'high'|'medium'|null}>} */
export async function classify(text) {
  const model = await loadModel();
  const score = abusiveScore(await model(text));
  const flagged = score > FLAG_THRESHOLD;
  const severity = !flagged ? null : score >= HIGH_THRESHOLD ? 'high' : 'medium';
  return { flagged, score, severity };
}
