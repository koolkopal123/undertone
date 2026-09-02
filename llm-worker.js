// Runs the on-device reflection model (Qwen2.5-0.5B-Instruct, via WebLLM) in
// a worker so generating a reply never freezes the UI. This uses WebLLM's
// own worker handler -- the main thread talks to it through
// CreateWebWorkerMLCEngine, which makes the remote engine feel local.

import { WebWorkerMLCEngineHandler } from "https://esm.run/@mlc-ai/web-llm";

const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (msg) => { handler.onmessage(msg); };
