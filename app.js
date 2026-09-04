// =====================================================================
// Undertone -- on-device voice journal
// Everything here runs locally: recording, transcription (Whisper, in
// whisper-worker.js) and reflection (Qwen2.5-0.5B via WebLLM, in
// llm-worker.js). No servers, no accounts, no API keys.
// =====================================================================

import { CreateWebWorkerMLCEngine } from "https://esm.run/@mlc-ai/web-llm";

// ---------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------
const DB_NAME = "undertone-db";
const DB_VERSION = 1;
const LLM_MODEL_ID = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC";
const MAX_CONTEXT_TURNS = 8; // most recent user+assistant turns kept per entry chat

const CHIP_PROMPTS = {
  summary: {
    label: "Summary",
    instruction: "Give me a short, warm summary of this journal entry in 2-3 sentences."
  },
  actions: {
    label: "Action Items",
    instruction:
      'Pull out any concrete action items or to-dos from this entry. List each one on its own line ' +
      'starting with "- [ ] ". If there genuinely are none, say so briefly instead of inventing any.'
  },
  organize: {
    label: "Organize Discussion",
    instruction:
      "Organize the different threads or topics in this entry into a few short labeled groups, so it's " +
      "easier to see everything I touched on."
  }
};

const BUILTIN_PROMPTS = [
  "What's been sitting with you today?",
  "What are you grateful for right now?",
  "What's one thing you want to remember from today?",
  "What took more out of you today than you expected?",
  "What's something you're looking forward to?"
];

const monthNames = ["January","February","March","April","May","June","July","August","September","October","November","December"];

// ---------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------
function el(id){ return document.getElementById(id); }
function uid(prefix){ return prefix + "_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8); }
function keyOf(d){ return d.getFullYear() + "-" + String(d.getMonth()+1).padStart(2,"0") + "-" + String(d.getDate()).padStart(2,"0"); }
function todayKey(){ return keyOf(new Date()); }
function formatKey(k){
  const parts = k.split("-");
  return monthNames[parseInt(parts[1],10)-1].slice(0,3) + " " + parseInt(parts[2],10);
}

let toastTimer = null;
function showToast(text){
  const t = el("toast");
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 3200);
}

// ---------------------------------------------------------------------
// IndexedDB layer
// ---------------------------------------------------------------------
let db;

function openDB(){
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const d = e.target.result;
      if (!d.objectStoreNames.contains("entries")){
        const store = d.createObjectStore("entries", { keyPath: "id" });
        store.createIndex("date", "date");
      }
      if (!d.objectStoreNames.contains("masterActionItems")){
        d.createObjectStore("masterActionItems", { keyPath: "id" });
      }
      if (!d.objectStoreNames.contains("prompts")){
        d.createObjectStore("prompts", { keyPath: "id" });
      }
      if (!d.objectStoreNames.contains("settings")){
        d.createObjectStore("settings", { keyPath: "key" });
      }
    };
    req.onsuccess = (e) => { db = e.target.result; resolve(db); };
    req.onerror = (e) => reject(e.target.error);
  });
}

function dbGet(store, key){
  return new Promise((resolve, reject) => {
    const req = db.transaction(store, "readonly").objectStore(store).get(key);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}
function dbGetAll(store){
  return new Promise((resolve, reject) => {
    const req = db.transaction(store, "readonly").objectStore(store).getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
}
function dbGetAllByIndex(store, indexName, value){
  return new Promise((resolve, reject) => {
    const req = db.transaction(store, "readonly").objectStore(store).index(indexName).getAll(value);
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
}
function dbPut(store, value){
  return new Promise((resolve, reject) => {
    const req = db.transaction(store, "readwrite").objectStore(store).put(value);
    req.onsuccess = () => resolve(value);
    req.onerror = () => reject(req.error);
  });
}
function dbDelete(store, key){
  return new Promise((resolve, reject) => {
    const req = db.transaction(store, "readwrite").objectStore(store).delete(key);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}
function dbClear(store){
  return new Promise((resolve, reject) => {
    const req = db.transaction(store, "readwrite").objectStore(store).clear();
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

async function seedBuiltInPrompts(){
  const existing = await dbGetAll("prompts");
  if (existing.length) return;
  for (const text of BUILTIN_PROMPTS){
    await dbPut("prompts", { id: uid("bp"), text, active: true, isBuiltIn: true });
  }
}

// ---------------------------------------------------------------------
// State
// ---------------------------------------------------------------------
const state = {
  tab: "journal",
  selectedKey: todayKey(),
  currentEntryId: null,
  prompts: [],
  theme: "light"
};

// ---------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------
async function loadTheme(){
  const row = await dbGet("settings", "theme");
  state.theme = (row && row.value) || "light";
  applyTheme();
}
function applyTheme(){
  document.documentElement.setAttribute("data-theme", state.theme);
  const sw = el("darkSwitch");
  if (sw) sw.classList.toggle("on", state.theme === "dark");
}
async function toggleTheme(){
  state.theme = state.theme === "dark" ? "light" : "dark";
  applyTheme();
  await dbPut("settings", { key: "theme", value: state.theme });
}

// ---------------------------------------------------------------------
// Model loading (Whisper worker + WebLLM worker)
// ---------------------------------------------------------------------
let whisperWorker = null;
let whisperReady = false;
const pendingTranscriptions = new Map();

let llmEngine = null;
let llmReady = false;

function setLoadingStatus(text){ const s = el("loadingStatus"); if (s) s.textContent = text; }
function setLoadingBar(pct){
  const f = el("loadingBarFill");
  if (f) f.style.width = Math.max(2, Math.min(100, pct)) + "%";
}

// Whisper (transformers.js) load progress is per-file; average what we've seen so far.
const whisperFileProgress = {};
function onWhisperWorkerMessage(e){
  const msg = e.data;
  if (msg.type === "whisper-file-progress"){
    whisperFileProgress[msg.file] = msg.progress;
    const vals = Object.values(whisperFileProgress);
    const avg = vals.length ? vals.reduce((a,b)=>a+b,0) / vals.length : 0;
    combinedProgress.whisper = avg;
    updateCombinedProgress("Preparing transcription\u2026");
  } else if (msg.type === "whisper-ready"){
    whisperReady = true;
    combinedProgress.whisper = 100;
    updateCombinedProgress();
  } else if (msg.type === "whisper-error"){
    onModelLoadError("transcription", msg.error);
  } else if (msg.type === "transcribe-result"){
    const p = pendingTranscriptions.get(msg.id);
    if (p){ p.resolve(msg.text); pendingTranscriptions.delete(msg.id); }
  } else if (msg.type === "transcribe-error"){
    const p = pendingTranscriptions.get(msg.id);
    if (p){ p.reject(new Error(msg.error)); pendingTranscriptions.delete(msg.id); }
  }
}

const combinedProgress = { whisper: 0, llm: 0 };
function updateCombinedProgress(statusText){
  const pct = (combinedProgress.whisper + combinedProgress.llm) / 2;
  setLoadingBar(pct);
  if (statusText) setLoadingStatus(statusText);
}

function loadWhisper(){
  return new Promise((resolve, reject) => {
    whisperWorker = new Worker("whisper-worker.js", { type: "module" });
    whisperWorker.onmessage = (e) => {
      onWhisperWorkerMessage(e);
      if (e.data.type === "whisper-ready") resolve();
      if (e.data.type === "whisper-error") reject(new Error(e.data.error));
    };
    whisperWorker.onerror = (err) => reject(err);
    whisperWorker.postMessage({ type: "load" });
  });
}

async function loadLLM(){
  if (!navigator.gpu){
    throw new Error(
      "This phone's browser doesn't support WebGPU yet, which the on-device reflection model needs. " +
      "Try updating your browser, or use a recent version of Chrome."
    );
  }
  const worker = new Worker("llm-worker.js", { type: "module" });
  llmEngine = await CreateWebWorkerMLCEngine(worker, LLM_MODEL_ID, {
    initProgressCallback: (report) => {
      combinedProgress.llm = (report.progress || 0) * 100;
      updateCombinedProgress(report.text || "Preparing reflection\u2026");
    }
  });
  llmReady = true;
}

function onModelLoadError(which, message){
  setLoadingStatus("Couldn't load " + which + ".");
  el("loadingNote").textContent = String(message || "Something went wrong. Check your connection and try again.");
  el("loadingRetryBtn").hidden = false;
}

async function loadModelsWithRetry(){
  el("loadingRetryBtn").hidden = true;
  el("loadingNote").textContent =
    "First launch downloads two small on-device models (about 450MB total) for transcription and " +
    "reflection. Everything after this runs fully offline, and nothing ever leaves your phone.";
  try {
    await Promise.all([loadWhisper(), loadLLM()]);
    return true;
  } catch (err){
    onModelLoadError("the on-device models", err && err.message ? err.message : err);
    return false;
  }
}

function transcribeAudio(id, audioBlob){
  return new Promise(async (resolve, reject) => {
    pendingTranscriptions.set(id, { resolve, reject });
    try {
      const buffer = await audioBlob.arrayBuffer();
      whisperWorker.postMessage(
        { type: "transcribe", id, mimeType: audioBlob.type, buffer },
        [buffer]
      );
    } catch (err){
      pendingTranscriptions.delete(id);
      reject(err);
    }
  });
}

async function chatWithLLM(messages){
  const reply = await llmEngine.chat.completions.create({ messages });
  return reply.choices[0].message.content;
}

// ---------------------------------------------------------------------
// Action item parsing (markdown checklist lines -> real checkable items)
// ---------------------------------------------------------------------
function extractActionItems(text){
  const items = [];
  text.split("\n").forEach((line) => {
    const m = line.match(/^\s*[-*]\s*\[ \]\s*(.+)/);
    if (m) items.push({ id: uid("a"), text: m[1].trim(), done: false, promoted: false });
  });
  return items;
}
function stripChecklistSyntax(text){
  const cleaned = text.split("\n").filter((line) => !/^\s*[-*]\s*\[ \]\s*/.test(line)).join("\n").trim();
  return cleaned || text.trim();
}

function trimForContext(messages){
  const system = messages[0];
  const rest = messages.slice(1);
  const trimmed = rest.length > MAX_CONTEXT_TURNS ? rest.slice(rest.length - MAX_CONTEXT_TURNS) : rest;
  return [system, ...trimmed];
}

// ---------------------------------------------------------------------
// Calendar + bottom panel (Journal tab)
// ---------------------------------------------------------------------
async function renderCalendar(){
  const now = new Date();
  const y = now.getFullYear(), m = now.getMonth();
  el("calTitle").textContent = monthNames[m] + " " + y;

  const monthEntries = await dbGetAll("entries");
  const datesWithEntries = new Set(monthEntries.map((e) => e.date));

  const first = new Date(y, m, 1).getDay();
  const daysInMonth = new Date(y, m + 1, 0).getDate();
  const grid = el("calGrid");
  grid.innerHTML = "";
  ["S","M","T","W","T","F","S"].forEach((d) => {
    const dow = document.createElement("div");
    dow.className = "cal-dow"; dow.textContent = d;
    grid.appendChild(dow);
  });
  for (let i = 0; i < first; i++){
    const blank = document.createElement("div");
    blank.className = "cal-day blank";
    grid.appendChild(blank);
  }
  const tKey = todayKey();
  for (let day = 1; day <= daysInMonth; day++){
    const k = keyOf(new Date(y, m, day));
    const btn = document.createElement("button");
    btn.className = "cal-day";
    if (k === tKey) btn.classList.add("today");
    if (k === state.selectedKey) btn.classList.add("selected");
    btn.innerHTML = "<span>" + day + "</span>";
    if (datesWithEntries.has(k)){
      const dot = document.createElement("span");
      dot.className = "cal-dot";
      btn.appendChild(dot);
    }
    btn.addEventListener("click", () => {
      state.selectedKey = k;
      renderCalendar();
      renderBottomPanel();
    });
    grid.appendChild(btn);
  }
}

function micSvg(){
  return '<svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">' +
    '<rect x="8" y="2" width="6" height="12" rx="3" fill="currentColor"/>' +
    '<path d="M5 10.5C5 14 7.5 16.5 11 16.5C14.5 16.5 17 14 17 10.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>' +
    '<line x1="11" y1="16.5" x2="11" y2="19.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>' +
    "</svg>";
}
function stopSvg(){
  return '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">' +
    '<rect x="5" y="5" width="10" height="10" rx="2" fill="var(--ivory)"/></svg>';
}

function todaysPromptText(){
  const active = state.prompts.filter((p) => p.active);
  if (!active.length) return null;
  const k = state.selectedKey;
  let sum = 0; for (let i = 0; i < k.length; i++) sum += k.charCodeAt(i);
  return active[sum % active.length].text;
}

async function renderBottomPanel(){
  const actionRow = el("actionRow");
  const scrollArea = el("entriesScroll");
  actionRow.innerHTML = "";
  scrollArea.innerHTML = "";

  const isToday = state.selectedKey === todayKey();
  const dayEntries = (await dbGetAllByIndex("entries", "date", state.selectedKey))
    .sort((a, b) => a.createdAt - b.createdAt);

  if (recordingState.active && recordingState.dateKey === state.selectedKey){
    renderRecordingActionRow(actionRow);
  } else if (isToday && !dayEntries.length){
    const promptText = todaysPromptText();
    const promptHtml = promptText ? '<p class="prompt-text">&#127807; ' + escapeHtml(promptText) + "</p>" : "";
    actionRow.innerHTML =
      '<div class="prompt-state">' + promptHtml +
        '<div class="record-row">' +
          '<button class="record-btn" id="recordBtn" aria-label="Record entry">' + micSvg() + "</button>" +
          '<span class="record-hint">Tap to Begin</span>' +
        "</div>" +
      "</div>";
    el("recordBtn").addEventListener("click", handleRecordButtonTap);
  } else {
    if (isToday){
      const addBtn = document.createElement("button");
      addBtn.className = "add-entry-btn";
      addBtn.innerHTML = micSvg() + "<span>New entry</span>";
      addBtn.addEventListener("click", handleRecordButtonTap);
      actionRow.appendChild(addBtn);
    } else {
      const backBtn = document.createElement("button");
      backBtn.className = "back-today";
      backBtn.textContent = "Back to Today";
      backBtn.addEventListener("click", () => {
        state.selectedKey = todayKey();
        renderCalendar();
        renderBottomPanel();
      });
      actionRow.appendChild(backBtn);
    }
  }

  if (!dayEntries.length){
    if (!(isToday && !recordingState.active)){
      const empty = document.createElement("div");
      empty.className = "empty-day";
      empty.textContent = "No Entry This Day.";
      scrollArea.appendChild(empty);
    }
    return;
  }

  const label = document.createElement("div");
  label.className = "day-label";
  label.textContent = dayEntries.length + " Entr" + (dayEntries.length > 1 ? "ies" : "y");
  scrollArea.appendChild(label);

  dayEntries.forEach((entry) => {
    const card = document.createElement("button");
    card.className = "entry-card";
    const time = new Date(entry.createdAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    const snippet = entry.status === "transcribing"
      ? "Transcribing\u2026"
      : entry.status === "error" && !entry.transcript
        ? "Transcription failed \u2014 tap to fix"
        : (entry.transcript || "").slice(0, 90) + ((entry.transcript || "").length > 90 ? "\u2026" : "");
    card.innerHTML = '<div class="entry-time">' + time + '</div><div class="entry-snippet">' + escapeHtml(snippet) + "</div>";
    card.addEventListener("click", () => openEntry(entry.id));
    scrollArea.appendChild(card);
  });
}

function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, (c) => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[c]));
}

// ---------------------------------------------------------------------
// Recording flow
// ---------------------------------------------------------------------
const recordingState = { active: false, dateKey: null, startedAt: 0, timerHandle: null, mediaRecorder: null, chunks: [], stream: null };

function renderRecordingActionRow(container){
  const elapsed = Math.floor((Date.now() - recordingState.startedAt) / 1000);
  const mm = Math.floor(elapsed / 60), ss = String(elapsed % 60).padStart(2, "0");
  container.innerHTML =
    '<div class="prompt-state">' +
      '<div class="record-row">' +
        '<button class="record-btn recording" id="recordBtn" aria-label="Stop recording">' + stopSvg() + "</button>" +
        '<span class="record-hint live"><span class="recording-pulse"></span>Recording\u2026 ' + mm + ":" + ss + "</span>" +
      "</div>" +
    "</div>";
  el("recordBtn").addEventListener("click", handleRecordButtonTap);
}

function pickSupportedMimeType(){
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  for (const c of candidates){
    if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(c)) return c;
  }
  return "";
}

async function handleRecordButtonTap(){
  if (recordingState.active){
    stopRecording();
    return;
  }
  if (!whisperReady){
    showToast("Transcription is still loading -- one moment.");
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err){
    showToast("Microphone access is needed to record an entry.");
    return;
  }
  const mimeType = pickSupportedMimeType();
  const mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  recordingState.chunks = [];
  mediaRecorder.ondataavailable = (e) => { if (e.data && e.data.size) recordingState.chunks.push(e.data); };
  mediaRecorder.onstop = onRecordingStopped;
  recordingState.mediaRecorder = mediaRecorder;
  recordingState.stream = stream;
  recordingState.active = true;
  recordingState.dateKey = todayKey();
  recordingState.startedAt = Date.now();
  mediaRecorder.start();
  recordingState.timerHandle = setInterval(() => {
    if (state.selectedKey === recordingState.dateKey) renderBottomPanel();
  }, 1000);
  if (state.selectedKey !== recordingState.dateKey) state.selectedKey = recordingState.dateKey;
  renderCalendar();
  renderBottomPanel();
}

function stopRecording(){
  clearInterval(recordingState.timerHandle);
  if (recordingState.mediaRecorder && recordingState.mediaRecorder.state !== "inactive"){
    recordingState.mediaRecorder.stop();
  }
  if (recordingState.stream) recordingState.stream.getTracks().forEach((t) => t.stop());
}

async function onRecordingStopped(){
  const mimeType = recordingState.mediaRecorder.mimeType || "audio/webm";
  const blob = new Blob(recordingState.chunks, { type: mimeType });
  const dateKey = recordingState.dateKey;
  recordingState.active = false;
  recordingState.mediaRecorder = null;
  recordingState.stream = null;

  if (blob.size < 500){
    showToast("That recording was too short to save.");
    renderBottomPanel();
    return;
  }

  const now = new Date();
  const entry = {
    id: uid("entry"),
    date: dateKey,
    createdAt: now.getTime(),
    promptShown: dateKey === todayKey() ? todaysPromptText() : null,
    audioBlob: blob,
    transcript: "",
    transcriptEdited: false,
    conversation: [],
    status: "transcribing"
  };
  await dbPut("entries", entry);
  renderCalendar();
  renderBottomPanel();
  openEntry(entry.id);
  runTranscription(entry.id, blob);
}

async function runTranscription(entryId, blob){
  try {
    const text = await transcribeAudio(entryId, blob);
    const entry = await dbGet("entries", entryId);
    if (!entry) return;
    entry.transcript = text || "(No speech detected -- tap Edit to write it yourself.)";
    entry.status = "transcribed";
    await dbPut("entries", entry);
  } catch (err){
    console.error("Transcription failed, raw error object:", err);
    const entry = await dbGet("entries", entryId);
    let errMsg = (err && err.message) ? err.message : (err ? String(err) : "");
    if (!errMsg) errMsg = "(empty error \u2014 open your browser devtools console for the real message)";
    if (entry){
      entry.status = "error";
      entry.transcript = "";
      entry.transcriptError = errMsg;
      await dbPut("entries", entry);
    }
    showToast("Transcription failed: " + errMsg);
  }
  if (state.currentEntryId === entryId) renderEntry();
  if (state.tab === "journal") renderBottomPanel();
}

// ---------------------------------------------------------------------
// Entry detail screen
// ---------------------------------------------------------------------
async function openEntry(entryId){
  state.currentEntryId = entryId;
  showScreen("entry");
  await renderEntry();
}

function showScreen(which){
  const overlay = which === "entry" || which === "prompts";
  el("screen-journal").hidden = which !== "journal";
  el("screen-todo").hidden = which !== "todo";
  el("screen-entry").hidden = which !== "entry";
  el("screen-prompts").hidden = which !== "prompts";
  el("tabbar").hidden = overlay;
  el("settingsFab").hidden = overlay;
  setSettingsOpen(false);
}

function closeOverlay(){
  showScreen(state.tab);
  renderCalendar();
  renderBottomPanel();
  if (state.tab === "todo") renderTodo();
}

async function renderEntry(){
  const entry = await dbGet("entries", state.currentEntryId);
  if (!entry){ closeOverlay(); return; }

  const time = new Date(entry.createdAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  el("entryHeadDate").innerHTML = formatKey(entry.date) + " &middot; " + time;

  const body = el("entryBody");
  body.innerHTML = "";

  if (entry.status === "transcribing"){
    const note = document.createElement("div");
    note.className = "processing-note";
    note.textContent = "Transcribing your recording\u2026";
    body.appendChild(note);
    el("entryChips").hidden = true;
    el("composerInput").disabled = true;
    el("composerSend").disabled = true;
    return;
  }
  el("entryChips").hidden = false;
  el("composerInput").disabled = false;
  el("composerSend").disabled = false;

  const card = document.createElement("div");
  card.className = "transcript-card";

  if (entry._confirmingDelete){
    card.innerHTML =
      '<div class="transcript-top"><span class="transcript-label">Delete This Entry?</span></div>' +
      '<div class="confirm-row">' +
        '<button class="delete-link" id="confirmDelete">Yes, Delete</button>' +
        '<button class="edit-link" id="cancelDelete">Cancel</button>' +
      "</div>";
    body.appendChild(card);
    el("confirmDelete").addEventListener("click", () => deleteCurrentEntry(entry.id));
    el("cancelDelete").addEventListener("click", () => { entry._confirmingDelete = false; renderEntry(); });
  } else if (entry._editing){
    card.innerHTML =
      '<div class="transcript-top"><span class="transcript-label">Transcript</span>' +
      '<button class="edit-link" id="editLink">Done</button></div>' +
      '<textarea class="transcript-edit" id="editArea" rows="4">' + escapeHtml(entry.transcript) + "</textarea>";
    body.appendChild(card);
    el("editLink").addEventListener("click", async () => {
      entry.transcript = el("editArea").value;
      entry.transcriptEdited = true;
      entry._editing = false;
      await dbPut("entries", entry);
      renderEntry();
    });
  } else {
    const transcriptHtml = entry.transcript
      ? escapeHtml(entry.transcript)
      : '<span style="color:var(--danger)">Transcription failed' + (entry.transcriptError ? (": " + escapeHtml(entry.transcriptError)) : "") + '. Tap Edit to write it yourself, or Delete to discard this entry.</span>';
    card.innerHTML =
      '<div class="transcript-top"><span class="transcript-label">Transcript</span>' +
      '<div class="transcript-actions"><button class="edit-link" id="editLink">Edit</button>' +
      '<button class="delete-link" id="deleteLink">Delete</button></div></div>' +
      '<div class="transcript-text">' + transcriptHtml + "</div>";
    body.appendChild(card);
    el("editLink").addEventListener("click", () => { entry._editing = true; renderEntry(); });
    el("deleteLink").addEventListener("click", () => { entry._confirmingDelete = true; renderEntry(); });
  }

  (entry.conversation || []).forEach((msg) => body.appendChild(renderMsg(entry, msg)));
  body.scrollTop = body.scrollHeight;
}

function renderMsg(entry, msg){
  const div = document.createElement("div");
  div.className = "msg " + msg.role;
  div.innerHTML = escapeHtml(msg.text).replace(/\*\*(.*?)\*\*/g, "<b>$1</b>").replace(/\n/g, "<br>");

  if (msg.actionItems && msg.actionItems.length){
    const wrap = document.createElement("div");
    wrap.className = "msg-actions";
    msg.actionItems.forEach((item) => {
      const row = document.createElement("div");
      row.className = "msg-action";
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.checked = item.done;
      cb.addEventListener("change", () => setActionItemDone(entry.id, item.id, cb.checked));
      const span = document.createElement("span");
      span.textContent = item.text;
      const addBtn = document.createElement("button");
      addBtn.className = "add-master" + (item.promoted ? " added" : "");
      addBtn.setAttribute("aria-label", item.promoted ? "Remove from to-do" : "Add to to-do");
      addBtn.textContent = item.promoted ? "\u2713" : "+";
      addBtn.addEventListener("click", () => toggleActionItemPromotion(entry.id, item.id, addBtn));
      row.appendChild(cb); row.appendChild(span); row.appendChild(addBtn);
      wrap.appendChild(row);
    });
    div.appendChild(wrap);
  }
  return div;
}

async function deleteCurrentEntry(entryId){
  await dbDelete("entries", entryId);
  const all = await dbGetAll("masterActionItems");
  for (const row of all){ if (row.entryId === entryId) await dbDelete("masterActionItems", row.id); }
  closeOverlay();
}

async function setActionItemDone(entryId, itemId, done){
  const entry = await dbGet("entries", entryId);
  if (!entry) return;
  let found = null;
  (entry.conversation || []).forEach((m) => (m.actionItems || []).forEach((it) => { if (it.id === itemId){ it.done = done; found = it; } }));
  await dbPut("entries", entry);
  if (found && found.promoted){
    const all = await dbGetAll("masterActionItems");
    const row = all.find((r) => r.itemId === itemId);
    if (row){ row.done = done; await dbPut("masterActionItems", row); }
  }
  if (state.tab === "todo") renderTodo();
}

async function toggleActionItemPromotion(entryId, itemId, buttonEl){
  const entry = await dbGet("entries", entryId);
  if (!entry) return;
  let target = null;
  (entry.conversation || []).forEach((m) => (m.actionItems || []).forEach((it) => { if (it.id === itemId) target = it; }));
  if (!target) return;

  target.promoted = !target.promoted;
  await dbPut("entries", entry);

  if (target.promoted){
    await dbPut("masterActionItems", { id: uid("master"), entryId, itemId, dateKey: entry.date, text: target.text, done: target.done });
  } else {
    const all = await dbGetAll("masterActionItems");
    const row = all.find((r) => r.itemId === itemId);
    if (row) await dbDelete("masterActionItems", row.id);
  }
  buttonEl.textContent = target.promoted ? "\u2713" : "+";
  buttonEl.classList.toggle("added", target.promoted);
  buttonEl.setAttribute("aria-label", target.promoted ? "Remove from to-do" : "Add to to-do");
}

// ---------------------------------------------------------------------
// Chat / analysis
// ---------------------------------------------------------------------
let sending = false;

async function sendChipMessage(kind){
  if (sending) return;
  const cfg = CHIP_PROMPTS[kind];
  await sendMessage(cfg.label, cfg.instruction);
}

async function sendCustomMessage(text){
  if (sending) return;
  await sendMessage(text, text);
}

async function sendMessage(displayText, modelInstruction){
  sending = true;
  const entry = await dbGet("entries", state.currentEntryId);
  if (!entry){ sending = false; return; }

  entry.conversation = entry.conversation || [];
  entry.conversation.push({ id: uid("m"), role: "user", text: displayText, promptOverride: modelInstruction, timestamp: Date.now() });
  await dbPut("entries", entry);
  await renderEntry();

  const body = el("entryBody");
  const typing = document.createElement("div");
  typing.className = "msg typing";
  typing.textContent = "Thinking\u2026";
  body.appendChild(typing);
  body.scrollTop = body.scrollHeight;

  if (!llmReady){
    typing.remove();
    showToast("Reflection is still loading -- one moment.");
    sending = false;
    return;
  }

  const systemPrompt =
    "You are a calm, concise journaling companion helping someone reflect on one voice-journal entry. " +
    'Their transcript for this entry is:\n\n"' + entry.transcript + '"\n\n' +
    "Respond only to what they ask about this entry. Keep replies warm but brief. When listing action " +
    'items, format each one on its own line starting with "- [ ] ".';

  const apiMessages = [{ role: "system", content: systemPrompt }];
  entry.conversation.forEach((m) => {
    if (m.role === "user") apiMessages.push({ role: "user", content: m.promptOverride || m.text });
    else apiMessages.push({ role: "assistant", content: m.rawText || m.text });
  });

  let replyText;
  try {
    replyText = await chatWithLLM(trimForContext(apiMessages));
  } catch (err){
    typing.remove();
    showToast("Couldn't get a reply -- try again.");
    sending = false;
    return;
  }
  typing.remove();

  const actionItems = extractActionItems(replyText);
  const assistantMsg = { id: uid("m"), role: "assistant", rawText: replyText, text: stripChecklistSyntax(replyText), timestamp: Date.now() };
  if (actionItems.length) assistantMsg.actionItems = actionItems;
  entry.conversation.push(assistantMsg);
  await dbPut("entries", entry);
  await renderEntry();
  sending = false;
}

// ---------------------------------------------------------------------
// To-do tab
// ---------------------------------------------------------------------
async function renderTodo(){
  const wrap = el("todoWrap");
  wrap.innerHTML = '<div class="todo-title">To-do</div>';
  const items = await dbGetAll("masterActionItems");
  if (!items.length){
    wrap.innerHTML += '<div class="todo-empty">Nothing promoted from an entry yet.</div>';
    return;
  }
  const byDate = {};
  items.forEach((row) => { (byDate[row.dateKey] = byDate[row.dateKey] || []).push(row); });
  Object.keys(byDate).sort().reverse().forEach((k) => {
    const label = document.createElement("div");
    label.className = "todo-group-label";
    label.textContent = formatKey(k);
    wrap.appendChild(label);
    byDate[k].forEach((row) => {
      const rowEl = document.createElement("div");
      rowEl.className = "todo-row";
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.checked = row.done;
      cb.addEventListener("change", () => setActionItemDone(row.entryId, row.itemId, cb.checked));
      const span = document.createElement("span");
      span.className = "todo-text" + (row.done ? " done" : "");
      span.textContent = row.text;
      rowEl.appendChild(cb); rowEl.appendChild(span);
      wrap.appendChild(rowEl);
    });
  });
}

// ---------------------------------------------------------------------
// Prompts screen
// ---------------------------------------------------------------------
async function refreshPromptsCache(){
  state.prompts = await dbGetAll("prompts");
}

function openPrompts(){
  showScreen("prompts");
  renderPrompts();
}

function promptRow(p){
  const row = document.createElement("div");
  row.className = "prompt-row";
  const cb = document.createElement("input");
  cb.type = "checkbox"; cb.checked = p.active;
  cb.addEventListener("change", async () => { p.active = cb.checked; await dbPut("prompts", p); await refreshPromptsCache(); });
  row.appendChild(cb);

  if (!p.isBuiltIn && p._editing){
    const input = document.createElement("input");
    input.type = "text"; input.value = p.text; input.className = "prompt-edit-input";
    row.appendChild(input);
    const saveBtn = document.createElement("button");
    saveBtn.className = "edit-link"; saveBtn.textContent = "Save";
    saveBtn.addEventListener("click", async () => {
      if (input.value.trim()) p.text = input.value.trim();
      p._editing = false;
      await dbPut("prompts", p);
      await refreshPromptsCache();
      renderPrompts();
    });
    const actionsWrap = document.createElement("div");
    actionsWrap.className = "prompt-row-actions";
    actionsWrap.appendChild(saveBtn);
    row.appendChild(actionsWrap);
    return row;
  }

  const span = document.createElement("span");
  span.className = "prompt-row-text";
  span.textContent = p.text;
  row.appendChild(span);

  const actions = document.createElement("div");
  actions.className = "prompt-row-actions";
  if (p.isBuiltIn){
    const dupBtn = document.createElement("button");
    dupBtn.className = "edit-link"; dupBtn.textContent = "Duplicate";
    dupBtn.addEventListener("click", async () => {
      await dbPut("prompts", { id: uid("cp"), text: p.text, active: true, isBuiltIn: false });
      await refreshPromptsCache();
      renderPrompts();
    });
    actions.appendChild(dupBtn);
  } else {
    const editBtn = document.createElement("button");
    editBtn.className = "edit-link"; editBtn.textContent = "Edit";
    editBtn.addEventListener("click", () => { p._editing = true; renderPrompts(); });
    const delBtn = document.createElement("button");
    delBtn.className = "delete-link"; delBtn.textContent = "Delete";
    delBtn.addEventListener("click", async () => {
      await dbDelete("prompts", p.id);
      await refreshPromptsCache();
      renderPrompts();
    });
    actions.appendChild(editBtn); actions.appendChild(delBtn);
  }
  row.appendChild(actions);
  return row;
}

function renderPrompts(){
  const body = el("promptsBody");
  body.innerHTML = "";
  const builtins = state.prompts.filter((p) => p.isBuiltIn);
  const customs = state.prompts.filter((p) => !p.isBuiltIn);

  const b1 = document.createElement("div"); b1.className = "todo-group-label"; b1.textContent = "Built-in";
  body.appendChild(b1);
  builtins.forEach((p) => body.appendChild(promptRow(p)));

  const b2 = document.createElement("div"); b2.className = "todo-group-label"; b2.textContent = "Your Prompts";
  body.appendChild(b2);
  if (!customs.length){
    const empty = document.createElement("div");
    empty.className = "todo-empty";
    empty.textContent = "None yet -- duplicate a built-in above, or add your own below.";
    body.appendChild(empty);
  }
  customs.forEach((p) => body.appendChild(promptRow(p)));
}

// ---------------------------------------------------------------------
// Settings panel
// ---------------------------------------------------------------------
function setSettingsOpen(open){
  el("settingsPanel").hidden = !open;
  el("settingsBackdrop").hidden = !open;
  el("resetConfirm").hidden = true;
}

async function exportData(){
  const entries = (await dbGetAll("entries")).map((e) => {
    const copy = Object.assign({}, e);
    delete copy.audioBlob; // keeps the backup small and portable; audio stays on-device only
    delete copy._editing; delete copy._confirmingDelete;
    return copy;
  });
  const masterActionItems = await dbGetAll("masterActionItems");
  const prompts = await dbGetAll("prompts");
  const payload = { app: "undertone", version: 1, exportedAt: new Date().toISOString(), entries, masterActionItems, prompts };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "undertone-backup-" + todayKey() + ".json";
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
  showToast("Backup downloaded.");
}

async function importDataFromFile(file){
  let payload;
  try { payload = JSON.parse(await file.text()); }
  catch (err){ showToast("That file couldn't be read."); return; }
  if (!payload || !Array.isArray(payload.entries)){ showToast("That doesn't look like an Undertone backup."); return; }
  for (const e of payload.entries) await dbPut("entries", e);
  for (const m of (payload.masterActionItems || [])) await dbPut("masterActionItems", m);
  for (const p of (payload.prompts || [])) await dbPut("prompts", p);
  await refreshPromptsCache();
  renderCalendar(); renderBottomPanel();
  showToast("Import complete.");
}

async function resetAllData(){
  await dbClear("entries");
  await dbClear("masterActionItems");
  await dbClear("prompts");
  await seedBuiltInPrompts();
  await refreshPromptsCache();
  state.selectedKey = todayKey();
  setSettingsOpen(false);
  renderCalendar(); renderBottomPanel();
  showToast("All data reset.");
}

// ---------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------
function wireStaticEventListeners(){
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      setSettingsOpen(false);
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      state.tab = tab.dataset.tab;
      el("screen-journal").hidden = state.tab !== "journal";
      el("screen-todo").hidden = state.tab !== "todo";
      if (state.tab === "todo") renderTodo();
    });
  });

  el("entryBack").addEventListener("click", closeOverlay);
  el("promptsBack").addEventListener("click", closeOverlay);

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => sendChipMessage(chip.dataset.chip));
  });
  el("composerSend").addEventListener("click", () => {
    const input = el("composerInput");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    sendCustomMessage(text);
  });
  el("composerInput").addEventListener("keydown", (e) => { if (e.key === "Enter") el("composerSend").click(); });

  el("addPromptBtn").addEventListener("click", async () => {
    const input = el("newPromptInput");
    const err = el("promptError");
    if (!input.value.trim()){ err.hidden = false; return; }
    err.hidden = true;
    await dbPut("prompts", { id: uid("cp"), text: input.value.trim(), active: true, isBuiltIn: false });
    input.value = "";
    await refreshPromptsCache();
    renderPrompts();
  });
  el("newPromptInput").addEventListener("input", () => { el("promptError").hidden = true; });

  el("settingsFab").addEventListener("click", () => setSettingsOpen(el("settingsPanel").hidden));
  el("settingsBackdrop").addEventListener("click", () => setSettingsOpen(false));
  el("darkToggleRow").addEventListener("click", toggleTheme);
  el("editPromptsRow").addEventListener("click", () => { setSettingsOpen(false); openPrompts(); });
  el("exportRow").addEventListener("click", () => { setSettingsOpen(false); exportData(); });
  el("importRow").addEventListener("click", () => { setSettingsOpen(false); el("importFileInput").click(); });
  el("importFileInput").addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) importDataFromFile(file);
    e.target.value = "";
  });
  el("resetRow").addEventListener("click", () => { el("resetConfirm").hidden = false; });
  el("resetConfirmNo").addEventListener("click", () => { el("resetConfirm").hidden = true; });
  el("resetConfirmYes").addEventListener("click", resetAllData);

  el("loadingRetryBtn").addEventListener("click", boot);
}

function registerServiceWorker(){
  if ("serviceWorker" in navigator){
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }
}

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------
let wired = false;

async function boot(){
  el("loadingOverlay").hidden = false;
  setLoadingBar(2);
  setLoadingStatus("Warming up\u2026");

  await openDB();
  await seedBuiltInPrompts();
  await refreshPromptsCache();
  await loadTheme();

  if (!wired){ wireStaticEventListeners(); wired = true; }

  const ok = await loadModelsWithRetry();
  if (!ok) return;

  el("loadingOverlay").hidden = true;
  el("tabbar").hidden = false;
  el("settingsFab").hidden = false;
  showScreen("journal");
  await renderCalendar();
  await renderBottomPanel();
  registerServiceWorker();
}

document.addEventListener("DOMContentLoaded", boot);
