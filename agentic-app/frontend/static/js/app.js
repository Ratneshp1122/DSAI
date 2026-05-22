/**
 * app.js — NeuraMind frontend logic
 * Handles: file upload, drag-and-drop, chat, streaming state updates,
 *          execution plan display, clarification flow, session management
 */

'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  sessionId: null,
  fileId: null,
  fileType: null,
  isUploading: false,
  isSending: false,
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const dropZone      = $('dropZone');
const fileInput     = $('fileInput');
const uploadProgress= $('uploadProgress');
const progressFill  = $('progressFill');
const progressLabel = $('progressLabel');
const fileCard      = $('fileCard');
const fileIcon      = $('fileIcon');
const fileName      = $('fileName');
const fileMeta      = $('fileMeta');
const clearFileBtn  = $('clearFile');
const confValue     = $('confValue');
const confBar       = $('confBar');
const textPreview   = $('textPreview');
const textPreviewWrap=$('textPreviewWrap');
const capsSection   = $('capsSection');
const messages      = $('messages');
const userInput     = $('userInput');
const sendBtn       = $('sendBtn');
const sessionIdDisplay = $('sessionIdDisplay');
const chatStatus    = $('chatStatus');
const execPlan      = $('execPlan');
const planToggle    = $('planToggle');
const planSteps     = $('planSteps');
const planChevron   = document.querySelector('.plan-chevron');
const newSessionBtn = $('newSession');
const sidebarToggle = $('sidebarToggle');
const leftPanel     = document.querySelector('.left-panel');
const toastContainer= $('toastContainer');

// ── Helpers ───────────────────────────────────────────────────────────────────

function genId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function showToast(msg, type = 'info', duration = 4000) {
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  toastContainer.appendChild(t);
  setTimeout(() => {
    t.style.animation = 'toastOut 0.3s ease forwards';
    setTimeout(() => t.remove(), 300);
  }, duration);
}

function setStatus(text) {
  chatStatus.textContent = text;
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

function fileTypeIcon(type) {
  const icons = { image: '🖼', pdf: '📕', audio: '🎧', text: '📄' };
  return icons[type] || '📄';
}

function formatMeta(data) {
  const parts = [data.file_type?.toUpperCase()];
  if (data.page_count) parts.push(`${data.page_count} pages`);
  if (data.duration_seconds) parts.push(`${data.duration_seconds.toFixed(1)}s`);
  if (data.word_count) parts.push(`${data.word_count.toLocaleString()} words`);
  if (data.method) parts.push(data.method.replace('_', ' '));
  return parts.filter(Boolean).join(' · ');
}

function setConfidence(conf) {
  const pct = Math.round((conf || 0) * 100);
  confValue.textContent = `${pct}%`;
  confBar.style.width = `${pct}%`;
  confBar.className = 'conf-bar-fill ' + (pct >= 70 ? 'high' : pct >= 40 ? 'medium' : 'low');
}

// ── Upload flow ───────────────────────────────────────────────────────────────

function handleFileDrop(e) {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer?.files[0] || e.target.files[0];
  if (file) uploadFile(file);
}

async function uploadFile(file) {
  if (state.isUploading) return;
  state.isUploading = true;
  sendBtn.disabled = true;

  // Show progress
  uploadProgress.classList.remove('hidden');
  fileCard.classList.add('hidden');
  capsSection.classList.add('hidden');

  progressLabel.textContent = 'Uploading…';
  let pct = 0;
  const ticker = setInterval(() => {
    pct = Math.min(pct + Math.random() * 15, 85);
    progressFill.style.width = `${pct}%`;
  }, 300);

  try {
    const formData = new FormData();
    formData.append('file', file);

    const resp = await fetch('/api/upload', { method: 'POST', body: formData });
    clearInterval(ticker);

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `Upload failed (${resp.status})`);
    }

    const data = await resp.json();

    // Complete progress bar
    progressFill.style.width = '100%';
    progressLabel.textContent = 'Extraction complete!';
    await delay(600);

    // Store session
    state.sessionId = data.session_id;
    state.fileId = data.file_id;
    state.fileType = data.file_type;
    sessionIdDisplay.textContent = state.sessionId.slice(0, 16) + '…';

    // Populate file card
    fileIcon.textContent = fileTypeIcon(data.file_type);
    fileName.textContent = file.name;
    fileMeta.textContent = formatMeta(data);
    setConfidence(data.confidence);

    if (data.extracted_text) {
      textPreview.textContent = data.extracted_text;
      textPreviewWrap.classList.remove('hidden');
    } else {
      textPreviewWrap.classList.add('hidden');
    }

    uploadProgress.classList.add('hidden');
    fileCard.classList.remove('hidden');

    // Prompt user
    setStatus(`${data.file_type.toUpperCase()} loaded — ask me anything`);
    appendAIMessage(
      `✅ Your **${data.file_type}** has been processed!\n\n` +
      `**${formatMeta(data)}**\n\n` +
      (data.warning ? `⚠️ ${data.warning}\n\n` : '') +
      `What would you like me to do with it?`
    );

    if (data.warning) showToast(`⚠️ ${data.warning}`, 'warning');
    showToast('File processed successfully!', 'success');
    userInput.focus();

  } catch (err) {
    clearInterval(ticker);
    uploadProgress.classList.add('hidden');
    capsSection.classList.remove('hidden');
    showToast(`Upload failed: ${err.message}`, 'error');
    setStatus('Upload failed — try again');
    console.error(err);
  } finally {
    state.isUploading = false;
    sendBtn.disabled = false;
    progressFill.style.width = '0%';
  }
}

// ── Chat flow ─────────────────────────────────────────────────────────────────

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || state.isSending) return;

  state.isSending = true;
  sendBtn.disabled = true;

  // Ensure session exists
  if (!state.sessionId) {
    state.sessionId = genId();
    sessionIdDisplay.textContent = state.sessionId.slice(0, 16) + '…';
  }

  appendUserMessage(text);
  userInput.value = '';
  autoResize(userInput);

  // Show typing indicator
  const typingId = appendTyping();
  setStatus('Thinking…');
  execPlan.classList.add('hidden');

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        message: text,
        file_id: state.fileId || null,
      }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      if (resp.status === 401 || resp.status === 503) {
        const msg = err.detail || 'API key error';
        removeTyping(typingId);
        appendAIMessage(
          `🔑 **API Key Required**\n\n${msg}\n\nGet your free key at **https://aistudio.google.com/app/apikey** and add it to your \`.env\` file as \`GEMINI_API_KEY=your_key_here\`, then restart the server.`,
          'error'
        );
        setStatus('API key error');
        showToast('Invalid API key — see chat for instructions', 'error', 6000);
        return;
      }
      if (resp.status === 429) {
        const msg = err.detail || 'Quota exhausted';
        removeTyping(typingId);
        appendAIMessage(
          `⚠️ **Quota Exhausted**\n\n${msg}`,
          'error'
        );
        setStatus('Quota limit hit — retry in ~1 min');
        showToast('API quota exhausted — see chat', 'warning', 6000);
        return;
      }
      throw new Error(err.detail || `Server error (${resp.status})`);
    }

    const data = await resp.json();
    removeTyping(typingId);

    // Render execution plan
    if (data.execution_plan?.length) {
      showExecPlan(data.execution_plan);
    }

    // Render response
    if (data.status === 'need_clarification' && data.follow_up_question) {
      appendClarification(data.follow_up_question, data.intent);
      setStatus('Needs clarification');
    } else if (data.result) {
      appendAIMessage(data.result, data.intent);
      setStatus(data.intent ? `Done — intent: ${data.intent}` : 'Done');
    } else {
      appendAIMessage('I processed your request but had no output to return.', 'error');
      setStatus('Completed with no output');
    }

    // Cost badge
    if (data.estimated_cost && data.estimated_cost > 0) {
      showToast(`💰 Est. cost: $${data.estimated_cost.toFixed(6)}`, 'info', 3000);
    }

  } catch (err) {
    removeTyping(typingId);
    appendAIMessage(`❌ Error: ${err.message}`, 'error');
    setStatus('Error — try again');
    showToast(err.message, 'error');
    console.error(err);
  } finally {
    state.isSending = false;
    sendBtn.disabled = false;
    userInput.focus();
  }
}

// ── Message renderers ─────────────────────────────────────────────────────────

function appendUserMessage(text) {
  const el = document.createElement('div');
  el.className = 'msg msg-user';
  el.innerHTML = `
    <div class="msg-avatar">👤</div>
    <div class="msg-content">
      <p class="msg-label">You</p>
      <div class="msg-bubble">${escHtml(text)}</div>
    </div>`;
  messages.appendChild(el);
  scrollBottom();
}

function appendAIMessage(text, intent) {
  const el = document.createElement('div');
  el.className = 'msg msg-ai';
  const formatted = formatAIText(text);
  el.innerHTML = `
    <div class="msg-avatar">⬡</div>
    <div class="msg-content">
      <p class="msg-label">NeuraMind${intent ? ` · ${intent}` : ''}</p>
      <div class="msg-bubble">${formatted}</div>
    </div>`;
  messages.appendChild(el);
  scrollBottom();
  return el;
}

function appendClarification(question, intent) {
  const el = document.createElement('div');
  el.className = 'msg msg-ai';
  el.innerHTML = `
    <div class="msg-avatar">⬡</div>
    <div class="msg-content">
      <p class="msg-label">NeuraMind</p>
      <div class="msg-bubble clarify">
        <span class="clarify-badge">NEEDS CLARIFICATION</span>
        <p>${escHtml(question)}</p>
      </div>
    </div>`;
  messages.appendChild(el);
  scrollBottom();
}

let typingCounter = 0;
function appendTyping() {
  const id = 'typing-' + (++typingCounter);
  const el = document.createElement('div');
  el.className = 'msg msg-ai';
  el.id = id;
  el.innerHTML = `
    <div class="msg-avatar">⬡</div>
    <div class="msg-content">
      <p class="msg-label">NeuraMind</p>
      <div class="msg-bubble typing-bubble">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>`;
  messages.appendChild(el);
  scrollBottom();
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function showExecPlan(steps) {
  planSteps.innerHTML = '';
  steps.forEach((step, i) => {
    const el = document.createElement('div');
    el.className = 'plan-step';
    el.style.animationDelay = `${i * 60}ms`;
    el.innerHTML = `<span class="plan-step-num">${i + 1}</span><span>${escHtml(step)}</span>`;
    planSteps.appendChild(el);
  });
  execPlan.classList.remove('hidden');
}

// ── Text formatting ───────────────────────────────────────────────────────────

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function formatAIText(text) {
  // Bold **text**
  let out = escHtml(text);
  out = out.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Bullet • or -
  out = out.replace(/^[•\-]\s+(.+)/gm, '<li>$1</li>');
  if (out.includes('<li>')) out = out.replace(/((<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');
  // Headers (ONE-LINE SUMMARY:, ANSWER:, etc.)
  out = out.replace(/^([A-Z][A-Z &]+):\s*/gm, '<strong style="color:var(--accent)">$1:</strong> ');
  // Line breaks
  out = out.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
  return out;
}

// ── Execution plan toggle ─────────────────────────────────────────────────────

planToggle.addEventListener('click', () => {
  const open = !planSteps.classList.contains('hidden');
  planSteps.classList.toggle('hidden', open);
  planChevron.classList.toggle('open', !open);
});

// ── Clear file ────────────────────────────────────────────────────────────────

clearFileBtn.addEventListener('click', () => {
  fileCard.classList.add('hidden');
  capsSection.classList.remove('hidden');
  state.fileId = null;
  state.fileType = null;
  setStatus('Ready — upload a file or start chatting');
  showToast('File cleared', 'info', 2000);
});

// ── New session ───────────────────────────────────────────────────────────────

newSessionBtn.addEventListener('click', () => {
  if (!confirm('Start a new session? Current conversation will be cleared.')) return;

  // Clear old session on server (fire-and-forget)
  if (state.sessionId) {
    fetch(`/api/chat/session/${state.sessionId}`, { method: 'DELETE' }).catch(() => {});
  }

  state.sessionId = null;
  state.fileId = null;
  state.fileType = null;
  sessionIdDisplay.textContent = '—';

  fileCard.classList.add('hidden');
  capsSection.classList.remove('hidden');
  execPlan.classList.add('hidden');
  messages.innerHTML = '';

  appendAIMessage('New session started! Upload a file or start chatting.');
  setStatus('Ready — new session');
});

// ── Sidebar toggle (mobile) ───────────────────────────────────────────────────

sidebarToggle.addEventListener('click', () => {
  leftPanel.classList.toggle('open');
});

// ── Drag & drop events ────────────────────────────────────────────────────────

dropZone.addEventListener('click', (e) => {
  // The <label for="fileInput"> already handles its own click natively.
  // If the click came from inside the label, don't also call fileInput.click()
  // or the OS file picker opens twice.
  if (e.target.closest('label')) return;
  fileInput.click();
});
fileInput.addEventListener('change', handleFileDrop);
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', handleFileDrop);

// Whole-page drop
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

// ── Input events ──────────────────────────────────────────────────────────────

userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
userInput.addEventListener('input', () => autoResize(userInput));
sendBtn.addEventListener('click', sendMessage);

// ── Utilities ─────────────────────────────────────────────────────────────────

function scrollBottom() {
  requestAnimationFrame(() => { messages.scrollTop = messages.scrollHeight; });
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Init ──────────────────────────────────────────────────────────────────────

userInput.focus();
