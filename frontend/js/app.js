const els = {
  conversationList: document.getElementById('conversation-list'),
  toggleSelectConversations: document.getElementById('toggle-select-conversations'),
  convSelectionBar: document.getElementById('conv-selection-bar'),
  convSelectedCount: document.getElementById('conv-selected-count'),
  selectAllConversations: document.getElementById('select-all-conversations'),
  clearSelectedConversations: document.getElementById('clear-selected-conversations'),
  deleteSelectedConversations: document.getElementById('delete-selected-conversations'),
  newChat: document.getElementById('new-chat'),
  skillList: document.getElementById('skill-list'),
  toolList: document.getElementById('tool-list'),
  toolsToggle: document.getElementById('tools-toggle'),
  thinkToggle: document.getElementById('think-toggle'),
  openToolHistory: document.getElementById('open-tool-history'),
  openMemory: document.getElementById('open-memory'),
  statusDot: document.getElementById('status-dot'),
  statusText: document.getElementById('status-text'),
  providerInfo: document.getElementById('provider-info'),
  contextInfo: document.getElementById('context-info'),
  gpuInfo: document.getElementById('gpu-info'),
  inferState: document.getElementById('infer-state'),
  openSettings: document.getElementById('open-settings'),
  toggleSidebar: document.getElementById('toggle-sidebar'),
  sidebar: document.getElementById('sidebar'),
  chatTitle: document.getElementById('chat-title'),
  messages: document.getElementById('messages'),
  composer: document.getElementById('composer'),
  input: document.getElementById('input'),
  send: document.getElementById('send'),
  settingsModal: document.getElementById('settings-modal'),
  settingModelField: document.getElementById('setting-model-field'),
  settingModel: document.getElementById('setting-model'),
  settingsTemperature: document.getElementById('setting-temperature'),
  settingsNumPredict: document.getElementById('setting-num-predict'),
  settingsSystem: document.getElementById('setting-system'),
  settingsSave: document.getElementById('settings-save'),
  settingsClose: document.getElementById('settings-close'),
  localapiEnabled: document.getElementById('localapi-enabled'),
  localapiPanel: document.getElementById('localapi-panel'),
  localapiEndpoint: document.getElementById('localapi-endpoint'),
  localapiCopyEndpoint: document.getElementById('localapi-copy-endpoint'),
  localapiModelLine: document.getElementById('localapi-model-line'),
  localapiKeys: document.getElementById('localapi-keys'),
  localapiCounter: document.getElementById('localapi-counter'),
  localapiNewName: document.getElementById('localapi-new-name'),
  localapiCreate: document.getElementById('localapi-create'),
  localapiReveal: document.getElementById('localapi-reveal'),
  localapiRevealKey: document.getElementById('localapi-reveal-key'),
  localapiCopyKey: document.getElementById('localapi-copy-key'),
  localapiExamples: document.getElementById('localapi-examples'),
  skillModal: document.getElementById('skill-modal'),
  skillTitle: document.getElementById('skill-title'),
  skillDescription: document.getElementById('skill-description'),
  skillInstructions: document.getElementById('skill-instructions'),
  skillActivate: document.getElementById('skill-activate'),
  toolModal: document.getElementById('tool-modal'),
  toolTitle: document.getElementById('tool-title'),
  toolDescription: document.getElementById('tool-description'),
  toolMode: document.getElementById('tool-mode'),
  toolArgs: document.getElementById('tool-args'),
  toolPermission: document.getElementById('tool-permission'),
  toolRemember: document.getElementById('tool-remember'),
  toolResult: document.getElementById('tool-result'),
  toolExecute: document.getElementById('tool-execute'),
  toolApprove: document.getElementById('tool-approve'),
  toolDeny: document.getElementById('tool-deny'),
  historyModal: document.getElementById('history-modal'),
  historyList: document.getElementById('history-list'),
  historyRefresh: document.getElementById('history-refresh'),
  memoryModal: document.getElementById('memory-modal'),
  memoryContextBox: document.getElementById('memory-context-box'),
  memoryRepackBtn: document.getElementById('memory-repack-btn'),
  memoryMdBtn: document.getElementById('memory-md-btn'),
  memoryRefreshBtn: document.getElementById('memory-refresh-btn'),
  memoryMdView: document.getElementById('memory-md-view'),
  memoryRepacks: document.getElementById('memory-repacks'),
  app: document.getElementById('app'),
  chat: document.getElementById('chat'),
  stop: document.getElementById('stop'),
  modelInfo: document.getElementById('model-info'),
  modelBadgeRow: document.getElementById('model-badge-row'),
  gpuRow: document.getElementById('gpu-row'),
  openAbout: document.getElementById('open-about'),
  aboutModal: document.getElementById('about-modal'),
  skillCount: document.getElementById('skill-count'),
  toolCount: document.getElementById('tool-count'),
  scrim: document.getElementById('scrim'),
  dialogModal: document.getElementById('dialog-modal'),
  dialogForm: document.getElementById('dialog-form'),
  dialogTitle: document.getElementById('dialog-title'),
  dialogMessage: document.getElementById('dialog-message'),
  dialogFields: document.getElementById('dialog-fields'),
  dialogOk: document.getElementById('dialog-ok'),
  toast: document.getElementById('toast'),
  attachmentStrip: document.getElementById('attachment-strip'),
  attachInput: document.getElementById('attach-input'),
  attachBtn: document.getElementById('attach-btn'),
  imageModal: document.getElementById('image-modal'),
  imageModalTitle: document.getElementById('image-modal-title'),
  imageModalImg: document.getElementById('image-modal-img'),
  imageModalMeta: document.getElementById('image-modal-meta'),
};

const state = {
  conversations: [],
  currentId: null,
  config: null,
  settings: { temperature: null, numPredict: null, systemPrompt: '' },
  streaming: false,
  controller: null,
  tools: [],
  toolsEnabled: true,
  skillActivations: {}, // conversationId -> [skill ids ativados]
  currentSkillId: null,
  currentToolId: null,
  pendingPermissionId: null,
  allowUnverified: false, // próxima mensagem aceita resposta só do modelo (rotulada)
  selectionMode: false,
  selectedIds: new Set(),
  deletingConversations: false,
  pendingAttachments: [],
  uploadingAttachments: 0,
  repackNoticeActive: false,
  overBudgetActive: false,
  autoScroll: true,
};

function isNearBottom(threshold = 80) {
  return els.messages.scrollHeight - els.messages.scrollTop - els.messages.clientHeight < threshold;
}

function scrollMessagesToBottom(force = false) {
  if (force || state.autoScroll) {
    els.messages.scrollTop = els.messages.scrollHeight;
  }
}

els.messages.addEventListener('scroll', () => {
  state.autoScroll = isNearBottom();
}, { passive: true });

function fallbackCopyText(value) {
  const ta = document.createElement('textarea');
  ta.value = value;
  ta.setAttribute('readonly', '');
  ta.style.cssText = 'position:fixed;top:-9999px;left:-9999px;opacity:0;';
  document.body.append(ta);
  const sel = document.getSelection();
  const prevRanges = [];
  if (sel) {
    for (let i = 0; i < sel.rangeCount; i += 1) prevRanges.push(sel.getRangeAt(i));
  }
  let ok = false;
  try {
    ta.select();
    ta.setSelectionRange(0, value.length);
    ok = Boolean(document.execCommand && document.execCommand('copy'));
  } catch (err) {
    ok = false;
  } finally {
    ta.remove();
    if (sel) {
      sel.removeAllRanges();
      for (const r of prevRanges) sel.addRange(r);
    }
  }
  return ok;
}

async function copyText(text) {
  const value = String(text ?? '');
  if (!value) throw new Error(tr('copy.nothing'));
  if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
    try {
      await navigator.clipboard.writeText(value);
      return;
    } catch (err) {
      if (fallbackCopyText(value)) return;
      throw err;
    }
  }
  if (fallbackCopyText(value)) return;
  throw new Error(tr('copy.unavailable'));
}

// Rótulos vêm de chaves i18n guardadas no dataset: trocar o idioma só exige refreshCopyButton.
function refreshCopyButton(btn) {
  const copied = btn.classList.contains('copied');
  const useEl = btn.querySelector('use');
  const labelEl = btn.querySelector('span');
  if (useEl) useEl.setAttribute('href', copied ? '#i-check' : '#i-copy');
  if (labelEl) labelEl.textContent = tr(copied ? 'copy.copied' : btn.dataset.labelKey);
  btn.setAttribute('aria-label', tr(copied ? 'copy.copiedAria' : btn.dataset.ariaKey));
  btn.title = btn.disabled ? tr('copy.waitTitle') : tr(btn.dataset.titleKey);
}

function setCopyButtonState(btn, copied) {
  btn.classList.toggle('copied', copied);
  refreshCopyButton(btn);
}

function createCopyButton({ labelKey = 'copy.label', ariaKey = 'copy.sentAria', titleKey = 'copy.sentTitle', getText }) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'copy-btn';
  btn.dataset.labelKey = labelKey;
  btn.dataset.ariaKey = ariaKey;
  btn.dataset.titleKey = titleKey;
  btn.innerHTML = '<svg class="ic" aria-hidden="true"><use href="#i-copy"/></svg><span></span>';
  refreshCopyButton(btn);
  let timer = null;
  btn.addEventListener('click', async () => {
    if (btn.disabled) return;
    try {
      await copyText(getText());
      clearTimeout(timer);
      setCopyButtonState(btn, true);
      timer = setTimeout(() => setCopyButtonState(btn, false), 1800);
    } catch (err) {
      setCopyButtonState(btn, false);
      showToast(tr('copy.failed'), 'error');
    }
  });
  return btn;
}

window.EnslavedCopy = { copyText, createCopyButton };

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

// ------------------------------------------------------------------ markdown (marked + DOMPurify em js/markdown.js)

// O texto bruto fica em dataset.raw (histórico intacto); só o DOM exibido é derivado dele.
function setContent(element, raw, partial = false) {
  element.dataset.raw = raw;
  element.replaceChildren(window.EnslavedMarkdown.render(raw, { partial }));
}

// Streaming: troca só os blocos que mudaram (os já concluídos, a seleção e a rolagem de código/tabelas ficam intactos).
function patchContent(element, raw) {
  element.dataset.raw = raw;
  const next = [...window.EnslavedMarkdown.render(raw, { partial: true }).childNodes];
  const current = [...element.childNodes];
  next.forEach((node, i) => {
    if (!current[i]) element.append(node);
    else if (!current[i].isEqualNode(node)) current[i].replaceWith(node);
  });
  current.slice(next.length).forEach((node) => node.remove());
}

const RENDER_INTERVAL_MS = 50; // no máximo ~20 renderizações/s: Markdown completo a cada token seria custo inútil

// Streaming: acumula o texto bruto (`replace` troca o que já havia) e renderiza com intervalo mínimo.
function appendContent(element, chunk, replace = false) {
  element.dataset.raw = replace ? chunk : (element.dataset.raw || '') + chunk;
  if (element._raf) return;
  const wait = Math.max(0, RENDER_INTERVAL_MS - (performance.now() - (element._last || 0)));
  element._raf = setTimeout(() => {
    element._raf = 0;
    element._last = performance.now();
    const pinned = state.autoScroll && isNearBottom();
    patchContent(element, element.dataset.raw);
    if (pinned) els.messages.scrollTop = els.messages.scrollHeight;
  }, wait);
}

// Renderização final completa (cancela a pendente do streaming).
function finalizeContent(element) {
  if (element._raf) { clearTimeout(element._raf); element._raf = 0; }
  if (element.dataset.raw) setContent(element, element.dataset.raw);
}

// Citações [web#N] do texto viram link para a fonte correspondente (só ids que existem; nunca cria link).
function linkifyWebCitations(contentEl, sources) {
  const byId = new Map(
    (sources || [])
      .filter((s) => s && s.id && /^https?:\/\//i.test(s.url || ''))
      .map((s) => [s.id, s]),
  );
  if (!byId.size) return;
  const walker = document.createTreeWalker(contentEl, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) => (n.parentElement && n.parentElement.closest('a, code, pre') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const textNode of nodes) {
    const text = textNode.nodeValue;
    if (!/\[web#\d+\]/.test(text)) continue;
    const frag = document.createDocumentFragment();
    let last = 0;
    let linked = false;
    for (const m of text.matchAll(/\[(web#\d+)\]/g)) {
      const src = byId.get(m[1]);
      if (!src) continue;
      const a = document.createElement('a');
      a.href = src.url;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.title = src.title || src.url;
      a.textContent = m[0];
      frag.append(text.slice(last, m.index), a);
      last = m.index + m[0].length;
      linked = true;
    }
    if (!linked) continue;
    frag.append(text.slice(last));
    textNode.replaceWith(frag);
  }
}

// Insere o nó antes do rodapé de ações, que fica sempre por último na mensagem do assistente.
function addToBubble(bubble, node) {
  bubble.insertBefore(node, bubble.querySelector(':scope > .msg-footer'));
}

// Resultado da verificação factual: fontes realmente consultadas ou o motivo de não ter verificado.
function verificationNote(v, question) {
  if (!v || v.status === 'skipped') return null;
  const box = document.createElement('div');
  box.className = `verify-note ${v.status}`;
  const known = ['verified', 'indication', 'local', 'unverified', 'no_evidence'];
  box.append(`${known.includes(v.status) ? tr(`verify.${v.status}`) : v.status}${v.message ? ` — ${tm(v.message)}` : ''}`);
  for (const src of v.sources || []) {
    if (!src.url) continue;
    const link = document.createElement('a');
    link.href = src.url; link.target = '_blank'; link.rel = 'noopener noreferrer';
    link.textContent = `${src.id ? `[${src.id}] ` : ''}${src.title || src.url}${src.level === 'snippet' ? tr('verify.snippet') : ''}`;
    box.append(' · ', link);
  }
  if (v.offer_unverified && question) {
    const ask = document.createElement('button');
    ask.type = 'button';
    ask.className = 'btn ghost verify-offer';
    ask.textContent = tr('verify.offer');
    ask.addEventListener('click', () => {
      if (state.streaming) return;
      ask.remove();
      state.allowUnverified = true;
      els.input.value = question;
      els.composer.requestSubmit();
    });
    box.append(document.createElement('br'), ask);
  }
  return box;
}

// Ícone "E" oficial (assets/brand); o brilho de "pensando/respondendo" é CSS.
function logoMark(cls) {
  return `<img class="${cls}" src="/static/assets/brand/logo-e-64.webp" srcset="/static/assets/brand/logo-e-64.webp 1x, /static/assets/brand/logo-e-128.webp 2x" width="24" height="24" alt="" aria-hidden="true">`;
}

// Indicador "pensando": aparece entre o envio e o primeiro token.
function createThinking() {
  const article = document.createElement('article');
  article.className = 'message assistant thinking';
  article.dataset.i18nAttr = 'aria-label:chat.thinkingAria';
  article.setAttribute('aria-label', tr('chat.thinkingAria'));
  article.innerHTML = `<header>${logoMark('avatar')}<span class="role">Enslaved</span></header><div class="content"><span class="think-text" data-i18n="chat.thinking">${tr('chat.thinking')}</span><span class="think-time"></span></div>`;
  const started = Date.now();
  const time = article.querySelector('.think-time');
  article._timer = setInterval(() => {
    const seconds = Math.round((Date.now() - started) / 1000);
    time.textContent = seconds >= 4 ? ` · ${seconds}s` : '';
  }, 1000);
  return article;
}

// Raciocínio do modelo, ao vivo: bloco recolhível que nasce no indicador "Pensando"
// e vai para dentro da resposta quando o primeiro token chega.
function appendReasoning(thinking, text) {
  let box = thinking._reasoning;
  if (!box) {
    box = document.createElement('details');
    box.className = 'reasoning live';
    box.open = true;
    box.innerHTML = `<summary data-i18n="chat.reasoning">${tr('chat.reasoning')}</summary><div class="reasoning-text"></div>`;
    box._started = Date.now();
    thinking._reasoning = box;
    thinking.append(box);
  }
  const body = box.querySelector('.reasoning-text');
  body.textContent += text;
  body.scrollTop = body.scrollHeight;
}

function settleReasoning(thinking, bubble) {
  const box = thinking._reasoning;
  if (!box) return;
  const seconds = Math.max(1, Math.round((Date.now() - box._started) / 1000));
  box.classList.remove('live');
  box.open = false;
  i18n.bind(box.querySelector('summary'), 'chat.reasoningTime', { seconds });
  bubble.insertBefore(box, bubble.querySelector('.content'));
}

function removeThinking(article) {
  if (!article) return;
  clearInterval(article._timer);
  article.remove();
}

function attachmentUrl(att, conversationId) {
  if (conversationId) {
    return `/api/conversations/${encodeURIComponent(conversationId)}/attachments/${encodeURIComponent(att.id)}`;
  }
  return `/api/attachments/${encodeURIComponent(att.id)}`;
}

function openImageModal(src, title, metaText = '') {
  els.imageModalTitle.textContent = title || tr('common.image');
  els.imageModalImg.src = src;
  els.imageModalImg.alt = title || tr('chat.imageAlt');
  els.imageModalMeta.textContent = metaText || '';
  els.imageModal.classList.remove('hidden');
}

function maxImagesPerMessage() {
  return state.config?.vision?.max_images || 4;
}

function renderPendingAttachments() {
  els.attachmentStrip.innerHTML = '';
  els.attachmentStrip.hidden = state.pendingAttachments.length === 0;
  for (const att of state.pendingAttachments) {
    const card = document.createElement('div');
    card.className = 'att-preview';
    const src = attachmentUrl(att, null);
    const img = document.createElement('img');
    img.src = src;
    img.alt = att.name || tr('chat.preview');
    const dim = att.width && att.height ? `${att.width}×${att.height}` : '';
    img.addEventListener('click', () => openImageModal(src, att.name, dim));

    const info = document.createElement('div');
    info.className = 'att-preview-info';
    const nameEl = document.createElement('span');
    nameEl.className = 'att-preview-name';
    nameEl.textContent = att.name || tr('chat.imageDefault');
    const dimEl = document.createElement('span');
    dimEl.className = 'att-preview-dim';
    dimEl.textContent = dim;
    info.append(nameEl, dimEl);

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'att-remove';
    i18n.bindAttr(remove, 'aria-label', 'chat.removeAtt', { name: att.name || tr('chat.imageDefault') });
    i18n.bindAttr(remove, 'title', 'chat.removeAttTitle');
    remove.textContent = '×';
    remove.addEventListener('click', () => removePendingAttachment(att.id));

    card.append(img, info, remove);
    els.attachmentStrip.append(card);
  }
}

async function removePendingAttachment(id) {
  state.pendingAttachments = state.pendingAttachments.filter((a) => a.id !== id);
  renderPendingAttachments();
  try {
    await api.del(`/api/attachments/${encodeURIComponent(id)}`);
  } catch (err) { /* já removido ou expirado */ }
}

function clearPendingAttachments() {
  const stale = [...state.pendingAttachments];
  state.pendingAttachments = [];
  renderPendingAttachments();
  for (const att of stale) {
    api.del(`/api/attachments/${encodeURIComponent(att.id)}`).catch(() => {});
  }
}

async function attachFiles(fileList) {
  const files = [...(fileList || [])].filter(Boolean);
  if (!files.length) return;
  const limit = maxImagesPerMessage();
  for (const file of files) {
    if (state.pendingAttachments.length + state.uploadingAttachments >= limit) {
      showToast(tr('chat.maxImages', { limit }), 'error');
      break;
    }
    state.uploadingAttachments += 1;
    els.attachBtn.disabled = true;
    try {
      const res = await api.uploadAttachment(file);
      if (res.attachment) {
        state.pendingAttachments.push(res.attachment);
        renderPendingAttachments();
      }
    } catch (err) {
      showToast(tm(err.message) || tr('chat.attachFailed'), 'error');
    } finally {
      state.uploadingAttachments = Math.max(0, state.uploadingAttachments - 1);
      els.attachBtn.disabled = state.uploadingAttachments > 0;
    }
  }
}

function renderMessageAttachments(atts, conversationId) {
  const wrap = document.createElement('div');
  wrap.className = 'message-attachments';
  for (const att of atts) {
    const src = attachmentUrl(att, conversationId);
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'msg-att-btn';
    btn.dataset.attId = att.id;
    i18n.bindAttr(btn, 'title', 'chat.enlarge', { name: att.name || tr('chat.imageDefault') });
    const img = document.createElement('img');
    img.className = 'msg-att-thumb';
    img.src = src;
    img.alt = att.name || tr('chat.imageAlt');
    img.loading = 'lazy';
    const dim = att.width && att.height ? `${att.width}×${att.height}` : '';
    btn.addEventListener('click', () => openImageModal(btn.querySelector('img').src, att.name, dim));
    btn.append(img);
    wrap.append(btn);
  }
  return wrap;
}

function setCopyReady(bubble, ready, partial = false) {
  const btn = bubble?.querySelector(':scope > header > .copy-btn, :scope > .msg-actions > .copy-btn');
  if (!btn) return;
  btn.disabled = !ready;
  if (ready && partial) { // o rótulo não apresenta um relatório incompleto como se fosse completo
    btn.dataset.labelKey = 'copy.partial';
    btn.dataset.titleKey = 'copy.partialTitle';
  }
  refreshCopyButton(btn);
}

function messageBubble(role, content, meta, conversationId = state.currentId) {
  const article = document.createElement('article');
  article.className = `message ${role}`;
  const header = document.createElement('header');
  const body = document.createElement('div');
  body.className = 'content';
  const raw = String(content ?? '');
  let footer = null;
  if (role === 'assistant') {
    header.innerHTML = `${logoMark('avatar')}<span class="role">Enslaved</span>`;
    const copyBtn = createCopyButton({
      labelKey: 'copy.label',
      ariaKey: 'copy.replyAria',
      titleKey: 'copy.replyTitle',
      getText: () => body.dataset.raw || '',
    });
    if (meta && meta.research && meta.research.completion && meta.research.completion.status === 'partial') {
      copyBtn.dataset.titleKey = 'copy.partialTitle';
      refreshCopyButton(copyBtn);
    }
    footer = document.createElement('div');
    footer.className = 'msg-actions msg-footer';
    footer.append(copyBtn);
    setContent(body, raw);
    if (meta && Array.isArray(meta.verification && meta.verification.sources)) {
      linkifyWebCitations(body, meta.verification.sources); // citações do histórico salvo também viram links
    }
  } else {
    body.dataset.raw = raw;
    body.textContent = raw;
  }
  const atts = role === 'user' && meta && Array.isArray(meta.attachments) && meta.attachments.length
    ? renderMessageAttachments(meta.attachments, conversationId)
    : null;
  if (atts) article.append(header, atts, body);
  else article.append(header, body);
  if (role === 'user') {
    const actions = document.createElement('div');
    actions.className = 'msg-actions';
    actions.append(createCopyButton({
      labelKey: 'copy.label',
      ariaKey: 'copy.sentAria',
      titleKey: 'copy.sentTitle',
      getText: () => body.dataset.raw || '',
    }));
    article.append(actions);
  }
  if (footer) article.append(footer); // sempre por último: depois do texto, das notas e do painel de evidências
  const note = role === 'assistant' && meta ? verificationNote(meta.verification) : null;
  if (note) addToBubble(article, note);
  if (role === 'assistant' && meta && meta.research && meta.research.view) {
    window.EnslavedResearch.attach(article, meta.research.view).link(); // painel de evidências também no histórico salvo
  }
  return article;
}

function errorBubble(text) {
  const article = document.createElement('article');
  article.className = 'message error';
  article.setAttribute('role', 'alert');
  article.textContent = tr('common.error', { text: tm(String(text)) });
  return article;
}

function syncEmptyState() {
  els.chat.classList.toggle('empty', !state.currentId && !els.messages.children.length);
}

let toastTimer = null;
function showToast(text, kind) {
  els.toast.textContent = text;
  els.toast.className = `toast${kind ? ` ${kind}` : ''}`;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { els.toast.hidden = true; }, 4500);
}

// ------------------------------------------------------------------ modais

function closeModal(modal) {
  modal.classList.add('hidden');
  modal.dispatchEvent(new Event('dismiss'));
}

function openModals() {
  return [...document.querySelectorAll('.modal:not(.hidden)')];
}

document.addEventListener('click', (event) => {
  const modal = event.target.closest('.modal');
  if (!modal) return;
  if (event.target === modal || event.target.closest('[data-close]')) closeModal(modal);
});
document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  const modals = openModals();
  if (modals.length) closeModal(modals[modals.length - 1]);
  else if (els.sidebar.classList.contains('open')) setSidebarOpen(false);
});

// Diálogo próprio no lugar de prompt()/confirm(). Resolve com os valores (ou true) e null ao cancelar.
function askDialog({ title, message = '', fields = [], okLabel = tr('common.confirm'), danger = false }) {
  return new Promise((resolve) => {
    els.dialogTitle.textContent = title;
    els.dialogMessage.textContent = message;
    els.dialogFields.innerHTML = '';
    els.dialogOk.textContent = okLabel;
    els.dialogOk.className = `btn ${danger ? 'danger' : 'primary'}`;
    const inputs = {};
    for (const field of fields) {
      const label = document.createElement('label');
      label.className = 'field';
      label.textContent = field.label;
      let input;
      if (field.options) {
        input = document.createElement('select');
        for (const [value, text] of field.options) input.add(new Option(text, value));
      } else {
        input = document.createElement('textarea');
        input.rows = field.rows || 3;
      }
      input.value = field.value || '';
      inputs[field.name] = input;
      label.append(input);
      els.dialogFields.append(label);
    }
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      els.dialogForm.removeEventListener('submit', onSubmit);
      els.dialogModal.removeEventListener('dismiss', onDismiss);
      resolve(value);
    };
    const onSubmit = (event) => {
      event.preventDefault();
      const values = Object.fromEntries(Object.entries(inputs).map(([name, input]) => [name, input.value]));
      els.dialogModal.classList.add('hidden');
      finish(fields.length ? values : true);
    };
    const onDismiss = () => finish(null);
    els.dialogForm.addEventListener('submit', onSubmit);
    els.dialogModal.addEventListener('dismiss', onDismiss);
    els.dialogModal.classList.remove('hidden');
    (Object.values(inputs)[0] || els.dialogOk).focus();
  });
}

function buildDeleteConfirmationMessage(count) {
  return tr('del.message', { target: count === 1 ? tr('del.targetOne') : tr('del.targetMany', { count }) });
}

function syncSelectionControls() {
  const visibleIds = new Set(state.conversations.map((c) => c.id));
  for (const id of [...state.selectedIds]) {
    if (!visibleIds.has(id)) state.selectedIds.delete(id);
  }
  const total = state.conversations.length;
  if (!total) state.selectionMode = false;
  els.toggleSelectConversations.hidden = !total;
  els.toggleSelectConversations.setAttribute('aria-pressed', String(state.selectionMode));
  els.toggleSelectConversations.textContent = state.selectionMode ? tr('sel.done') : tr('sel.toggle');
  delete els.toggleSelectConversations.dataset.i18n;
  els.convSelectionBar.hidden = !(state.selectionMode && total);
  const selectedCount = state.selectedIds.size;
  i18n.bind(els.convSelectedCount, 'sel.count', { count: selectedCount });
  els.selectAllConversations.disabled = selectedCount === total;
  els.clearSelectedConversations.disabled = selectedCount === 0;
  els.deleteSelectedConversations.disabled = selectedCount === 0 || state.deletingConversations;
}

function toggleConversationSelected(conversationId, checked) {
  if (checked === undefined) {
    if (state.selectedIds.has(conversationId)) state.selectedIds.delete(conversationId);
    else state.selectedIds.add(conversationId);
  } else if (checked) {
    state.selectedIds.add(conversationId);
  } else {
    state.selectedIds.delete(conversationId);
  }
  renderConversationList();
}

async function confirmAndDeleteConversations(ids, { fromBulk = false } = {}) {
  const uniqueIds = [...new Set(ids)].filter(Boolean);
  if (!uniqueIds.length || state.deletingConversations) return;
  const ok = await askDialog({
    title: uniqueIds.length === 1 ? tr('del.titleOne') : tr('del.titleMany'),
    message: buildDeleteConfirmationMessage(uniqueIds.length),
    okLabel: tr('del.ok'),
    danger: true,
  });
  if (!ok) return;

  state.deletingConversations = true;
  syncSelectionControls();
  const includesCurrent = Boolean(state.currentId && uniqueIds.includes(state.currentId));
  if (includesCurrent && state.streaming && state.controller) {
    state.controller.abort();
  }

  let deletedCount = 0;
  try {
    if (!fromBulk && uniqueIds.length === 1) {
      try {
        await api.del(`/api/conversations/${encodeURIComponent(uniqueIds[0])}`);
        deletedCount = 1;
      } catch (err) {
        if (err.status !== 404) throw err;
      }
    } else {
      const res = await api.post('/api/conversations/delete', { ids: uniqueIds });
      deletedCount = Number(res.deleted_count || 0);
    }

    for (const id of uniqueIds) {
      state.selectedIds.delete(id);
      delete state.skillActivations[id];
    }
    if (fromBulk) state.selectionMode = false;
    if (includesCurrent) newChat();
    await loadConversations();
    loadMemoryStatus().catch((err) => console.error(err));
    if (deletedCount > 0) {
      showToast(deletedCount === 1 ? tr('del.doneOne') : tr('del.doneMany', { count: deletedCount }));
    } else {
      showToast(tr('del.none'));
    }
  } catch (err) {
    showToast(tm(err.message) || tr('del.failed'), 'error');
  } finally {
    state.deletingConversations = false;
    syncSelectionControls();
  }
}

function renderConversationList() {
  syncSelectionControls();
  els.conversationList.innerHTML = '';
  if (!state.conversations.length) {
    const empty = document.createElement('div');
    empty.className = 'list-empty';
    empty.textContent = tr('chat.empty');
    els.conversationList.append(empty);
    return;
  }
  for (const conversation of state.conversations) {
    const isSelected = state.selectionMode && state.selectedIds.has(conversation.id);
    const item = document.createElement('div');
    item.className = 'conversation-item'
      + (conversation.id === state.currentId ? ' active' : '')
      + (state.selectionMode ? ' selecting' : '')
      + (isSelected ? ' selected' : '');
    item.dataset.conversationId = conversation.id;

    if (state.selectionMode) {
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.className = 'conv-check';
      checkbox.checked = isSelected;
      checkbox.setAttribute('aria-label', tr('chat.selectConversation', { title: conversation.title }));
      checkbox.addEventListener('change', () => toggleConversationSelected(conversation.id, checkbox.checked));
      item.append(checkbox);
    }

    const title = document.createElement('button');
    title.type = 'button';
    title.className = 'title';
    title.textContent = conversation.title;
    title.title = conversation.preview || conversation.title;
    if (conversation.id === state.currentId) title.setAttribute('aria-current', 'true');
    title.addEventListener('click', () => {
      if (state.selectionMode) toggleConversationSelected(conversation.id);
      else openConversation(conversation.id);
    });
    item.append(title);

    if (!state.selectionMode) {
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'delete';
      remove.setAttribute('aria-label', tr('chat.deleteConversation', { title: conversation.title }));
      remove.title = tr('chat.deleteConversationTitle');
      remove.innerHTML = '<svg class="ic" aria-hidden="true"><use href="#i-trash"/></svg>';
      remove.addEventListener('click', (event) => {
        event.stopPropagation();
        confirmAndDeleteConversations([conversation.id], { fromBulk: false });
      });
      item.append(remove);
    }

    els.conversationList.append(item);
  }
}

function setSidebarOpen(open) {
  els.sidebar.classList.toggle('open', open);
  els.scrim.hidden = !open;
}

function toggleSidebar() {
  if (window.matchMedia('(max-width: 820px)').matches) setSidebarOpen(!els.sidebar.classList.contains('open'));
  else els.app.classList.toggle('collapsed');
}

async function loadConversations() {
  const data = await api.get('/api/conversations');
  state.conversations = data.conversations;
  renderConversationList();
}

async function openConversation(id, { keepAttachments = false } = {}) {
  try {
    const data = await api.get(`/api/conversations/${id}`);
    state.currentId = id;
    window.EnslavedScene?.reset();
    els.chatTitle.textContent = data.conversation.title;
    els.messages.innerHTML = '';
    for (const message of data.messages) {
      if (message.role === 'system') continue;
      if (message.role === 'tool') {
        let ok = null;
        let preview = '';
        try {
          const parsed = JSON.parse((message.content || '').split('\n').slice(1, -1).join('\n'));
          ok = parsed.ok !== false;
          preview = typeof parsed.output === 'string' ? parsed.output : JSON.stringify(parsed.output);
        } catch (err) {
          preview = message.content;
        }
        if (ok === false) preview = tr('tool.error', { text: tm(message.meta && message.meta.error) });
        els.messages.append(toolEventCard(
          ok === false ? 'fail' : 'ok',
          tr('tool.resultPrefix', { name: message.meta && message.meta.tool_name }),
          String(preview).slice(0, 400),
        ));
        continue;
      }
      if (message.role === 'assistant' && message.meta && message.meta.tool_calls) {
        const names = message.meta.tool_calls.map((c) => c.name).join(', ');
        els.messages.append(toolEventCard('call', tr('tool.requested'), names));
        continue;
      }
      els.messages.append(messageBubble(message.role, message.content, message.meta, id));
    }
    state.repackNoticeActive = false;
    if (!keepAttachments) clearPendingAttachments();
    renderConversationList();
    renderSkillList();
    loadMemoryStatus();
    syncEmptyState();
    state.autoScroll = true;
    scrollMessagesToBottom(true);
    setSidebarOpen(false);
  } catch (err) {
    if (err.status === 404) {
      newChat();
      await loadConversations();
    } else {
      els.messages.append(errorBubble(err.message));
    }
  }
}

function newChat() {
  state.currentId = null;
  state.repackNoticeActive = false;
  clearPendingAttachments();
  window.EnslavedScene?.reset();
  els.chatTitle.textContent = tr('chat.new');
  els.messages.innerHTML = '';
  renderConversationList();
  loadMemoryStatus();
  syncEmptyState();
  setSidebarOpen(false);
  els.input.focus();
}

function activeSkillSet() {
  return state.skillActivations[state.currentId || 'nova'] || [];
}

function activateSkillLocally(skillId, active) {
  const key = state.currentId || 'nova';
  const current = new Set(state.skillActivations[key] || []);
  if (active) current.add(skillId); else current.delete(skillId);
  state.skillActivations[key] = [...current];
  renderSkillList();
}

async function loadConfig() {
  const config = await api.get('/api/config');
  state.config = config;
  els.providerInfo.textContent = config.provider.name;
  els.modelInfo.textContent = config.provider.model;
  els.modelBadgeRow.hidden = !config.provider.experimental;
  els.settingsTemperature.value = config.defaults.temperature;
  els.settingsNumPredict.value = config.defaults.num_predict;
  state.settings.temperature = String(config.defaults.temperature);
  state.settings.numPredict = String(config.defaults.num_predict);
  const models = config.provider.models || [];
  if (models.length > 1) {
    els.settingModelField.hidden = false;
    els.settingModel.innerHTML = '';
    for (const entry of models) {
      const opt = document.createElement('option');
      opt.value = entry.id;
      opt.textContent = entry.experimental ? `${entry.id} · ${tr('about.experimental')}` : entry.id;
      if (entry.id === config.provider.model) opt.selected = true;
      els.settingModel.appendChild(opt);
    }
    state.settings.savedModel = config.provider.model;
  }
  if (config.vision && (config.vision.enabled === false || config.provider.supports_vision === false)) {
    els.attachBtn.hidden = true;
  }
  if (config.provider.supports_tools === false) {
    // Modelo sem tool calling nativo: sem botão de Tools (nada que o modelo não possa cumprir)
    state.toolsEnabled = false;
    els.toolsToggle.checked = false;
    els.toolsToggle.disabled = true;
    const chip = els.toolsToggle.closest('label');
    if (chip) chip.hidden = true;
  }
}

async function loadHealth() {
  try {
    const health = await api.get('/api/health');
    const provider = health.provider;
    const model = state.config && state.config.provider ? state.config.provider.model : null;
    if (!provider.available) {
      els.statusDot.className = 'dot err';
      els.statusText.textContent = tr('status.unavailable', { name: provider.name });
    } else if (model && !(provider.models || []).includes(model)) {
      els.statusDot.className = 'dot wait';
      els.statusText.textContent = tr('status.notInstalled', { model });
      els.statusText.title = tr('status.install', { model });
    } else {
      els.statusDot.className = 'dot ok';
      els.statusText.textContent = `${tr('status.connected', { name: provider.name })}${model ? ` · ${model}` : ''}`;
      els.statusText.title = '';
    }
  } catch (err) {
    els.statusDot.className = 'dot err';
    els.statusText.textContent = tr('status.backendDown');
    els.statusText.title = err.message;
  }
}

function formatGb(mb) {
  return i18n.formatNumber(mb / 1024, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
}

async function loadDiagnostics() {
  try {
    const data = await api.get('/api/diagnostics');
    const gpu = data.gpus && data.gpus[0];
    if (!gpu || gpu.memory_total_mb === null) {
      els.gpuRow.hidden = true;
      return;
    }
    let text = tr('about.vram', { name: gpu.name, used: formatGb(gpu.memory_used_mb), total: formatGb(gpu.memory_total_mb) });
    if (gpu.utilization_gpu_percent !== null && gpu.utilization_gpu_percent !== undefined) {
      text += ` · ${gpu.utilization_gpu_percent}%`;
    }
    const loaded = (data.loaded_models || []).find((m) => m.vram_mb > 0);
    if (loaded) text += ` · ${tr('about.modelOnGpu', { gb: formatGb(loaded.vram_mb) })}`;
    els.gpuInfo.textContent = text;
    els.gpuRow.hidden = false;
  } catch (err) {
    els.gpuRow.hidden = true;
  }
}

let skillCatalog = [];

function renderSkillList() {
  els.skillList.innerHTML = '';
  const activated = new Set(activeSkillSet());
  for (const skill of skillCatalog.skills || []) {
    const item = document.createElement('li');
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = skill.name;
    item.append(name);
    const flags = [];
    if (!skill.enabled) flags.push(tr('skills.disabled'));
    if (!skill.available) flags.push(tr('skills.incompatible'));
    if (activated.has(skill.id)) item.insertAdjacentHTML('beforeend', `<span class="tag on">${escapeHtml(tr('skills.active'))}</span>`);
    else if (flags.length) item.insertAdjacentHTML('beforeend', `<span class="tag">${escapeHtml(flags.join(' · '))}</span>`);
    item.title = skill.description;
    item.tabIndex = 0;
    item.setAttribute('role', 'button');
    if (!skill.enabled || !skill.available) item.classList.add('skill-disabled');
    item.addEventListener('click', () => openSkill(skill.id));
    item.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        openSkill(skill.id);
      }
    });
    els.skillList.append(item);
  }
  if (skillCatalog.errors && skillCatalog.errors.length) {
    const warn = document.createElement('li');
    warn.className = 'skill-error';
    warn.textContent = tr('skills.invalid', { count: skillCatalog.errors.length });
    warn.title = skillCatalog.errors.map((e) => `${e.path}: ${e.reason}`).join('\n');
    els.skillList.append(warn);
  }
  if (!(skillCatalog.skills || []).length && !(skillCatalog.errors || []).length) {
    const empty = document.createElement('li');
    empty.className = 'skill-error';
    empty.textContent = tr('skills.none');
    els.skillList.append(empty);
  }
  const total = (skillCatalog.skills || []).length;
  const active = (skillCatalog.skills || []).filter((s) => activated.has(s.id)).length;
  els.skillCount.textContent = total ? (active ? tr('skills.activeCount', { count: active, total }) : String(total)) : '';
}

async function loadSkills() {
  skillCatalog = await api.get('/api/skills');
  renderSkillList();
}

async function openSkill(skillId) {
  try {
    const skill = await api.get(`/api/skills/${skillId}`);
    state.currentSkillId = skillId;
    els.skillTitle.textContent = skill.name;
    els.skillDescription.textContent = skill.description;
    const flags = [
      skill.has_scripts && tr('skills.hasScripts'),
      skill.has_references && tr('skills.hasReferences'),
      skill.enabled === false && tr('skills.disabled'),
      skill.available === false && tr('skills.missing', { list: skill.missing_requirements.join(', ') }),
    ].filter(Boolean).join(' · ');
    els.skillInstructions.textContent = flags
      ? `${skill.instructions}\n\n[${flags}]`
      : skill.instructions;
    const activated = activeSkillSet().includes(skillId);
    i18n.bind(els.skillActivate, activated ? 'skill.deactivate' : 'skill.activate');
    els.skillActivate.dataset.active = activated ? '1' : '';
    els.skillActivate.disabled = !skill.available && !activated;
    els.skillModal.classList.remove('hidden');
  } catch (err) {
    console.error(err);
  }
}

// ------------------------------------------------------------------ ferramentas

function modeBadge(mode) {
  const badge = document.createElement('span');
  badge.className = `badge mode-${mode}`;
  i18n.bind(badge, mode === 'allow' ? 'tools.modeAllow' : mode === 'ask' ? 'tools.modeAsk' : 'tools.modeDeny');
  return badge;
}

function renderToolList() {
  els.toolList.innerHTML = '';
  for (const tool of state.tools) {
    const item = document.createElement('li');
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = tm(tool.name);
    item.append(name, modeBadge(tool.mode));
    item.title = `${tm(tool.description)}\n${tr('tools.permission', { list: tool.capabilities.join(', ') || '—' })}`;
    item.tabIndex = 0;
    item.setAttribute('role', 'button');
    item.addEventListener('click', () => openTool(tool.id));
    item.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        openTool(tool.id);
      }
    });
    els.toolList.append(item);
  }
  if (!state.tools.length) {
    const empty = document.createElement('li');
    empty.className = 'skill-error';
    empty.textContent = tr('tools.none');
    els.toolList.append(empty);
  }
  els.toolCount.textContent = state.tools.length ? String(state.tools.length) : '';
}

async function loadTools() {
  try {
    const data = await api.get('/api/tools');
    state.tools = data.tools || [];
    state.toolsEnabled = data.enabled !== false;
    if (!els.toolsToggle.disabled) els.toolsToggle.checked = state.toolsEnabled;
    renderToolList();
  } catch (err) {
    console.error(err);
  }
}

function openTool(toolId) {
  const tool = state.tools.find((t) => t.id === toolId);
  if (!tool) return;
  state.currentToolId = toolId;
  state.pendingPermissionId = null;
  els.toolTitle.textContent = tm(tool.name);
  els.toolTitle.removeAttribute('data-i18n');
  els.toolDescription.textContent = tm(tool.description);
  els.toolMode.innerHTML = '';
  els.toolMode.append(modeBadge(tool.mode));
  const defaults = Object.fromEntries(
    Object.entries(tool.parameters.properties || {}).map(([key, schema]) => [key, schema.default !== undefined ? schema.default : schema.type === 'integer' ? 0 : ''])
  );
  els.toolArgs.value = Object.keys(defaults).length ? JSON.stringify(defaults, null, 2) : '{}';
  els.toolResult.classList.add('hidden');
  els.toolResult.textContent = '';
  els.toolPermission.classList.add('hidden');
  els.toolExecute.classList.remove('hidden');
  els.toolApprove.classList.add('hidden');
  els.toolDeny.classList.add('hidden');
  els.toolModal.classList.remove('hidden');
}

function showToolResult(result) {
  els.toolResult.classList.remove('hidden');
  els.toolResult.textContent = JSON.stringify(result, null, 2);
}

async function executeCurrentTool() {
  if (!state.currentToolId) return;
  let args;
  try {
    args = els.toolArgs.value.trim() ? JSON.parse(els.toolArgs.value) : {};
  } catch (err) {
    showToolResult({ error: tr('tool.invalidJson', { text: err.message }) });
    return;
  }
  els.toolExecute.disabled = true;
  try {
    const data = await api.post(`/api/tools/${state.currentToolId}/invoke`, { arguments: args });
    if (data.status === 'permission_required') {
      state.pendingPermissionId = data.request.id;
      els.toolPermission.classList.remove('hidden');
      els.toolApprove.classList.remove('hidden');
      els.toolDeny.classList.remove('hidden');
      els.toolExecute.classList.add('hidden');
      showToolResult({ status: tr('tool.awaiting'), [tr('tool.requestKey')]: data.request });
    } else {
      showToolResult(data.status === 'executed' ? data.result : data);
    }
  } catch (err) {
    showToolResult({ error: tm(err.message) });
  } finally {
    els.toolExecute.disabled = false;
  }
}

async function decideCurrentTool(granted) {
  if (!state.pendingPermissionId) return;
  const remember = els.toolRemember.checked;
  els.toolApprove.disabled = true;
  els.toolDeny.disabled = true;
  try {
    const data = await api.post(`/api/permissions/${state.pendingPermissionId}/decision`, {
      granted,
      remember_session: remember,
      expected_source: 'manual',
    });
    state.pendingPermissionId = null;
    els.toolPermission.classList.add('hidden');
    els.toolApprove.classList.add('hidden');
    els.toolDeny.classList.add('hidden');
    els.toolExecute.classList.remove('hidden');
    showToolResult(data.result !== null && data.result !== undefined ? data.result : data);
    loadTools();
  } catch (err) {
    showToolResult({ error: tm(err.message) });
  } finally {
    els.toolApprove.disabled = false;
    els.toolDeny.disabled = false;
  }
}

async function loadToolHistory() {
  els.historyList.innerHTML = '';
  try {
    const data = await api.get('/api/tools/history?limit=50');
    const events = (data.events || []).filter((e) => e.event === 'tool_run' || e.event === 'tool_denied');
    if (!events.length) {
      const empty = document.createElement('p');
      empty.className = 'history-empty';
      empty.textContent = tr('history.empty');
      els.historyList.append(empty);
      return;
    }
    for (const event of events) {
      const row = document.createElement('div');
      row.className = `history-row ${event.event === 'tool_denied' || event.ok === false ? 'fail' : 'ok'}`;
      const when = document.createElement('span');
      when.className = 'when';
      when.textContent = event.ts ? i18n.formatDateTime(event.ts) : '';
      const what = document.createElement('span');
      what.className = 'what';
      what.textContent = event.event === 'tool_denied'
        ? tr('history.denied', { tool: event.tool || '?', by: event.decided_by || '?' })
        : tr('history.run', { tool: event.tool || '?', result: event.ok ? tr('history.ok') : tr('tool.error', { text: tm(event.error) || '?' }), ms: event.duration_ms ?? '?', source: event.source || '?' });
      row.append(when, what);
      els.historyList.append(row);
    }
  } catch (err) {
    const error = document.createElement('p');
    error.className = 'history-empty';
    error.textContent = tr('history.failed', { text: tm(err.message) });
    els.historyList.append(error);
  }
}

function toolEventCard(kind, title, detail) {
  const article = document.createElement('article');
  article.className = `message tool ${kind}`;
  const header = document.createElement('header');
  const label = document.createElement('span');
  label.className = 'role';
  label.textContent = title;
  header.append(label);
  const body = document.createElement('div');
  body.className = 'content';
  body.textContent = detail;
  article.append(header, body);
  return article;
}

// Escolha de profundidade: cartão compacto no próprio chat, só quando o pedido é uma investigação complexa e o nível não está claro.
function depthCard(event) {
  const article = document.createElement('article');
  article.className = 'message permission research-depth';
  const body = document.createElement('div');
  body.className = 'content';
  const question = document.createElement('div');
  i18n.bindTm(question, event.question);
  const actions = document.createElement('div');
  actions.className = 'permission-actions';
  const buttons = [];
  for (const option of event.options) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = option.mode === 'deep' ? 'btn primary' : 'btn ghost';
    btn.title = tm(option.hint);
    btn.textContent = tm(option.label);
    const hint = document.createElement('span');
    hint.className = 'meta';
    hint.textContent = ` ${tm(option.hint)} `;
    btn.addEventListener('click', () => {
      buttons.forEach((b) => { b.disabled = true; });
      const done = document.createElement('span');
      done.className = 'meta ok';
      done.textContent = tr('depth.chosen', { label: tm(option.label) });
      actions.replaceChildren(done);
      resumeConversation(event.conversation_id || state.currentId, true, option.mode);
    });
    buttons.push(btn);
    actions.append(btn, hint);
  }
  body.append(question, actions);
  article.append(body);
  return article;
}

function permissionCard(event) {
  const article = document.createElement('article');
  article.className = 'message permission';
  const header = document.createElement('header');
  const label = document.createElement('span');
  label.className = 'role';
  label.textContent = tr('perm.title');
  header.append(label);
  const body = document.createElement('div');
  body.className = 'content';
  // Cartão de pesquisa pública da conversa: uma autorização para todas as buscas desta conversa
  // (sem cartão por domínio), revogável pelo próprio cartão após a aprovação.
  const isResearch = event.inline && (event.tool === 'Pesquisa na internet' || event.tool === 'Web search');
  const intro = document.createElement('div');
  intro.textContent = event.research && event.tool === 'Investigação passiva'
    ? tr('perm.passive')
    : isResearch
    ? tr('perm.research')
    : event.inline
    ? tr('perm.inline')
    : tr('perm.tool', { tool: event.tool });
  const args = document.createElement('pre');
  args.className = 'args';
  args.textContent = JSON.stringify(event.arguments, null, 2);
  body.append(intro, args);
  if (event.reason) {
    const reason = document.createElement('div');
    reason.className = 'reason';
    reason.textContent = tm(event.reason);
    body.append(reason);
  }
  const actions = document.createElement('div');
  actions.className = 'permission-actions';
  const rememberLabel = document.createElement('label');
  const remember = document.createElement('input');
  remember.type = 'checkbox';
  const rememberText = document.createElement('span');
  rememberText.textContent = tr('perm.remember');
  rememberLabel.append(remember, rememberText);
  const approve = document.createElement('button');
  approve.className = 'btn primary';
  approve.type = 'button';
  approve.textContent = tr('tool.approve');
  const deny = document.createElement('button');
  deny.className = 'btn ghost';
  deny.type = 'button';
  deny.textContent = tr('tool.deny');
  const finish = (text, ok) => {
    approve.remove();
    deny.remove();
    rememberLabel.remove();
    const status = document.createElement('span');
    status.className = `meta ${ok ? 'ok' : 'denied'}`;
    status.textContent = text;
    actions.append(status);
  };
  const addRevoke = () => {
    const revoke = document.createElement('button');
    revoke.className = 'btn ghost';
    revoke.type = 'button';
    revoke.textContent = tr('perm.revoke');
    revoke.addEventListener('click', async () => {
      revoke.disabled = true;
      try {
        const conv = event.conversation_id || state.currentId;
        const qs = new URLSearchParams({ capability: 'net.fetch' });
        if (conv) qs.set('conversation_id', conv);
        await api.del(`/api/permissions/grants?${qs.toString()}`);
        const status = document.createElement('span');
        status.className = 'meta denied';
        status.textContent = tr('perm.revoked');
        revoke.replaceWith(status);
        loadTools();
      } catch (err) {
        revoke.disabled = false;
        showToast(tr('perm.failed', { text: tm(err.message) }), 'error');
      }
    });
    actions.append(revoke);
  };
  const decide = async (granted) => {
    approve.disabled = true;
    deny.disabled = true;
    try {
      const data = await api.post(`/api/permissions/${event.request_id}/decision`, {
        granted,
        remember_session: remember.checked,
        expected_source: 'chat',
        conversation_id: state.currentId || undefined,
      });
      finish(granted ? tr('perm.granted') : tr('perm.denied'), granted);
      if (isResearch && granted) addRevoke(); // revogação explícita da autorização de pesquisa
      loadTools();
      if (event.inline) {
        resumeConversation(event.conversation_id || state.currentId, true); // repete a pergunta com a decisão sobre a pesquisa
      } else if (data && data.conversation_id && granted) {
        resumeConversation(data.conversation_id);
      }
    } catch (err) {
      approve.disabled = false;
      deny.disabled = false;
      finish(tr('perm.failed', { text: tm(err.message) }), false);
    }
  };
  approve.addEventListener('click', () => decide(true));
  deny.addEventListener('click', () => decide(false));
  if (!isResearch) actions.append(rememberLabel); // pesquisa da conversa: a aprovação já vale para a conversa inteira
  actions.append(approve, deny);
  body.append(actions);
  article.append(header, body);
  return article;
}

function setBusy(busy) {
  els.composer.classList.toggle('busy', busy);
  els.send.disabled = busy;
  els.stop.hidden = !busy;
}

function numericOrNull(value) {
  return value === '' || value === null ? null : Number(value);
}

// Retorna o desfecho do stream: 'done', 'error' ou 'pending' (aguarda autorização de ferramenta).
async function consumeStream(body, assistantBubble, userBubble = null) {
  let assistantAttached = false;
  let progressCard = null;
  let researchPanel = null;
  let webSources = null;
  let outcome = 'done';
  setCopyReady(assistantBubble, false);
  const thinking = createThinking();
  // eventos novos entram antes do indicador, que fica sempre por último até o primeiro token
  const addNode = (node) => { els.messages.append(node); if (thinking.isConnected) els.messages.append(thinking); scrollMessagesToBottom(); };
  els.messages.append(thinking);
  scrollMessagesToBottom();
  state.controller = new AbortController();
  try {
    await api.chatStream(body, (event) => {
      if (event.type === 'meta') {
        state.currentId = event.conversation_id;
        els.chatTitle.textContent = event.conversation_title;
        syncEmptyState();
        renderConversationList();
        if (userBubble && event.conversation_id) {
          for (const btn of userBubble.querySelectorAll('.msg-att-btn[data-att-id]')) {
            const attId = btn.dataset.attId;
            const img = btn.querySelector('img');
            if (attId && img) img.src = attachmentUrl({ id: attId }, event.conversation_id);
          }
        }
        state.repackNoticeActive = Boolean(event.context && event.context.repack_triggered);
        state.overBudgetActive = Boolean(event.context && event.context.over_budget);
        if (event.context) {
          const pct = Math.round((event.context.usage_ratio || 0) * 100);
          const note = event.context.repack_triggered ? ` · ${tm(event.context.repack_notice) || tr('context.optimized')}` : '';
          const warn = event.context.over_budget ? ` · ${tr('context.overBudget')}` : '';
          els.contextInfo.textContent = `${tr('context.line', { used: event.context.estimated_tokens, budget: event.context.input_budget, pct })}${note}${warn}`;
          if (event.context.notice) showToast(event.context.notice, 'error');
        }
      } else if (event.type === 'thinking') {
        appendReasoning(thinking, event.content);
        scrollMessagesToBottom();
      } else if (event.type === 'tool_call') {
        addNode(toolEventCard(
          'call',
          tr('tool.cardTitle', { name: event.tool }),
          tr('tool.arguments', { json: JSON.stringify(event.arguments) }),
        ));
      } else if (event.type === 'tool_result') {
        const detail = event.ok
          ? `${event.output_preview || tr('tool.noText')}${event.data && event.data.duration_ms !== undefined ? ` · ${event.data.duration_ms} ms` : ''}`
          : tr('tool.error', { text: tm(event.error) || '?' });
        addNode(toolEventCard(event.ok ? 'ok' : 'fail', tr('tool.resultPrefix', { name: event.tool }), detail));
      } else if (event.type === 'research_depth_request') {
        addNode(depthCard(event));
        if (event.conversation_id && !state.currentId) state.currentId = event.conversation_id;
        outcome = 'pending';
      } else if (event.type === 'research_progress' || event.type === 'research_summary') {
        // um único cartão na coluna do chat, atualizado no lugar (sem duplicar nem usar posição fixa)
        if (!progressCard) { progressCard = window.EnslavedResearch.createCard(); addNode(progressCard.node); }
        progressCard.update(event);
      } else if (event.type === 'permission_request') {
        addNode(permissionCard(event));
        if (event.inline && event.conversation_id && !state.currentId) state.currentId = event.conversation_id;
        outcome = 'pending';
      } else if (event.type === 'token' || event.type === 'replace') { // replace: o servidor corrigiu o texto já exibido
        if (!assistantAttached) {
          settleReasoning(thinking, assistantBubble);
          removeThinking(thinking);
          addNode(assistantBubble);
          assistantBubble.classList.add('live');
          assistantBubble.querySelector('.content').classList.add('streaming');
          assistantAttached = true;
        }
        const content = assistantBubble.querySelector('.content');
        appendContent(content, event.content, event.type === 'replace');
      } else if (event.type === 'verification') {
        webSources = Array.isArray(event.sources) ? event.sources : null;
        const note = verificationNote(event, body.message);
        if (note) addToBubble(assistantBubble, note);
      } else if (event.type === 'done') {
        finalizeContent(assistantBubble.querySelector('.content'));
        if (webSources) linkifyWebCitations(assistantBubble.querySelector('.content'), webSources);
        assistantBubble.querySelector('.content').classList.remove('streaming');
        const isPartial = Boolean(event.research && event.research.completion && event.research.completion.status === 'partial');
        setCopyReady(assistantBubble, true, isPartial);
        if (progressCard) progressCard.finish(isPartial ? 'partial' : 'done');
        if (event.research && event.research.view) {
          researchPanel = window.EnslavedResearch.attach(assistantBubble, event.research.view);
          researchPanel.link();
        }
        scrollMessagesToBottom();
      } else if (event.type === 'error') {
        outcome = 'error';
        if (progressCard) progressCard.finish('error');
        removeThinking(thinking);
        if (assistantAttached) assistantBubble.remove();
        addNode(errorBubble(event.message));
      }
    }, state.controller.signal);
    return outcome;
  } finally {
    removeThinking(thinking);
    assistantBubble.classList.remove('live');
    finalizeContent(assistantBubble.querySelector('.content'));
    if (webSources) linkifyWebCitations(assistantBubble.querySelector('.content'), webSources); // stream interrompido: citações também viram links
    assistantBubble.querySelector('.content').classList.remove('streaming');
    if (researchPanel) researchPanel.link(); // a renderização final recria o texto: religa F#/E# ao painel
    if (progressCard) progressCard.finish('interrupted'); // cancelamento/desconexão: nunca deixa o cartão "em andamento"
    scrollMessagesToBottom();
    state.controller = null;
  }
}

async function resumeConversation(conversationId, retry = false, researchMode = null) {
  if (state.streaming) return;
  state.streaming = true;
  state.autoScroll = isNearBottom();
  setBusy(true);
  window.EnslavedScene?.busy();
  els.inferState.textContent = tr('infer.continuing');
  const assistantBubble = messageBubble('assistant', '');
  const body = {
    message: '',
    conversation_id: conversationId,
    resume: !retry,
    retry,
    research_mode: researchMode,
    system_prompt: state.settings.systemPrompt || null,
    temperature: numericOrNull(state.settings.temperature),
    num_predict: numericOrNull(state.settings.numPredict),
    use_tools: els.toolsToggle.checked && state.tools.length > 0,
    think: els.thinkToggle.checked,
    activated_skills: activeSkillSet(),
    language: i18n.lang,
  };
  let outcome = 'error';
  try {
    outcome = await consumeStream(body, assistantBubble);
  } catch (err) {
    if (err.name === 'AbortError') outcome = 'cancel';
    else els.messages.append(errorBubble(err.message));
  } finally {
    window.EnslavedScene?.finish(outcome);
    state.streaming = false;
    state.controller = null;
    setBusy(false);
    els.inferState.textContent = '';
    scrollMessagesToBottom();
  }
  loadConversations().catch((err) => console.error(err));
  loadMemoryStatus().catch((err) => console.error(err));
}

async function sendMessage(event) {
  event.preventDefault();
  const text = els.input.value.trim();
  if (state.uploadingAttachments > 0) {
    showToast(tr('composer.waitUpload'));
    return;
  }
  const currentAttachments = [...state.pendingAttachments];
  if ((!text && !currentAttachments.length) || state.streaming) return;
  state.streaming = true;
  state.autoScroll = true;
  state.repackNoticeActive = false;
  setBusy(true);
  window.EnslavedScene?.busy();
  els.inferState.textContent = tr('infer.generating');
  els.input.value = '';
  autosizeInput();
  state.pendingAttachments = [];
  renderPendingAttachments();

  const displayText = text || tr('composer.imageOnly');
  if (!state.currentId) els.messages.innerHTML = '';
  const userBubble = messageBubble('user', displayText, { attachments: currentAttachments }, state.currentId);
  els.messages.append(userBubble);
  syncEmptyState();
  scrollMessagesToBottom(true);
  const assistantBubble = messageBubble('assistant', '');
  const allowUnverified = state.allowUnverified;
  state.allowUnverified = false;

  const body = {
    message: text,
    attachment_ids: currentAttachments.map((a) => a.id),
    conversation_id: state.currentId,
    system_prompt: state.settings.systemPrompt || null,
    temperature: numericOrNull(state.settings.temperature),
    num_predict: numericOrNull(state.settings.numPredict),
    allow_unverified: allowUnverified,
    use_tools: els.toolsToggle.checked && state.tools.length > 0,
    think: els.thinkToggle.checked,
    activated_skills: activeSkillSet(),
    language: i18n.lang,
  };
  let outcome = 'error';
  try {
    outcome = await consumeStream(body, assistantBubble, userBubble);
  } catch (err) {
    if (err.name === 'AbortError') outcome = 'cancel';
    else els.messages.append(errorBubble(err.message));
  } finally {
    window.EnslavedScene?.finish(outcome);
    state.streaming = false;
    state.controller = null;
    setBusy(false);
    els.inferState.textContent = '';
    els.input.focus({ preventScroll: true });
    scrollMessagesToBottom();
  }
  loadConversations().catch((err) => console.error(err));
  loadMemoryStatus().catch((err) => console.error(err));
}

els.composer.addEventListener('submit', sendMessage);
els.attachBtn.addEventListener('click', () => els.attachInput.click());
els.attachInput.addEventListener('change', () => {
  if (els.attachInput.files && els.attachInput.files.length) {
    attachFiles(els.attachInput.files);
    els.attachInput.value = '';
  }
});
els.input.addEventListener('paste', (event) => {
  const items = [...(event.clipboardData?.items || [])];
  const files = items
    .filter((it) => it.kind === 'file' && (!it.type || it.type.startsWith('image/')))
    .map((it) => it.getAsFile())
    .filter(Boolean);
  if (files.length) {
    event.preventDefault();
    attachFiles(files);
  }
});
for (const zone of [els.composer, els.chat]) {
  zone.addEventListener('dragover', (event) => {
    if (openModals().length) return;
    if (event.dataTransfer && [...event.dataTransfer.types].includes('Files')) {
      event.preventDefault();
      zone.classList.add('dragover');
    }
  });
  zone.addEventListener('dragleave', (event) => {
    if (!zone.contains(event.relatedTarget)) zone.classList.remove('dragover');
  });
  zone.addEventListener('drop', (event) => {
    zone.classList.remove('dragover');
    if (openModals().length) return;
    const files = event.dataTransfer?.files;
    if (files && files.length) {
      event.preventDefault();
      event.stopPropagation();
      attachFiles(files);
    }
  });
}
els.newChat.addEventListener('click', newChat);
els.toggleSelectConversations.addEventListener('click', () => {
  state.selectionMode = !state.selectionMode;
  if (!state.selectionMode) state.selectedIds.clear();
  renderConversationList();
});
els.selectAllConversations.addEventListener('click', () => {
  for (const c of state.conversations) state.selectedIds.add(c.id);
  renderConversationList();
});
els.clearSelectedConversations.addEventListener('click', () => {
  state.selectedIds.clear();
  renderConversationList();
});
els.deleteSelectedConversations.addEventListener('click', () => {
  confirmAndDeleteConversations([...state.selectedIds], { fromBulk: true });
});
els.openAbout.addEventListener('click', () => els.aboutModal.classList.remove('hidden'));
els.openSettings.addEventListener('click', () => {
  els.settingsTemperature.value = state.settings.temperature ?? '';
  els.settingsNumPredict.value = state.settings.numPredict ?? '';
  els.settingsSystem.value = state.settings.systemPrompt || '';
  els.settingsModal.classList.remove('hidden');
  loadLocalApi();
});
els.settingsClose.addEventListener('click', () => closeModal(els.settingsModal));
els.skillActivate.addEventListener('click', async () => {
  if (!state.currentSkillId) return;
  const activate = !els.skillActivate.dataset.active;
  try {
    await api.post(`/api/skills/${state.currentSkillId}/enabled`, { enabled: true });
  } catch (err) { /* skill pode já estar habilitada; ativação local segue */ }
  activateSkillLocally(state.currentSkillId, activate);
  i18n.bind(els.skillActivate, activate ? 'skill.deactivate' : 'skill.activate');
  els.skillActivate.dataset.active = activate ? '1' : '';
});
els.toggleSidebar.addEventListener('click', toggleSidebar);
els.scrim.addEventListener('click', () => setSidebarOpen(false));
// no celular, abrir um modal pelo menu fecha a gaveta para não ficar atrás dele
els.sidebar.querySelector('.sidebar-bottom').addEventListener('click', (event) => {
  if (event.target.closest('.nav-item, .nav-group li')) setSidebarOpen(false);
});
els.stop.addEventListener('click', () => { if (state.controller) state.controller.abort(); });

function autosizeInput() {
  els.input.style.height = 'auto';
  els.input.style.height = `${Math.min(els.input.scrollHeight, 220)}px`;
}
els.input.addEventListener('input', () => {
  autosizeInput();
  window.EnslavedScene?.typing(els.input.value.trim().length > 0);
});
els.input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    els.composer.requestSubmit();
  }
});
els.settingsSave.addEventListener('click', async () => {
  state.settings.temperature = els.settingsTemperature.value;
  state.settings.numPredict = els.settingsNumPredict.value;
  state.settings.systemPrompt = els.settingsSystem.value;
  if (!els.settingModelField.hidden && state.settings.savedModel && els.settingModel.value !== state.settings.savedModel) {
    const chosen = els.settingModel.value;
    try {
      const res = await api.post('/api/model', { model: chosen });
      state.settings.savedModel = chosen;
      showToast(tr('settings.modelRestart', { model: res.selected || chosen }));
    } catch (err) {
      els.settingModel.value = state.settings.savedModel;
      showToast(tm(err.message) || tr('settings.modelSwitchFailed'), 'error');
    }
  }
  closeModal(els.settingsModal);
});
els.toolsToggle.addEventListener('change', () => {
  state.toolsEnabled = els.toolsToggle.checked;
});

// ------------------------------------------------------------------ API Local (/v1 para IDEs e CLIs)

let localApiStatus = null;

function renderLocalApiKeys() {
  if (!localApiStatus) return;
  els.localapiCounter.textContent = `${localApiStatus.active_keys}/${localApiStatus.max_keys}`;
  els.localapiCreate.disabled = localApiStatus.active_keys >= localApiStatus.max_keys;
  els.localapiKeys.innerHTML = '';
  if (!localApiStatus.keys.length) {
    const empty = document.createElement('div');
    empty.className = 'list-empty';
    i18n.bind(empty, 'localapi.noKeys');
    els.localapiKeys.append(empty);
    return;
  }
  for (const key of localApiStatus.keys) {
    const row = document.createElement('div');
    row.className = 'memory-row';
    const head = document.createElement('div');
    head.className = 'memory-row-head';
    const name = document.createElement('strong');
    name.textContent = key.name;
    const badge = document.createElement('span');
    badge.className = 'chip';
    i18n.bind(badge, key.status === 'active' ? 'localapi.statusActive' : 'localapi.statusRevoked');
    head.append(name, badge);
    const meta = document.createElement('div');
    meta.className = 'memory-origin';
    meta.textContent = [
      key.masked,
      `${tr('localapi.created')}: ${i18n.formatDateTime(key.created_at)}`,
      `${tr('localapi.lastUsed')}: ${key.last_used_at ? i18n.formatDateTime(key.last_used_at) : tr('localapi.neverUsed')}`,
    ].join(' · ');
    const actions = document.createElement('div');
    actions.className = 'memory-actions';
    if (key.status === 'active') {
      const revoke = document.createElement('button');
      revoke.type = 'button';
      revoke.className = 'btn xs danger';
      i18n.bind(revoke, 'localapi.revoke');
      revoke.addEventListener('click', () => revokeLocalApiKey(key));
      actions.append(revoke);
    }
    row.append(head, meta, actions);
    els.localapiKeys.append(row);
  }
}

function renderLocalApiExamples() {
  if (!localApiStatus) return;
  const { endpoint, model, num_ctx, num_predict } = localApiStatus;
  // Doc oficial (opencode.ai/docs/models): modelo customizado precisa declarar limites e tools reais.
  const opencode = {
    $schema: 'https://opencode.ai/config.json',
    providers: {
      enslaved: {
        name: 'Enslaved (local)',
        env: ['ENSLAVED_API_KEY'],
        package: '@opencode/ai/providers/openai-compatible',
        settings: { baseURL: endpoint, apiKey: '{env:ENSLAVED_API_KEY}' },
        models: {
          [model]: {
            name: 'Enslaved RANA',
            capabilities: { tools: Boolean(localApiStatus.model_supports_tools), input: ['text'], output: ['text'] },
            limit: { context: num_ctx, output: num_predict },
          },
        },
      },
    },
  };
  const opencodeV1 = {
    $schema: 'https://opencode.ai/config.json',
    provider: {
      enslaved: {
        name: 'Enslaved (local)',
        npm: '@ai-sdk/openai-compatible',
        options: { baseURL: endpoint, apiKey: '{env:ENSLAVED_API_KEY}' },
        models: {
          [model]: {
            name: 'Enslaved RANA',
            tool_call: Boolean(localApiStatus.model_supports_tools),
            limit: { context: num_ctx, output: num_predict },
          },
        },
      },
    },
  };
  const cont = [
    'name: Enslaved',
    'version: 0.0.1',
    'schema: v1',
    '',
    'models:',
    '  - name: Enslaved RANA (local)',
    '    provider: openai',
    `    model: ${model}`,
    `    apiBase: ${endpoint}`,
    '    apiKey: ens_YOUR_KEY',
    '    roles: [chat, edit, apply]',
    '    defaultCompletionOptions:',
    `      contextLength: ${num_ctx}`,
    `      maxTokens: ${num_predict}  # acima disso a API responde HTTP 400 (output_budget_exceeded)`,
  ].join('\n');
  const cli = [
    `curl ${endpoint}/chat/completions \\`,
    '  -H "Authorization: Bearer ens_YOUR_KEY" \\',
    '  -H "Content-Type: application/json" \\',
    `  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Write a Python FizzBuzz function"}], "stream": true}'`,
  ].join('\n');
  els.localapiExamples.textContent = [
    `${tr('localapi.exOpencode')}\n${JSON.stringify(opencode, null, 2)}`,
    `${tr('localapi.exOpencodeV1')}\n${JSON.stringify(opencodeV1, null, 2)}`,
    tr('localapi.exOpencodeNote'),
    `${tr('localapi.exContinue')}\n${cont}`,
    `${tr('localapi.exCli')}\n${cli}`,
    tr('localapi.exKeyNote'),
  ].join('\n\n');
}

async function loadLocalApi() {
  try {
    localApiStatus = await api.get('/api/localapi');
    els.localapiEnabled.value = localApiStatus.enabled ? 'on' : 'off';
    els.localapiPanel.hidden = !localApiStatus.enabled;
    els.localapiEndpoint.value = localApiStatus.endpoint;
    els.localapiModelLine.textContent = tr('localapi.modelLine', { model: localApiStatus.model });
    renderLocalApiKeys();
    renderLocalApiExamples();
  } catch (err) {
    console.error('localapi', err);
  }
}

async function revokeLocalApiKey(key) {
  const ok = await askDialog({
    title: tr('localapi.revokeTitle'),
    message: tr('localapi.revokeMessage', { name: key.name }),
    okLabel: tr('localapi.revoke'),
    danger: true,
  });
  if (!ok) return;
  try {
    await api.del(`/api/localapi/keys/${encodeURIComponent(key.id)}`);
    showToast(tr('localapi.revokedDone'));
  } catch (err) {
    showToast(err.message, 'error');
  }
  await loadLocalApi();
}

els.localapiEnabled.addEventListener('change', async () => {
  const enabled = els.localapiEnabled.value === 'on';
  try {
    await api.post('/api/localapi/enabled', { enabled });
    els.localapiPanel.hidden = !enabled;
    if (enabled) {
      els.localapiReveal.hidden = true;
      await loadLocalApi();
    }
    showToast(tr(enabled ? 'localapi.enabledToast' : 'localapi.disabledToast'));
  } catch (err) {
    els.localapiEnabled.value = enabled ? 'off' : 'on';
    showToast(err.message, 'error');
  }
});

els.localapiCreate.addEventListener('click', async () => {
  const name = els.localapiNewName.value.trim();
  if (!name) {
    showToast(tr('localapi.nameRequired'), 'error');
    return;
  }
  try {
    const created = await api.post('/api/localapi/keys', { name });
    els.localapiNewName.value = '';
    els.localapiReveal.hidden = false;
    els.localapiRevealKey.value = created.secret;
    await loadLocalApi();
    showToast(tr('localapi.createdToast'));
  } catch (err) {
    showToast(err.message, 'error');
  }
});

els.localapiNewName.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    els.localapiCreate.click();
  }
});

async function copyLocalApiValue(input, successKey) {
  try {
    await copyText(input.value);
    showToast(tr(successKey));
  } catch (err) {
    showToast(err.message || tr('copy.unavailable'), 'error');
  }
}
els.localapiCopyEndpoint.addEventListener('click', () => copyLocalApiValue(els.localapiEndpoint, 'copy.copied'));
els.localapiCopyKey.addEventListener('click', () => copyLocalApiValue(els.localapiRevealKey, 'localapi.keyCopied'));

// A chave completa só existe enquanto o dono não fechou as Configurações: ao fechar, some da tela.
els.settingsModal.addEventListener('dismiss', () => {
  els.localapiReveal.hidden = true;
  els.localapiRevealKey.value = '';
});
els.openToolHistory.addEventListener('click', () => {
  els.historyModal.classList.remove('hidden');
  loadToolHistory();
});
els.historyRefresh.addEventListener('click', loadToolHistory);
els.toolExecute.addEventListener('click', executeCurrentTool);
els.toolApprove.addEventListener('click', () => decideCurrentTool(true));
els.toolDeny.addEventListener('click', () => decideCurrentTool(false));

// ------------------------------------------------------------------ contexto e histórico

async function loadMemoryStatus() {
  try {
    const qs = state.currentId ? `?conversation_id=${encodeURIComponent(state.currentId)}` : '';
    const data = await api.get(`/api/memory/status${qs}`);
    const cs = data.context_state;
    const notice = state.repackNoticeActive ? ` · ${tr('context.optimized')}` : '';
    const warn = state.overBudgetActive ? ` · ${tr('context.overBudget')}` : '';
    if (cs) {
      const pct = Math.round((cs.usage_ratio || 0) * 100);
      els.contextInfo.textContent = `${tr('context.line', { used: cs.estimated_tokens, budget: cs.budget_tokens, pct })}${notice}${warn}`;
    } else {
      els.contextInfo.textContent = `${tr('context.zero', { budget: data.input_budget })}${notice}${warn}`;
    }
    return data;
  } catch (err) {
    els.contextInfo.textContent = tr('context.empty');
    return null;
  }
}

function renderMemoryContextBox(status) {
  if (!status) {
    els.memoryContextBox.textContent = tr('memory.unavailable');
    return;
  }
  const cs = status.context_state;
  const pct = cs ? Math.round((cs.usage_ratio || 0) * 100) : 0;
  const est = cs ? cs.estimated_tokens : 0;
  const bud = cs ? cs.budget_tokens : status.input_budget;
  const thr = cs ? cs.threshold_tokens : status.threshold_tokens;
  const exportLine = status.export
    ? tr('memory.exportLine', { status: status.export.status, seq: status.export.last_exported_seq, path: status.export.markdown_path })
    : tr('memory.exportWaiting');
  const line = (labelKey, value) => `<strong>${escapeHtml(tr(labelKey))}</strong> ${escapeHtml(value)}`;
  els.memoryContextBox.innerHTML = [
    line('memory.budget', tr('memory.budgetLine', { est, bud, pct, thr, thrPct: Math.round(status.threshold_ratio * 100) })),
    line('memory.ctxModel', tr('memory.ctxLine', { numCtx: status.num_ctx, reserve: status.output_reserve, calib: status.calibration_factor })),
    line('memory.repacksHere', String(status.repack_count)),
    `<strong>${escapeHtml(exportLine)}</strong>`,
  ].join('<br>');
}

function renderMemoryRepacks(repacks) {
  els.memoryRepacks.innerHTML = '';
  if (!repacks.length) {
    const empty = document.createElement('p');
    empty.className = 'history-empty';
    empty.textContent = tr('memory.noRepacks');
    els.memoryRepacks.append(empty);
    return;
  }
  for (const r of repacks) {
    const row = document.createElement('div');
    row.className = 'memory-row';
    const head = document.createElement('div');
    head.className = 'memory-row-head';
    head.innerHTML = `<strong>Repack v${String(r.version).padStart(3, '0')} · seq ${r.covered_from_seq}..${r.covered_to_seq}</strong><span class="meta">${r.token_estimate_before} -> ${r.token_estimate_after} tokens</span>`;
    const body = document.createElement('div');
    body.textContent = r.summary_text;
    const origin = document.createElement('div');
    origin.className = 'memory-origin';
    origin.textContent = tr('memory.origin', { ids: JSON.stringify(r.source_message_ids), path: r.markdown_path });
    row.append(head, body, origin);
    els.memoryRepacks.append(row);
  }
}

async function loadMemoryModalData() {
  const status = await loadMemoryStatus();
  renderMemoryContextBox(status);
  els.memoryRepackBtn.disabled = !state.currentId;
  els.memoryMdBtn.disabled = !state.currentId;
  const repacksUrl = state.currentId
    ? `/api/memory/repacks?conversation_id=${encodeURIComponent(state.currentId)}`
    : '/api/memory/repacks?limit=15';
  const repacksData = await api.get(repacksUrl);
  renderMemoryRepacks(repacksData.repacks || []);
}

els.openMemory.addEventListener('click', async () => {
  els.memoryMdView.classList.add('hidden');
  els.memoryModal.classList.remove('hidden');
  await loadMemoryModalData();
});
els.memoryRefreshBtn.addEventListener('click', loadMemoryModalData);
els.memoryRepackBtn.addEventListener('click', async () => {
  if (!state.currentId) return;
  els.memoryRepackBtn.disabled = true;
  try {
    const res = await api.post(`/api/memory/conversations/${state.currentId}/repack`, {});
    if (!res.ok && res.error) showToast(tm(res.error), 'error');
    else showToast(tr('memory.compacted'));
    await loadMemoryModalData();
  } finally {
    els.memoryRepackBtn.disabled = !state.currentId;
  }
});
els.memoryMdBtn.addEventListener('click', async () => {
  if (!state.currentId) return;
  if (!els.memoryMdView.classList.contains('hidden')) {
    els.memoryMdView.classList.add('hidden');
    return;
  }
  try {
    const data = await api.get(`/api/memory/conversations/${state.currentId}/markdown`);
    els.memoryMdView.textContent = `${tr('memory.file', { path: data.path })}\n\n${data.content}`;
    els.memoryMdView.classList.remove('hidden');
  } catch (err) { showToast(err.message, 'error'); }
});

// ------------------------------------------------------------------ idioma da interface (Configurações)

const languageSelect = document.getElementById('setting-language');
for (const [code, name] of Object.entries(i18n.SUPPORTED)) languageSelect.add(new Option(name, code));
languageSelect.value = i18n.lang;
languageSelect.addEventListener('change', () => i18n.setLanguage(languageSelect.value)); // imediato; a preferência fica no localStorage

// O texto estático troca sozinho (data-i18n); aqui só o que o JavaScript desenha. A conversa aberta é redesenhada a partir do
// histórico salvo (nada é alterado nem retraduzido): rascunho, anexos pendentes e rolagem ficam como estão.
document.addEventListener('enslaved:languagechange', async () => {
  languageSelect.value = i18n.lang;
  if (!state.currentId && !els.messages.children.length) els.chatTitle.textContent = tr('chat.new');
  renderConversationList();
  renderSkillList();
  renderToolList();
  if (localApiStatus) {
    els.localapiModelLine.textContent = tr('localapi.modelLine', { model: localApiStatus.model });
    renderLocalApiKeys();
    renderLocalApiExamples();
  }
  loadHealth();
  loadDiagnostics();
  loadMemoryStatus();
  if (!els.memoryModal.classList.contains('hidden')) loadMemoryModalData().catch((err) => console.error(err));
  if (!els.historyModal.classList.contains('hidden')) loadToolHistory();
  if (state.currentId && !state.streaming) {
    const scrollTop = els.messages.scrollTop;
    await openConversation(state.currentId, { keepAttachments: true });
    els.messages.scrollTop = scrollTop;
    state.autoScroll = isNearBottom();
  }
});

loadConfig()
  .then(() => Promise.all([loadSkills(), loadHealth(), loadTools(), loadLocalApi()]))
  .catch((err) => {
    console.error(err);
    return loadHealth();
  });
loadDiagnostics();
setInterval(loadHealth, 15000);
setInterval(loadDiagnostics, 30000);
loadConversations().catch((err) => console.error(err));
newChat();
