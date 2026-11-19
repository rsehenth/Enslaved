// Internacionalização do frontend (pt-BR / en-US). Catálogos planos em /static/i18n/<idioma>.json.
// API: tr(chave, {parâmetros}) · tm(textoDoBackend) · i18n.setLanguage(código) · i18n.bind/bindAttr (nós que se retraduzem sozinhos).
// Idiomas novos: ver "Translations" no README (arquivo JSON + uma linha em SUPPORTED).
(() => {
  const SUPPORTED = { 'en-US': 'English', 'pt-BR': 'Português (Brasil)' }; // nomes próprios: nunca traduzidos
  const DEFAULT = 'en-US'; // instalações novas sem preferência salva iniciam em inglês
  const STORAGE_KEY = 'enslaved.language'; // só o código do idioma; nenhum dado sensível
  const catalogs = {};
  const missing = new Set();
  const patternCache = new Map();

  // shortcut: XHR síncrono porque app.js desenha a interface no carregamento; são dois JSON locais pequenos. Migrar para async se o catálogo crescer muito.
  function load(lang) {
    if (catalogs[lang]) return catalogs[lang];
    try {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', `/static/i18n/${lang}.json`, false);
      xhr.send();
      catalogs[lang] = xhr.status === 200 ? JSON.parse(xhr.responseText) : {};
    } catch (err) {
      console.error(`i18n: catálogo ${lang} indisponível`, err);
      catalogs[lang] = {};
    }
    return catalogs[lang];
  }

  function readPreference() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved && SUPPORTED[saved] ? saved : DEFAULT;
    } catch (err) { return DEFAULT; }
  }

  let lang = readPreference();
  load(DEFAULT);
  load(lang);

  function noteMissing(key) {
    const id = `${lang}:${key}`;
    if (missing.has(id)) return;
    missing.add(id);
    console.warn(`i18n: chave ausente "${key}" em ${lang}`);
  }

  function interpolate(text, params) {
    return text.replace(/\{(\w+)\}/g, (match, name) => (params && name in params ? String(params[name]) : match));
  }

  // Plural: com params.count procura `chave_one` (count === 1) ou `chave_other`; cai na chave simples.
  function tr(key, params) {
    let k = key;
    if (params && typeof params.count === 'number') {
      const plural = `${key}_${params.count === 1 ? 'one' : 'other'}`;
      if (typeof catalogs[lang][plural] === 'string' || typeof catalogs[DEFAULT][plural] === 'string') k = plural;
    }
    let text = catalogs[lang][k];
    if (typeof text !== 'string') {
      noteMissing(k);
      text = catalogs[DEFAULT][k]; // fallback seguro: idioma-fonte
    }
    if (typeof text !== 'string') return key;
    return interpolate(text, params);
  }

  function cachedRegExp(source, flags) {
    const id = `${flags}/${source}`;
    if (!patternCache.has(id)) patternCache.set(id, new RegExp(source, flags));
    return patternCache.get(id);
  }

  // Textos fixos gerados pelo backend → idioma atual. Só traduz o que está catalogado; o resto passa intacto.
  // "backend.exact": {texto: tradução} · "backend.patterns": [{re, t, flags?}] com {1}, {2}… para os grupos capturados.
  function tm(text) {
    if (typeof text !== 'string' || !text) return text;
    const cat = catalogs[lang];
    const exact = cat['backend.exact'];
    if (exact && Object.prototype.hasOwnProperty.call(exact, text)) return exact[text];
    let out = text;
    for (const p of cat['backend.patterns'] || []) {
      const global = (p.flags || '').includes('g');
      const re = cachedRegExp(global ? p.re : `^(?:${p.re})$`, p.flags || '');
      if (global) { out = out.replace(re, (...m) => p.t.replace(/\{(\d+)\}/g, (x, n) => m[n] ?? '')); continue; }
      const m = re.exec(out);
      if (m) return p.t.replace(/\{(\d+)\}/g, (x, n) => m[n] ?? '');
    }
    return out;
  }

  // ------------------------------------------------------------ DOM
  function applyTo(node, root) {
    void root;
    if (node.dataset.i18n) {
      let params;
      if (node.dataset.i18nArgs) { try { params = JSON.parse(node.dataset.i18nArgs); } catch (err) { params = undefined; } }
      node.textContent = tr(node.dataset.i18n, params);
    }
    if (node.dataset.tm !== undefined) node.textContent = tm(node.dataset.tm);
    const attrs = node.dataset.i18nAttr; // estático: "title:chave;aria-label:chave"
    if (attrs) {
      for (const pair of attrs.split(';')) {
        const [attr, key] = pair.split(':');
        if (attr && key) node.setAttribute(attr.trim(), tr(key.trim()));
      }
    }
    if (node.dataset.i18nAttrs) { // dinâmico: {"title": ["chave", {params}]}
      try {
        for (const [attr, [key, params]] of Object.entries(JSON.parse(node.dataset.i18nAttrs))) node.setAttribute(attr, tr(key, params));
      } catch (err) { /* marcação inválida: ignora */ }
    }
  }

  function apply(root = document) {
    const sel = '[data-i18n], [data-tm], [data-i18n-attr], [data-i18n-attrs]';
    if (root.matches && root.matches(sel)) applyTo(root);
    for (const node of root.querySelectorAll(sel)) applyTo(node);
  }

  function bind(el, key, params) {
    el.dataset.i18n = key;
    if (params) el.dataset.i18nArgs = JSON.stringify(params); else delete el.dataset.i18nArgs;
    el.textContent = tr(key, params);
    return el;
  }

  function bindAttr(el, attr, key, params) {
    let map = {};
    try { map = JSON.parse(el.dataset.i18nAttrs || '{}'); } catch (err) { map = {}; }
    map[attr] = [key, params || null];
    el.dataset.i18nAttrs = JSON.stringify(map);
    el.setAttribute(attr, tr(key, params));
    return el;
  }

  function bindTm(el, raw) { // texto do backend que se retraduz quando o idioma muda
    el.dataset.tm = raw;
    el.textContent = tm(raw);
    return el;
  }

  // ------------------------------------------------------------ formatação localizada
  const formatNumber = (value, opts) => new Intl.NumberFormat(lang, opts).format(value);
  const toDate = (value) => (value instanceof Date ? value : new Date(value));
  const formatDate = (value, opts = { dateStyle: 'medium' }) => {
    const d = toDate(value);
    return Number.isNaN(d.getTime()) ? String(value ?? '') : new Intl.DateTimeFormat(lang, opts).format(d);
  };
  const formatDateTime = (value) => formatDate(value, { dateStyle: 'short', timeStyle: 'short' });
  const formatTime = (value) => formatDate(value, { timeStyle: 'medium' });

  function setLanguage(next) {
    if (!SUPPORTED[next] || next === lang) return lang;
    load(next);
    lang = next;
    try { localStorage.setItem(STORAGE_KEY, next); } catch (err) { /* preferência só nesta sessão */ }
    document.documentElement.lang = lang;
    apply();
    document.dispatchEvent(new CustomEvent('enslaved:languagechange', { detail: { lang } }));
    return lang;
  }

  document.documentElement.lang = lang;
  window.tr = tr;
  window.tm = tm;
  window.i18n = {
    SUPPORTED, DEFAULT, STORAGE_KEY, tr, tm, apply, bind, bindAttr, bindTm, setLanguage,
    formatNumber, formatDate, formatDateTime, formatTime,
    get lang() { return lang; },
    get missing() { return [...missing]; },
    catalog: (l = lang) => load(l),
  };
  apply(); // o script vem no fim do body: o DOM estático já existe
})();
