// Renderização segura de respostas: marked (GFM) + DOMPurify, ambos locais em /static/vendor.
// O texto original nunca é alterado; só o DOM exibido é derivado dele.
(function () {
  const ALLOWED_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'br', 'hr', 'strong', 'em', 'del', 'code', 'pre',
    'blockquote', 'ul', 'ol', 'li', 'a', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'input', 'span', 'sup', 'sub'];
  const ALLOWED_ATTR = ['href', 'title', 'class', 'colspan', 'rowspan', 'align', 'start', 'type', 'checked', 'disabled'];
  const TABLE_ROW = /^\s*\|.*\|?\s*$/;

  // Tabela GFM inválida (sem linha delimitadora, colunas incoerentes) vira bloco literal rolável: nada é inventado.
  function looksLikePipeTable(text) {
    return text.split('\n').filter((line) => TABLE_ROW.test(line)).length >= 2;
  }

  marked.use({
    gfm: true,
    breaks: true,
    walkTokens(token) {
      if (token.type === 'paragraph' && looksLikePipeTable(token.text)) {
        token.type = 'code';
        token.lang = 'tabela';
        token.text = token.raw.trim();
      }
    },
  });

  DOMPurify.addHook('afterSanitizeAttributes', (node) => {
    if (node.tagName === 'A') {
      node.setAttribute('target', '_blank');
      node.setAttribute('rel', 'noopener noreferrer');
    }
    if (node.tagName === 'INPUT' && node.getAttribute('type') !== 'checkbox') node.remove();
  });

  // Markdown incompleto durante o streaming: fecha a cerca de código aberta só para a exibição.
  function healPartial(source) {
    const fences = (source.match(/^ {0,3}(```|~~~)/gm) || []).length;
    return fences % 2 ? `${source}\n\`\`\`` : source;
  }

  function render(source, { partial = false } = {}) {
    const text = String(source || '');
    const html = marked.parse(partial ? healPartial(text) : text, { async: false });
    const fragment = DOMPurify.sanitize(html, {
      ALLOWED_TAGS,
      ALLOWED_ATTR,
      ALLOWED_URI_REGEXP: /^(?:https?|mailto):/i,
      RETURN_DOM_FRAGMENT: true,
    });
    for (const table of fragment.querySelectorAll('table')) {
      const wrap = document.createElement('div');
      wrap.className = 'table-wrap';
      table.replaceWith(wrap);
      wrap.append(table);
    }
    for (const pre of fragment.querySelectorAll('pre')) {
      const code = pre.querySelector('code');
      const lang = code && /language-([\w-]+)/.exec(code.className || '');
      pre.classList.add('code');
      if (lang) pre.dataset.lang = lang[1];
    }
    return fragment;
  }

  window.EnslavedMarkdown = { render };
})();
