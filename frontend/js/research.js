// Investigação profunda na interface: cartão de progresso dentro do fluxo do chat e painel de evidências do relatório.
// Tudo é montado por DOM/textContent: nenhum texto vindo de páginas externas ou do modelo vira HTML.
(function () {
  const T = window.tr; // alias: algumas funções abaixo têm uma variável local chamada tr (linha da tabela)
  const STEPS = ['planning', 'collecting', 'analyzing', 'verifying', 'reporting'].map((key) => ({ key, i18n: `research.step.${key}` }));
  const categoryTip = (cat) => (['A', 'H', 'B', 'C', 'D'].includes(cat) ? T(`research.cat.${cat}`) : cat);

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  const countLabel = (key, n) => T(`research.${key}`, { count: n });

  function countsText(event) {
    if (!event.requests && !event.pages) return '';
    const parts = [countLabel('pages', event.pages || 0), countLabel('requests', event.requests || 0)];
    if (event.evidence) parts.push(countLabel('evidence', event.evidence));
    return parts.join(' · ');
  }

  // ------------------------------------------------------------------ cartão de progresso
  // Um único cartão por investigação, atualizado no lugar. Fica na coluna das mensagens (mesma largura das respostas),
  // sem posição fixa: não cobre o composer nem a barra lateral.
  function createCard() {
    const node = el('article', 'message research-card');
    node.setAttribute('role', 'group');
    node.dataset.i18nAttr = 'aria-label:research.progress';
    node.setAttribute('aria-label', T('research.progress'));
    node.setAttribute('aria-busy', 'true');
    node.dataset.state = 'running';
    const header = el('header');
    header.append(i18n.bind(el('span', 'role'), 'research.role'));
    const stateLabel = i18n.bind(el('span', 'rc-state'), 'research.running');
    header.append(stateLabel);
    const list = el('ol', 'rc-steps');
    const items = STEPS.map((step) => {
      const li = i18n.bind(el('li', 'rc-step'), step.i18n);
      li.dataset.state = 'pending';
      list.append(li);
      return li;
    });
    const counts = el('p', 'rc-counts');
    const detail = el('p', 'rc-detail');
    const live = el('p', 'visually-hidden');
    live.setAttribute('role', 'status');
    live.setAttribute('aria-live', 'polite');
    node.append(header, list, counts, detail, live);
    let current = -1;
    let finished = false;
    let lastCounts = null;
    // os números são texto montado no JS: ao trocar de idioma o cartão (em andamento ou concluído) reescreve a contagem
    const onLanguage = () => {
      if (!node.isConnected) { document.removeEventListener('enslaved:languagechange', onLanguage); return; }
      if (lastCounts) counts.textContent = countsText(lastCounts);
    };
    document.addEventListener('enslaved:languagechange', onLanguage);

    function paint(activeState = 'active') {
      items.forEach((li, i) => {
        li.dataset.state = i < current ? 'done' : i === current ? activeState : 'pending';
        if (i === current && activeState === 'active') li.setAttribute('aria-current', 'step'); else li.removeAttribute('aria-current');
      });
    }

    function update(event) {
      if (finished) return;
      if (event.type === 'research_progress') {
        const idx = STEPS.findIndex((s) => s.key === event.phase);
        if (idx > current || (event.batch && idx >= 2)) {
          current = idx;
          i18n.bind(stateLabel, STEPS[current].i18n);
          live.textContent = `${T(STEPS[current].i18n)}${countsText(event) ? `. ${countsText(event)}` : ''}`; // anunciado só quando a etapa muda
        }
        i18n.bindTm(detail, event.message || '');
      }
      const text = countsText(event);
      if (text) { counts.textContent = text; lastCounts = event; }
      paint();
    }

    function finish(status) {
      if (finished) return;
      finished = true;
      node.setAttribute('aria-busy', 'false');
      node.dataset.state = status;
      if (status === 'done') {
        current = STEPS.length;
        paint();
        i18n.bind(stateLabel, 'research.done');
        detail.textContent = '';
        delete detail.dataset.tm;
        live.textContent = T('research.doneLive');
      } else {
        paint('stopped');
        i18n.bind(stateLabel, status === 'partial' ? 'research.partial' : status === 'error' ? 'research.failed' : 'research.interrupted');
        delete detail.dataset.tm;
        i18n.bind(detail, status === 'partial' ? 'research.partialDetail' : status === 'error' ? 'research.errorDetail' : 'research.interruptedDetail');
        live.textContent = detail.textContent;
      }
    }

    return { node, update, finish, get finished() { return finished; } };
  }

  // ------------------------------------------------------------------ painel de evidências
  const shortUrl = (u) => String(u || '').replace(/^https?:\/\//, '').slice(0, 80);

  function ref(root, id, kind) {
    const a = el('a', 'ref-link', id);
    a.href = '#';
    a.dataset.refTo = id;
    i18n.bindAttr(a, 'aria-label', kind === 'F' ? 'research.viewFinding' : 'research.viewEvidence', { id });
    a.addEventListener('click', (ev) => { ev.preventDefault(); openRef(root, id); });
    return a;
  }

  function openRef(root, id) {
    const target = root.querySelector(`[data-ref="${CSS.escape(id)}"]`);
    if (!target) return;
    for (let p = target.parentElement; p && p !== root; p = p.parentElement) if (p.tagName === 'DETAILS') p.open = true;
    target.scrollIntoView({ block: 'center', behavior: 'smooth' });
    target.classList.add('flash');
    target.setAttribute('tabindex', '-1');
    target.focus({ preventScroll: true });
    setTimeout(() => target.classList.remove('flash'), 1800);
  }

  function refs(root, ids, limit = 8) {
    const box = el('span', 'rp-refs');
    ids.slice(0, limit).forEach((id) => box.append(ref(root, id, id[0])));
    if (ids.length > limit) box.append(el('span', 'rp-more', `+${ids.length - limit}`));
    return box;
  }

  function categoryBadge(cat) {
    const b = el('span', `rp-cat cat-${cat}`, cat);
    b.title = categoryTip(cat);
    b.setAttribute('aria-label', categoryTip(cat));
    return b;
  }

  function groupItem(root, item) {
    const li = el('li', 'rp-item');
    li.dataset.state = item.ok ? 'ok' : 'fail';
    li.append(el('p', 'rp-what', tm(item.what)), el('p', 'rp-result', item.result));
    const pages = el('p', 'rp-pages');
    pages.append(`${countLabel('pages', item.pages.length)}: `, item.pages.slice(0, 3).map(shortUrl).join(', ') + (item.pages.length > 3 ? `, +${item.pages.length - 3}` : ''));
    li.append(pages);
    const supports = el('p', 'rp-supports');
    if (item.supports.length) {
      supports.append(T('research.supports'));
      item.supports.forEach((f) => { supports.append(categoryBadge(f.category), ' '); supports.append(ref(root, f.id, 'F'), ` ${f.title} `); });
    } else {
      supports.textContent = T('research.contextNote');
    }
    li.append(supports);
    if (item.limit) li.append(el('p', 'rp-limit', T('research.limit', { text: tm(item.limit) })));
    const ids = el('p', 'rp-ids');
    ids.append(T('research.records'), refs(root, item.ids));
    li.append(ids);
    return li;
  }

  function matrixTable(matrix) {
    const wrap = el('div', 'table-wrap');
    const table = el('table', 'rp-matrix');
    const head = el('tr');
    head.append(el('th', null, T('research.page')));
    matrix.columns.forEach((c) => head.append(el('th', null, c.label)));
    const thead = el('thead');
    thead.append(head);
    table.append(thead);
    const body = el('tbody');
    matrix.rows.forEach((row) => {
      const tr = el('tr');
      tr.append(el('td', null, shortUrl(row.url)));
      matrix.columns.forEach((c) => {
        const ok = row.present[c.key];
        const td = el('td', ok ? 'yes' : 'no', ok ? '✓' : '—');
        td.setAttribute('aria-label', `${c.label}: ${ok ? T('research.present') : T('research.absent')}`);
        tr.append(td);
      });
      body.append(tr);
    });
    table.append(body);
    wrap.append(table);
    return wrap;
  }

  function ledgerTable(root, ledger) {
    const wrap = el('div', 'table-wrap');
    const table = el('table', 'rp-ledger-table');
    const head = el('tr');
    ['id', 'kind', 'url', 'sha', 'obs', 'findings'].forEach((k) => head.append(el('th', null, T(`research.col.${k}`))));
    const thead = el('thead');
    thead.append(head);
    table.append(thead);
    const body = el('tbody');
    ledger.forEach((e) => {
      const tr = el('tr');
      tr.dataset.ref = e.id;
      tr.dataset.state = e.ok ? 'ok' : 'fail';
      tr.append(el('td', 'rp-id', e.id), el('td', null, `${e.kind}${e.ok ? '' : T('research.failedTag')}`), el('td', 'rp-url', e.url), el('td', 'rp-hash', e.sha256 || '—'), el('td', null, e.observation));
      const td = el('td');
      if (e.findings.length) td.append(refs(root, e.findings)); else td.textContent = '—';
      tr.append(td);
      body.append(tr);
    });
    table.append(body);
    wrap.append(table);
    return wrap;
  }

  function coverageMatrixTable(rows) {
    const wrap = el('div', 'table-wrap');
    const table = el('table', 'rp-coverage-matrix');
    const head = el('tr');
    ['area', 'status', 'scope'].forEach((k) => head.append(el('th', null, T(`research.col.${k}`))));
    const thead = el('thead');
    thead.append(head);
    table.append(thead);
    const body = el('tbody');
    rows.forEach((row) => {
      const tr = el('tr');
      const st = el('span', 'rp-cov-status', tm(row.status));
      st.dataset.status = row.status;
      const tdStatus = el('td');
      tdStatus.append(st);
      tr.append(el('td', null, tm(row.area)), tdStatus, el('td', null, tm(row.detail)));
      body.append(tr);
    });
    table.append(body);
    wrap.append(table);
    return wrap;
  }

  function formatEvidenceText(view) {
    const lines = [`# ${T('research.evTitle')}`];
    if (view.summary && view.summary.length) {
      lines.push('', `## ${T('research.evSummary')}`);
      view.summary.forEach((s) => lines.push(`- ${tm(s.label)}: ${tm(s.value)}`));
    }
    if (view.findings && view.findings.length) {
      lines.push('', `## ${T('research.evFindings')}`);
      view.findings.forEach((f) => {
        lines.push(`- [${f.category}][${f.id}] ${f.title}${f.severity ? ` (${tm(f.severity_label)} ${f.severity})` : ''} — ${f.evidence.join(', ') || T('research.evNoEvidence')}`);
      });
    }
    if (view.coverage_matrix && view.coverage_matrix.length) {
      lines.push('', `## ${T('research.evCoverage')}`);
      view.coverage_matrix.forEach((m) => lines.push(`- ${tm(m.area)}: ${tm(m.status)} — ${tm(m.detail)}`));
    }
    if (view.ledger && view.ledger.length) {
      lines.push('', `## ${T('research.evLedger')}`);
      view.ledger.forEach((e) => {
        lines.push(`- ${e.id} [${e.kind}${e.ok ? '' : T('research.evFailedShort')}] ${e.url} | sha256=${e.sha256 || '—'} | ${e.observation}`);
      });
    }
    return lines.join('\n');
  }

  function buildPanel(view) {
    const root = el('section', 'research-panel');
    root.dataset.i18nAttr = 'aria-label:research.panelAria';
    root.setAttribute('aria-label', T('research.panelAria'));
    root.dataset.status = (view.completion && view.completion.status) || 'complete';
    const head = el('div', 'rp-head');
    const title = i18n.bind(el('h3', 'rp-title'), 'research.panelTitle');
    if (root.dataset.status === 'partial') title.append(i18n.bind(el('span', 'rp-partial'), 'research.partialReport'));
    head.append(title);
    if (window.EnslavedCopy && typeof window.EnslavedCopy.createCopyButton === 'function') {
      const copyBtn = window.EnslavedCopy.createCopyButton({
        labelKey: 'research.copyEvidence',
        ariaKey: 'research.copyEvidenceAria',
        titleKey: 'research.copyEvidenceTitle',
        getText: () => formatEvidenceText(view),
      });
      copyBtn.classList.add('rp-copy-btn');
      head.append(copyBtn);
    }
    root.append(head);
    const dl = el('dl', 'rp-summary');
    view.summary.forEach((s) => {
      const box = el('div');
      box.append(el('dt', null, tm(s.label)), el('dd', null, tm(s.value)));
      dl.append(box);
    });
    root.append(dl);
    if (view.findings.length) {
      const list = el('ul', 'rp-findings');
      view.findings.forEach((f) => {
        const li = el('li', 'rp-finding');
        li.dataset.ref = f.id;
        const fhead = el('p', 'rp-finding-head');
        fhead.append(categoryBadge(f.category), ' ', el('strong', null, `${f.id} ${f.title}`), f.severity ? el('span', 'rp-sev', ` · ${tm(f.severity_label)} ${f.severity}`) : '');
        li.append(fhead);
        if (f.proof) li.append(el('p', 'rp-proof', T('research.directProof', { text: f.proof })));
        if (f.pages.length) li.append(el('p', 'rp-pages', T('research.pagesLabel', { list: `${f.pages.slice(0, 3).map(shortUrl).join(', ')}${f.pages.length > 3 ? `, +${f.pages.length - 3}` : ''}` })));
        if (f.evidence.length) { const ev = el('p', 'rp-ids'); ev.append(T('research.evidences'), refs(root, f.evidence)); li.append(ev); }
        list.append(li);
      });
      root.append(el('h4', 'rp-sub', T('research.findingsAndEvidence')), list);
    }
    if (view.coverage_matrix && view.coverage_matrix.length) {
      root.append(el('h4', 'rp-sub', T('research.coverageTitle')), coverageMatrixTable(view.coverage_matrix));
    }
    if (view.header_matrix && view.header_matrix.rows.length) {
      root.append(el('h4', 'rp-sub', T('research.headersTitle')), matrixTable(view.header_matrix));
    }
    view.groups.forEach((g) => {
      const d = el('details', 'rp-group');
      d.append(el('summary', null, `${tm(g.label)} · ${countLabel('records', g.evidence_count)}${g.items.length < g.evidence_count ? T('research.groupedIn', { items: countLabel('grouped', g.items.length) }) : ''}`));
      const ul = el('ul', 'rp-items');
      g.items.forEach((item) => ul.append(groupItem(root, item)));
      d.append(ul);
      root.append(d);
    });
    const tech = el('details', 'rp-ledger');
    tech.append(el('summary', null, T('research.viewTech', { count: view.ledger.length })), ledgerTable(root, view.ledger));
    root.append(tech);
    return root;
  }

  // Transforma F#/E# do texto do relatório em links para o painel (só ids que existem; nunca altera o texto salvo).
  function linkRefs(content, root) {
    const known = new Set([...root.querySelectorAll('[data-ref]')].map((n) => n.dataset.ref));
    const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (n.parentElement && n.parentElement.closest('a, code, pre') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const textNode of nodes) {
      const text = textNode.nodeValue;
      if (!/\b[EF]\d{1,3}\b/.test(text)) continue;
      const frag = document.createDocumentFragment();
      let last = 0;
      for (const m of text.matchAll(/\b([EF]\d{1,3})\b/g)) {
        if (!known.has(m[1])) continue;
        frag.append(text.slice(last, m.index), ref(root, m[1], m[1][0]));
        last = m.index + m[0].length;
      }
      if (last === 0) continue;
      frag.append(text.slice(last));
      textNode.replaceWith(frag);
    }
  }

  // Anexa o painel ao balão do assistente (uma vez) e devolve `link()` para ligar F#/E# após cada renderização do texto.
  function attach(bubble, view) {
    if (!view || !Array.isArray(view.ledger) || !view.ledger.length) return { link() {} };
    let root = bubble.querySelector(':scope > .research-panel');
    if (!root) {
      root = buildPanel(view);
      bubble.insertBefore(root, bubble.querySelector(':scope > .msg-footer')); // o rodapé de cópia fica depois do painel
    }
    return { root, link() { const content = bubble.querySelector('.content'); if (content) linkRefs(content, root); } };
  }

  window.EnslavedResearch = { createCard, attach, STEPS };
})();
