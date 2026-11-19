// Cena cômica do chat: só troca o estado (data-mood) do SVG; a animação é CSS.
// Expõe window.EnslavedScene para app.js. Preferência de mostrar/ocultar fica no localStorage.
(() => {
  const root = document.getElementById('scene');
  const toggle = document.getElementById('scene-toggle');
  if (!root || !toggle) return;

  const KEY = 'enslaved.scene';
  // 'done' (joinha) permanece até o usuário digitar ou sair da conversa (reset); só erro/cancelamento são passageiros
  const DWELL = { error: 2500, cancel: 1500 };
  const TYPING_IDLE_MS = 1500;
  let timer = null;
  let typingTimer = null;
  let mood = 'idle';
  let busy = false;

  function set(next, dwell) {
    clearTimeout(timer);
    clearTimeout(typingTimer);
    mood = next;
    root.dataset.mood = next;
    if (dwell) timer = setTimeout(() => set('idle'), dwell);
  }

  function show(on) {
    root.hidden = !on;
    toggle.checked = on;
    root.setAttribute('aria-hidden', 'true');
    if (!on) set('idle');
  }

  let remembered = true;
  try { remembered = localStorage.getItem(KEY) !== 'off'; } catch (err) { /* sem armazenamento: padrão ligado */ }
  show(remembered);

  toggle.addEventListener('change', () => {
    show(toggle.checked);
    try { localStorage.setItem(KEY, toggle.checked ? 'on' : 'off'); } catch (err) { /* preferência só nesta sessão */ }
  });

  window.EnslavedScene = {
    // chamado a cada tecla: digitando enquanto há texto e um pouco depois do último toque
    typing(hasText) {
      if (busy) return;
      if (!hasText) { if (mood === 'typing') set('idle'); return; }
      set('typing');
      typingTimer = setTimeout(() => { if (mood === 'typing') set('idle'); }, TYPING_IDLE_MS);
    },
    busy() {
      busy = true;
      set('busy');
    },
    // sair da conversa atual (nova conversa / abrir outra) volta ao repouso; não interrompe uma resposta em andamento
    reset() {
      if (!busy) set('idle');
    },
    // outcome: 'done' | 'error' | 'cancel' | 'pending' (aguardando autorização de ferramenta)
    finish(outcome) {
      busy = false;
      if (outcome === 'pending') set('idle');
      else set(['done', 'error', 'cancel'].includes(outcome) ? outcome : 'idle', DWELL[outcome]);
    },
  };
})();
