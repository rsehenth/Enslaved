const api = {
  async request(path, options = {}) {
    const response = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try {
        const data = await response.json();
        message = data.error || (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data));
      } catch (err) { /* corpo sem JSON */ }
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    return response.json();
  },
  get(path) { return this.request(path); },
  post(path, body) { return this.request(path, { method: 'POST', body: JSON.stringify(body) }); },
  patch(path, body) { return this.request(path, { method: 'PATCH', body: JSON.stringify(body) }); },
  del(path) { return this.request(path, { method: 'DELETE' }); },

  async uploadAttachment(file) {
    const response = await fetch('/api/attachments', {
      method: 'POST',
      headers: { 'X-Filename': encodeURIComponent(file.name || 'image.png') },
      body: file,
    });
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try {
        const data = await response.json();
        message = data.error || (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data));
      } catch (err) { /* sem JSON */ }
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    return response.json();
  },

  async chatStream(body, onEvent, signal) {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify(body),
      signal,
    });
    if (!response.ok || !response.body) {
      let message = tr('stream.fail', { status: response.status });
      try {
        const data = await response.json();
        message = data.error || data.detail || message;
      } catch (err) { /* sem corpo */ }
      throw new Error(message);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let index;
      while ((index = buffer.indexOf('\n\n')) >= 0) {
        const rawEvent = buffer.slice(0, index);
        buffer = buffer.slice(index + 2);
        for (const line of rawEvent.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          try {
            onEvent(JSON.parse(payload));
          } catch (err) {
            console.error('Evento SSE inválido', err);
          }
        }
      }
    }
  },
};
