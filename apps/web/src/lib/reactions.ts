import { getOrCreateAnonId } from './identity';

export const REACTION_EMOJIS = ['❤️', '👀', '🤔', '😮', '😂', '🔥'];

const DEFAULT_BASE = 'https://gg-reactions.gruberplatte.workers.dev';
const WORKER_BASE = (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_REACTIONS_BASE)
  ? String(import.meta.env.VITE_REACTIONS_BASE).replace(/\/$/, '')
  : DEFAULT_BASE;

async function parseJson(response) {
  try {
    return await response.json();
  } catch (error) {
    return null;
  }
}

export async function sendReaction(pieceId, emoji) {
  const uid = getOrCreateAnonId();
  const payload = {
    pieceId,
    emoji,
    uid,
    hp: '',
    ts: Date.now(),
  };

  const response = await fetch(`${WORKER_BASE}/react`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
    credentials: 'omit',
    mode: 'cors',
  });

  const data = await parseJson(response);

  if (!response.ok) {
    const message = data?.error ?? `Reaction failed (${response.status})`;
    const error = new Error(message);
    error.status = response.status;
    error.payload = data;
    throw error;
  }

  return data ?? { ok: true };
}

export async function fetchReactionCounts(pieceId) {
  const response = await fetch(`${WORKER_BASE}/counts?piece=${encodeURIComponent(pieceId)}`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
    credentials: 'omit',
    mode: 'cors',
  });

  const data = await parseJson(response);
  if (!response.ok) {
    const message = data?.error ?? `Counts failed (${response.status})`;
    const error = new Error(message);
    error.status = response.status;
    error.payload = data;
    throw error;
  }

  return Array.isArray(data) ? data : [];
}
