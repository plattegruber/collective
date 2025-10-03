import { getOrCreateAnonId } from './identity';

export const REACTION_EMOJIS = ['❤️', '👀', '🤔', '😮', '😂', '🔥'];

export interface ReactionCount {
  emoji: string;
  count: number;
}

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

export async function fetchReactionCounts(pieceId): Promise<ReactionCount[]> {
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
  if (!Array.isArray(data)) {
    return [];
  }

  return data
    .map((item) => {
      const count = Number(item?.count ?? 0);
      const emoji = typeof item?.emoji === 'string' ? item.emoji : '';
      return { emoji, count };
    }) as ReactionCount[];
}

export async function fetchCountsForPieces(pieceIds: string[], concurrency = 4): Promise<Map<string, ReactionCount[]>> {
  const ids = pieceIds.filter((id) => typeof id === 'string' && id.trim().length > 0);
  const limit = Math.max(1, Math.min(concurrency, 8));
  const queue = [...ids];
  const results = new Map<string, ReactionCount[]>();

  async function worker() {
    while (queue.length) {
      const nextId = queue.shift();
      if (!nextId) continue;
      try {
        const counts = await fetchReactionCounts(nextId);
        results.set(nextId, counts);
      } catch (error) {
        console.warn('Failed to load reaction counts for', nextId, error);
      }
    }
  }

  const workers = Array.from({ length: limit }, () => worker());
  await Promise.all(workers);

  return results;
}
