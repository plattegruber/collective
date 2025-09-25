export interface Env {
  REACTIONS_KV: KVNamespace;
  ALLOWED_ORIGIN: string;
}

const EMOJI_SET = new Set(["❤️", "👀", "🤔", "😮", "😂", "🔥"]);

function cors(origin: string) {
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "content-type",
  };
}

async function json(env: Env, data: unknown, status = 200): Promise<Response> {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...cors(env.ALLOWED_ORIGIN) },
  });
}

function isLegitUA(ua: string | null) {
  if (!ua) return false;
  const bad = /(curl|wget|python-requests|bot|crawler|spider|headless|scrapy)/i;
  return !bad.test(ua);
}

const now = () => Date.now();
const minuteBucket = () => Math.floor(now() / 60000);
const dayStamp = () => new Date().toISOString().slice(0, 10);

const ipMinuteKey = (ip: string) => `rl:ip:${ip}:${minuteBucket()}`;
const ipDayKey = (ip: string) => `rl:ipday:${ip}:${dayStamp()}`;
const uidSeenKey = (uid: string, piece: string) => `seen:${piece}:${uid}`;

async function bump(env: Env, key: string, ttl: number, max: number) {
  const v = Number(await env.REACTIONS_KV.get(key)) || 0;
  if (v >= max) return false;
  await env.REACTIONS_KV.put(key, String(v + 1), { expirationTtl: ttl });
  return true;
}

function safePieceId(raw: string) {
  const ok = raw.replace(/[^a-zA-Z0-9_\-\.]/g, "_");
  return ok.slice(0, 128);
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);

    if (req.method === "OPTIONS") {
      return new Response("ok", { headers: cors(env.ALLOWED_ORIGIN) });
    }

    const origin = req.headers.get("Origin");
    if (origin && origin !== env.ALLOWED_ORIGIN) {
      return new Response("forbidden", { status: 403, headers: cors(env.ALLOWED_ORIGIN) });
    }

    if (req.method === "GET" && url.pathname === "/counts") {
      const piece = url.searchParams.get("piece");
      if (!piece) return json(env, { error: "missing piece" }, 400);
      const pid = safePieceId(piece);

      const keys = Array.from(EMOJI_SET).map((e) => `count:${pid}:${e}`);
      const vals = await Promise.all(keys.map((k) => env.REACTIONS_KV.get(k)));
      const payload = Array.from(EMOJI_SET).map((e, i) => ({ emoji: e, count: Number(vals[i] ?? 0) }));
      return json(env, payload);
    }

    if (req.method === "POST" && url.pathname === "/react") {
      const ip = req.headers.get("CF-Connecting-IP") ?? "0.0.0.0";
      const ua = req.headers.get("User-Agent");

      if (!isLegitUA(ua)) {
        return json(env, { error: "suspicious" }, 429);
      }

      let body: any = null;
      try {
        body = await req.json();
      } catch {}
      if (!body) return json(env, { error: "bad_json" }, 400);

      const { pieceId, emoji, uid, ts, hp } = body as {
        pieceId: string;
        emoji: string;
        uid: string;
        ts?: number;
        hp?: string;
      };

      if (typeof hp === "string" && hp.trim().length > 0) {
        return json(env, { ok: true }, 200);
      }

      if (!pieceId || !uid || !emoji || !EMOJI_SET.has(emoji)) {
        return json(env, { error: "missing_or_invalid_fields" }, 400);
      }

      const pid = safePieceId(pieceId);

      const seenK = uidSeenKey(uid, pid);
      const seenValue = await env.REACTIONS_KV.get(seenK);
      const lastSeen = seenValue ? Number(seenValue) : null;
      if (lastSeen !== null && Number.isFinite(lastSeen) && now() - lastSeen < 10_000) {
        return json(env, { ok: true, throttled: true }, 200);
      }
      await env.REACTIONS_KV.put(seenK, String(now()), { expirationTtl: 120 });

      const minuteOk = await bump(env, ipMinuteKey(ip), 70, 10);
      const dayOk = await bump(env, ipDayKey(ip), 60 * 60 * 24, 200);
      if (!minuteOk || !dayOk) {
        return json(env, { error: "rate_limited" }, 429);
      }

      const countKey = `count:${pid}:${emoji}`;
      const current = Number((await env.REACTIONS_KV.get(countKey)) ?? 0);
      await env.REACTIONS_KV.put(countKey, String(current + 1));
      return json(env, { ok: true }, 200);
    }

    if (req.method === "GET" && url.pathname === "/health") {
      return new Response("ok", { status: 200, headers: cors(env.ALLOWED_ORIGIN) });
    }

    return new Response("not found", { status: 404, headers: cors(env.ALLOWED_ORIGIN) });
  },
};
