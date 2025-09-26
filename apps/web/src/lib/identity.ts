const STORAGE_KEY = 'gg.uid';

function createFallbackId() {
  const random = Math.random().toString(36).slice(2, 12);
  const timestamp = Date.now().toString(36);
  return `anon-${timestamp}-${random}`;
}

export function getOrCreateAnonId() {
  if (typeof window === 'undefined') {
    return createFallbackId();
  }

  try {
    const existing = window.localStorage.getItem(STORAGE_KEY);
    if (existing) {
      return existing;
    }

    const uid = typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : createFallbackId();
    window.localStorage.setItem(STORAGE_KEY, uid);

    try {
      const maxAge = 60 * 60 * 24 * 365 * 2; // two years
      document.cookie = `gg_uid=${uid}; Path=/; SameSite=Lax; Max-Age=${maxAge}`;
    } catch (error) {
      console.warn('Failed to set gg_uid cookie', error);
    }

    return uid;
  } catch (error) {
    console.warn('Unable to read/write anon id; falling back to random id.', error);
    return createFallbackId();
  }
}
