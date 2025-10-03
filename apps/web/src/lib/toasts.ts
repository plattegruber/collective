import { writable } from 'svelte/store';

export type ToastTone = 'default' | 'error' | 'success';

export interface ToastOptions {
  tone?: ToastTone;
  duration?: number;
}

export interface ToastEntry {
  id: string;
  message: string;
  tone: ToastTone;
  duration: number;
}

let counter = 0;

function makeId() {
  counter = (counter + 1) % Number.MAX_SAFE_INTEGER;
  return `${Date.now()}-${counter}`;
}

const DEFAULT_DURATION = 3200;

export const toasts = writable<ToastEntry[]>([]);

export function pushToast(message: string, options: ToastOptions = {}) {
  if (!message) return '';
  const id = makeId();
  const entry: ToastEntry = {
    id,
    message,
    tone: options.tone ?? 'default',
    duration: Math.max(0, options.duration ?? DEFAULT_DURATION),
  };
  toasts.update((list) => [...list, entry]);
  return id;
}

export function removeToast(id: string) {
  if (!id) return;
  toasts.update((list) => list.filter((item) => item.id !== id));
}

export function clearToasts() {
  toasts.set([]);
}
