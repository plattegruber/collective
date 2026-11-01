export type ToastTone = 'default' | 'success' | 'error';

export interface Toast {
	id: number;
	message: string;
	tone: ToastTone;
}

const DEFAULT_DURATION_MS = 3600;
let nextId = 1;

class ToastStore {
	items = $state<Toast[]>([]);
	#timers = new Map<number, ReturnType<typeof setTimeout>>();

	push(message: string, tone: ToastTone = 'default', duration = DEFAULT_DURATION_MS): number {
		const id = nextId++;
		this.items = [...this.items, { id, message, tone }];
		if (duration > 0) {
			this.#timers.set(
				id,
				setTimeout(() => this.dismiss(id), duration)
			);
		}
		return id;
	}

	dismiss(id: number): void {
		const timer = this.#timers.get(id);
		if (timer) clearTimeout(timer);
		this.#timers.delete(id);
		this.items = this.items.filter((toast) => toast.id !== id);
	}
}

export const toasts = new ToastStore();
