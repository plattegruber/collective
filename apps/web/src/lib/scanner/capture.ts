/**
 * Hands a frame crop from the scanner to the "add artwork" flow through sessionStorage,
 * so an unknown piece on the wall becomes a new entry with its first photo in two taps.
 */
const KEY = 'collective.scanCapture';

export interface ScanCapture {
	dataUrl: string;
	capturedAt: number;
}

export async function storeCapture(blob: Blob): Promise<void> {
	const dataUrl = await new Promise<string>((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(String(reader.result));
		reader.onerror = () => reject(reader.error ?? new Error('Could not read the capture.'));
		reader.readAsDataURL(blob);
	});
	sessionStorage.setItem(KEY, JSON.stringify({ dataUrl, capturedAt: Date.now() }));
}

export function peekCapture(): ScanCapture | null {
	if (typeof sessionStorage === 'undefined') return null;
	try {
		const raw = sessionStorage.getItem(KEY);
		if (!raw) return null;
		const parsed = JSON.parse(raw) as ScanCapture;
		// Stale captures (older than an hour) are discarded rather than attached by surprise.
		if (Date.now() - parsed.capturedAt > 60 * 60 * 1000) {
			sessionStorage.removeItem(KEY);
			return null;
		}
		return parsed;
	} catch {
		return null;
	}
}

export function clearCapture(): void {
	sessionStorage.removeItem(KEY);
}

export async function captureToFile(capture: ScanCapture): Promise<File> {
	const blob = await (await fetch(capture.dataUrl)).blob();
	return new File([blob], 'scanned.jpg', { type: blob.type || 'image/jpeg' });
}
