export type CameraPermission = 'unknown' | 'granted' | 'denied' | 'prompt' | 'unsupported';

export async function queryCameraPermission(): Promise<CameraPermission> {
	if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia)
		return 'unsupported';
	if (!navigator.permissions?.query) return 'unknown';
	try {
		const status = await navigator.permissions.query({ name: 'camera' as PermissionName });
		return status.state;
	} catch {
		return 'unknown';
	}
}

export class CameraError extends Error {
	constructor(
		public readonly kind: 'denied' | 'unavailable' | 'insecure' | 'unknown',
		message: string
	) {
		super(message);
	}
}

/** Opens the rear camera at a modest resolution; the detector only needs 320px anyway. */
export async function openRearCamera(): Promise<MediaStream> {
	if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
		throw new CameraError(
			typeof window !== 'undefined' && !window.isSecureContext ? 'insecure' : 'unavailable',
			'Camera access needs a secure (https) connection and a browser that supports it.'
		);
	}
	try {
		return await navigator.mediaDevices.getUserMedia({
			audio: false,
			video: {
				facingMode: { ideal: 'environment' },
				width: { ideal: 1280 },
				height: { ideal: 720 }
			}
		});
	} catch (cause) {
		const name = cause instanceof DOMException ? cause.name : '';
		if (name === 'NotAllowedError' || name === 'SecurityError') {
			throw new CameraError(
				'denied',
				'Camera permission was denied. Allow camera access in your browser settings.'
			);
		}
		if (name === 'NotFoundError' || name === 'OverconstrainedError') {
			throw new CameraError('unavailable', 'No camera was found on this device.');
		}
		throw new CameraError('unknown', 'The camera could not be started.');
	}
}

export function stopStream(stream: MediaStream | null | undefined): void {
	stream?.getTracks().forEach((track) => track.stop());
}
