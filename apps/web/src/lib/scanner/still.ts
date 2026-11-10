import { cropToBlob } from '#lib/recognition/frames.js';

/** The edge the hand-off still is capped at; sessionStorage must hold it as a data URL. */
export const STILL_MAX_EDGE = 2400;

interface PhotoCapturer {
	takePhoto(): Promise<Blob>;
}
declare const ImageCapture: (new (track: MediaStreamTrack) => PhotoCapturer) | undefined;

/**
 * A still of what the scanner sees, as sharp as the device allows: the camera's photo
 * pipeline through `ImageCapture` where it exists (Android Chrome), else the current video
 * frame. Returned whole; the studio finds the piece in it again at full resolution.
 */
export async function takeStill(stream: MediaStream, video: HTMLVideoElement): Promise<Blob> {
	const track = stream.getVideoTracks()[0];
	if (track && typeof ImageCapture !== 'undefined') {
		try {
			const photo = await new ImageCapture(track).takePhoto();
			const bitmap = await createImageBitmap(photo, { imageOrientation: 'from-image' });
			try {
				return await cropToBlob(
					bitmap,
					{ x1: 0, y1: 0, x2: bitmap.width, y2: bitmap.height },
					STILL_MAX_EDGE,
					0.92
				);
			} finally {
				bitmap.close();
			}
		} catch (cause) {
			console.warn('ImageCapture failed; using the video frame.', cause);
		}
	}
	return cropToBlob(
		video,
		{ x1: 0, y1: 0, x2: video.videoWidth, y2: video.videoHeight },
		STILL_MAX_EDGE,
		0.92
	);
}
