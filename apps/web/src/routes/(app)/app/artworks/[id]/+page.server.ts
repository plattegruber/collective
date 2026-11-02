import { error, fail, redirect } from '@sveltejs/kit';
import { deleteArtwork, getArtwork } from '#lib/server/artworks.js';
import { deleteArtworkMedia, removeArtworkImage, setPrimaryImage } from '#lib/server/images.js';
import { db, schema } from '#lib/server/db/index.js';
import { and, eq } from 'drizzle-orm';
import type { Actions, PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ parent, params, url, depends }) => {
	depends('app:artwork');
	const { user } = await parent();
	const artwork = await getArtwork(user.id, params.id);
	if (!artwork) error(404, 'Artwork not found');
	return { artwork, created: url.searchParams.has('created') };
};

export const actions: Actions = {
	delete: async ({ locals, params }) => {
		const userId = locals.user!.id;
		await deleteArtworkMedia(userId, params.id);
		const removed = await deleteArtwork(userId, params.id);
		if (!removed) error(404, 'Artwork not found');
		redirect(303, '/app/artworks');
	},
	deleteImage: async ({ locals, params, request }) => {
		const imageId = String((await request.formData()).get('imageId') ?? '');
		const ok = await removeArtworkImage(locals.user!.id, params.id, imageId);
		if (!ok) return fail(404, { message: 'Image not found.' });
		return { ok: true };
	},
	setPrimary: async ({ locals, params, request }) => {
		const imageId = String((await request.formData()).get('imageId') ?? '');
		const ok = await setPrimaryImage(locals.user!.id, params.id, imageId);
		if (!ok) return fail(404, { message: 'Image not found.' });
		return { ok: true };
	},
	setPublic: async ({ locals, params, request }) => {
		const isPublic = String((await request.formData()).get('isPublic')) === 'true';
		await db()
			.update(schema.artworks)
			.set({ isPublic, updatedAt: new Date() })
			.where(and(eq(schema.artworks.id, params.id), eq(schema.artworks.userId, locals.user!.id)));
		return { ok: true };
	}
};
