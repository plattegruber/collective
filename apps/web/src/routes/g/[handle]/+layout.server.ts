import { error } from '@sveltejs/kit';
import { findUserByHandle } from '#lib/server/users.js';
import type { LayoutServerLoad } from './$types';

export const load: LayoutServerLoad = async ({ params, locals }) => {
	const owner = await findUserByHandle(params.handle);
	const isOwner = Boolean(owner && locals.user?.id === owner.id);
	if (!owner || (!owner.galleryPublic && !isOwner)) error(404, 'Gallery not found');
	return {
		gallery: {
			ownerId: owner.id,
			handle: owner.handle!,
			title: owner.galleryTitle || `${owner.name}’s collection`,
			intro: owner.galleryIntro,
			ownerName: owner.name,
			isPublic: owner.galleryPublic,
			isOwner
		}
	};
};
