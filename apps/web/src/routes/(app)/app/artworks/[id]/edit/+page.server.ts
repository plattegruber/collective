import { error, redirect } from '@sveltejs/kit';
import { artworkFormSchema } from '#lib/schemas/artwork.js';
import { getArtwork, listTags, updateArtwork, validateMoney } from '#lib/server/artworks.js';
import { fieldFailure, parseForm } from '#lib/server/forms.js';
import { artworkToFormValues } from '#lib/artworkFormValues.js';
import type { Actions, PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ parent, params }) => {
	const { user } = await parent();
	const [artwork, tags] = await Promise.all([getArtwork(user.id, params.id), listTags(user.id)]);
	if (!artwork) error(404, 'Artwork not found');
	return {
		title: artwork.title,
		values: artworkToFormValues(artwork),
		tagSuggestions: tags.map((tag) => tag.name)
	};
};

export const actions: Actions = {
	default: async ({ request, locals, params }) => {
		const formData = await request.formData();
		const parsed = parseForm(formData, artworkFormSchema);
		if (!parsed.ok) return parsed.failure;
		const moneyErrors = validateMoney(parsed.data);
		if (Object.keys(moneyErrors).length > 0) return fieldFailure(moneyErrors, formData);

		const ok = await updateArtwork(locals.user!.id, params.id, parsed.data);
		if (!ok) error(404, 'Artwork not found');
		redirect(303, `/app/artworks/${params.id}`);
	}
};
