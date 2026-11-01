import { redirect } from '@sveltejs/kit';
import { artworkFormSchema } from '#lib/schemas/artwork.js';
import { createArtwork, listTags, validateMoney } from '#lib/server/artworks.js';
import { fieldFailure, parseForm } from '#lib/server/forms.js';
import type { Actions, PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ parent }) => {
	const { user } = await parent();
	const tags = await listTags(user.id);
	return { tagSuggestions: tags.map((tag) => tag.name) };
};

export const actions: Actions = {
	default: async ({ request, locals }) => {
		const formData = await request.formData();
		const parsed = parseForm(formData, artworkFormSchema);
		if (!parsed.ok) return parsed.failure;
		const moneyErrors = validateMoney(parsed.data);
		if (Object.keys(moneyErrors).length > 0) return fieldFailure(moneyErrors, formData);

		const id = await createArtwork(locals.user!.id, parsed.data);
		redirect(303, `/app/artworks/${id}?created=1`);
	}
};
