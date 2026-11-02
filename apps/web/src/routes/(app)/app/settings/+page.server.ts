import { fail, redirect } from '@sveltejs/kit';
import { eq } from 'drizzle-orm';
import { db, schema } from '#lib/server/db/index.js';
import {
	clearSessionCookie,
	hashPassword,
	invalidateOtherSessions,
	invalidateUserSessions,
	verifyPassword
} from '#lib/server/auth/index.js';
import { createApiToken, listApiTokens, revokeApiToken } from '#lib/server/apiTokens.js';
import { activeRelease, indexStatus, releaseManifest } from '#lib/server/recognition.js';
import { deleteUserAndData, findUserByHandle } from '#lib/server/users.js';
import { formFailure, parseForm } from '#lib/server/forms.js';
import {
	deleteAccountSchema,
	gallerySchema,
	passwordChangeSchema,
	profileSchema,
	tokenSchema
} from '#lib/schemas/settings.js';
import { collectionStats } from '#lib/server/artworks.js';
import type { Actions, PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ parent }) => {
	const { user } = await parent();
	const [tokens, stats, release] = await Promise.all([
		listApiTokens(user.id),
		collectionStats(user.id),
		activeRelease()
	]);
	const recognizer = release
		? { manifest: releaseManifest(release), status: await indexStatus(user.id, release.id) }
		: null;
	return { tokens, stats, recognizer };
};

export const actions: Actions = {
	profile: async ({ request, locals }) => {
		const formData = await request.formData();
		const parsed = parseForm(formData, profileSchema);
		if (!parsed.ok) return parsed.failure;
		await db()
			.update(schema.users)
			.set({ name: parsed.data.name, updatedAt: new Date() })
			.where(eq(schema.users.id, locals.user!.id));
		return { profileSaved: true };
	},

	gallery: async ({ request, locals }) => {
		const formData = await request.formData();
		const parsed = parseForm(formData, gallerySchema);
		if (!parsed.ok) return parsed.failure;
		const taken = await findUserByHandle(parsed.data.handle);
		if (taken && taken.id !== locals.user!.id) {
			return formFailure(400, 'That handle is taken.', formData, {
				handle: 'Choose another handle.'
			});
		}
		await db()
			.update(schema.users)
			.set({
				handle: parsed.data.handle,
				galleryPublic: parsed.data.galleryPublic,
				galleryTitle: parsed.data.galleryTitle ?? null,
				galleryIntro: parsed.data.galleryIntro ?? null,
				updatedAt: new Date()
			})
			.where(eq(schema.users.id, locals.user!.id));
		return { gallerySaved: true };
	},

	password: async ({ request, locals }) => {
		const formData = await request.formData();
		const parsed = parseForm(formData, passwordChangeSchema);
		if (!parsed.ok) return parsed.failure;
		const user = await db().query.users.findFirst({ where: eq(schema.users.id, locals.user!.id) });
		if (!user || !(await verifyPassword(parsed.data.current, user.passwordHash))) {
			return formFailure(400, 'Your current password is not right.', formData, {
				current: 'Check and try again.'
			});
		}
		await db()
			.update(schema.users)
			.set({ passwordHash: await hashPassword(parsed.data.password), updatedAt: new Date() })
			.where(eq(schema.users.id, user.id));
		if (locals.sessionId) await invalidateOtherSessions(user.id, locals.sessionId);
		else await invalidateUserSessions(user.id);
		return { passwordChanged: true };
	},

	createToken: async ({ request, locals }) => {
		const formData = await request.formData();
		const parsed = parseForm(formData, tokenSchema);
		if (!parsed.ok) return parsed.failure;
		const { token } = await createApiToken(
			locals.user!.id,
			parsed.data.name ?? 'Training pipeline'
		);
		return { newToken: token };
	},

	revokeToken: async ({ request, locals }) => {
		const id = String((await request.formData()).get('id') ?? '');
		const ok = await revokeApiToken(locals.user!.id, id);
		return ok ? { tokenRevoked: true } : fail(404, { message: 'Token not found.' });
	},

	deleteAccount: async ({ request, locals, cookies }) => {
		const formData = await request.formData();
		const parsed = parseForm(formData, deleteAccountSchema);
		if (!parsed.ok) return parsed.failure;
		const user = await db().query.users.findFirst({ where: eq(schema.users.id, locals.user!.id) });
		if (!user || !(await verifyPassword(parsed.data.password, user.passwordHash))) {
			return formFailure(400, 'Your password is not right.', formData, {
				password: 'Check and try again.'
			});
		}
		await deleteUserAndData(user.id);
		clearSessionCookie(cookies);
		redirect(303, '/?deleted=1');
	}
};
