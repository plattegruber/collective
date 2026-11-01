import { redirect } from '@sveltejs/kit';
import { clearSessionCookie, invalidateSession } from '#lib/server/auth/index.js';
import type { Actions } from './$types';

export const actions: Actions = {
	default: async ({ locals, cookies }) => {
		if (locals.sessionId) await invalidateSession(locals.sessionId);
		clearSessionCookie(cookies);
		redirect(303, '/');
	}
};
