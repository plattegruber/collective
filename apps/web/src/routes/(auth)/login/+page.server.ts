import { redirect } from '@sveltejs/kit';
import { loginSchema } from '#lib/schemas/auth.js';
import { createSession, setSessionCookie, verifyPassword } from '#lib/server/auth/index.js';
import { safeNextPath } from '#lib/server/auth/guard.js';
import { formFailure, parseForm } from '#lib/server/forms.js';
import { bumpCounter, minuteBucket } from '#lib/server/kv.js';
import { findUserByEmail } from '#lib/server/users.js';
import type { Actions, PageServerLoad } from './$types';

export const load: PageServerLoad = ({ locals, url }) => {
	if (locals.user) redirect(303, safeNextPath(url.searchParams.get('next')));
	return { next: safeNextPath(url.searchParams.get('next')) };
};

export const actions: Actions = {
	default: async ({ request, cookies, getClientAddress }) => {
		const formData = await request.formData();
		const parsed = parseForm(formData, loginSchema);
		if (!parsed.ok) return parsed.failure;

		const ip = getClientAddress();
		if (!(await bumpCounter(`rl:login:${ip}:${minuteBucket()}`, 120, 20))) {
			return formFailure(429, 'Too many attempts. Try again in a minute.', formData);
		}

		const user = await findUserByEmail(parsed.data.email);
		const valid = user ? await verifyPassword(parsed.data.password, user.passwordHash) : false;
		if (!user || !valid) {
			return formFailure(400, 'That email and password do not match.', formData);
		}

		const { token, expiresAt } = await createSession(user.id);
		setSessionCookie(cookies, token, expiresAt);
		redirect(303, safeNextPath(parsed.data.next));
	}
};
