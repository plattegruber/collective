import { redirect } from '@sveltejs/kit';
import { env } from 'cloudflare:workers';
import { signupSchema } from '#lib/schemas/auth.js';
import { createSession, setSessionCookie } from '#lib/server/auth/index.js';
import { formFailure, parseForm } from '#lib/server/forms.js';
import { bumpCounter, minuteBucket } from '#lib/server/kv.js';
import { createUser, findUserByEmail } from '#lib/server/users.js';
import type { Actions, PageServerLoad } from './$types';

/** Set the INVITE_CODE secret to make signup invite-only. */
function inviteCode(): string | undefined {
	const value = (env as unknown as { INVITE_CODE?: string }).INVITE_CODE;
	return value && value.length > 0 ? value : undefined;
}

export const load: PageServerLoad = ({ locals }) => {
	if (locals.user) redirect(303, '/app');
	return { inviteRequired: Boolean(inviteCode()) };
};

export const actions: Actions = {
	default: async ({ request, cookies, getClientAddress }) => {
		const formData = await request.formData();
		const parsed = parseForm(formData, signupSchema);
		if (!parsed.ok) return parsed.failure;

		const ip = getClientAddress();
		if (!(await bumpCounter(`rl:signup:${ip}:${minuteBucket()}`, 600, 5))) {
			return formFailure(429, 'Too many sign-ups from this network. Try again later.', formData);
		}

		const required = inviteCode();
		if (required && parsed.data.invite !== required) {
			return formFailure(400, 'That invite code is not valid.', formData, {
				invite: 'Check the code and try again.'
			});
		}

		if (await findUserByEmail(parsed.data.email)) {
			return formFailure(400, 'An account with that email already exists.', formData, {
				email: 'Try signing in instead.'
			});
		}

		const userId = await createUser(parsed.data);
		const { token, expiresAt } = await createSession(userId);
		setSessionCookie(cookies, token, expiresAt);
		redirect(303, '/app?welcome=1');
	}
};
