import { requireUser } from '#lib/server/auth/guard.js';
import type { LayoutServerLoad } from './$types';

export const load: LayoutServerLoad = (event) => {
	const user = requireUser(event);
	return { user };
};
