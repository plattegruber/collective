import { z } from 'zod';

export const emailSchema = z.email('Enter a valid email address.').max(254);
export const passwordSchema = z
	.string('Enter a password.')
	.min(10, 'Use at least 10 characters.')
	.max(200, 'That password is too long.');

export const signupSchema = z.object({
	name: z.string('Tell us what to call you.').min(1, 'Tell us what to call you.').max(80),
	email: emailSchema,
	password: passwordSchema,
	invite: z.string().optional()
});

export const loginSchema = z.object({
	email: emailSchema,
	password: z.string('Enter your password.').min(1, 'Enter your password.'),
	next: z.string().optional()
});

export const HANDLE_PATTERN = /^[a-z0-9](?:[a-z0-9-]{1,28}[a-z0-9])?$/;
export const RESERVED_HANDLES = new Set([
	'app',
	'api',
	'g',
	'login',
	'signup',
	'logout',
	'media',
	'admin',
	'settings',
	'static',
	'scan'
]);

export const handleSchema = z
	.string()
	.trim()
	.toLowerCase()
	.regex(HANDLE_PATTERN, 'Use 3–30 lowercase letters, numbers, or hyphens.')
	.refine((value) => !RESERVED_HANDLES.has(value), 'That handle is reserved.');
