import { z } from 'zod';
import { handleSchema, passwordSchema } from './auth.js';

export const profileSchema = z.object({
	name: z.string('Enter your name.').trim().min(1, 'Enter your name.').max(80)
});

export const gallerySchema = z.object({
	handle: handleSchema,
	galleryPublic: z
		.literal('on')
		.optional()
		.transform((value) => value === 'on'),
	galleryTitle: z.string().trim().max(120).optional(),
	galleryIntro: z.string().trim().max(1000).optional()
});

export const passwordChangeSchema = z
	.object({
		current: z.string('Enter your current password.').min(1, 'Enter your current password.'),
		password: passwordSchema,
		confirm: z.string().optional()
	})
	.refine((data) => data.confirm === undefined || data.confirm === data.password, {
		path: ['confirm'],
		message: 'Passwords do not match.'
	});

export const tokenSchema = z.object({
	name: z.string().trim().max(60).optional()
});

export const deleteAccountSchema = z.object({
	password: z.string('Enter your password to confirm.').min(1, 'Enter your password to confirm.'),
	confirm: z.literal('DELETE', { error: 'Type DELETE to confirm.' })
});
