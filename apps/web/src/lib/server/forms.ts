import { fail } from '@sveltejs/kit';
import { z } from 'zod';

export type FieldErrors = Record<string, string>;

export type FormFailure = {
	errors: FieldErrors;
	formError?: string;
	values: Record<string, string>;
};

/** Plain string values from a FormData, suitable for echoing back into inputs. */
export function formValues(
	formData: FormData,
	omit: string[] = ['password', 'confirm']
): Record<string, string> {
	const values: Record<string, string> = {};
	for (const [key, value] of formData.entries()) {
		if (typeof value === 'string' && !omit.includes(key)) values[key] = value;
	}
	return values;
}

/** Empty strings become undefined so optional fields validate naturally. */
export function formObject(formData: FormData): Record<string, string | undefined> {
	const out: Record<string, string | undefined> = {};
	for (const [key, value] of formData.entries()) {
		if (typeof value !== 'string') continue;
		const trimmed = value.trim();
		out[key] = trimmed === '' ? undefined : trimmed;
	}
	return out;
}

export function firstIssuePerField(error: z.ZodError): FieldErrors {
	const errors: FieldErrors = {};
	for (const issue of error.issues) {
		const key = issue.path.map(String).join('.') || '_form';
		errors[key] ??= issue.message;
	}
	return errors;
}

/**
 * Validate a FormData against a schema. On failure returns a SvelteKit `fail(400, …)`
 * with per-field messages and the submitted values so the page can re-render the form.
 */
export function parseForm<S extends z.ZodType>(
	formData: FormData,
	schema: S
): { ok: true; data: z.output<S> } | { ok: false; failure: ReturnType<typeof fail<FormFailure>> } {
	const result = schema.safeParse(formObject(formData));
	if (result.success) return { ok: true, data: result.data };
	return { ok: false, failure: fieldFailure(firstIssuePerField(result.error), formData) };
}

/** A 400 with per-field messages; use when validation needs context the schema lacks. */
export function fieldFailure(errors: FieldErrors, formData: FormData) {
	return fail<FormFailure>(400, { errors, values: formValues(formData) });
}

export function formFailure(
	status: number,
	formError: string,
	formData: FormData,
	errors: FieldErrors = {}
) {
	return fail<FormFailure>(status, { errors, formError, values: formValues(formData) });
}
