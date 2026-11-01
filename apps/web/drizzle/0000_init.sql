CREATE TABLE `api_tokens` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`name` text NOT NULL,
	`prefix` text NOT NULL,
	`last_used_at` integer,
	`created_at` integer DEFAULT (unixepoch('subsec') * 1000) NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `api_tokens_user_idx` ON `api_tokens` (`user_id`);--> statement-breakpoint
CREATE TABLE `artwork_images` (
	`id` text PRIMARY KEY NOT NULL,
	`artwork_id` text NOT NULL,
	`key` text NOT NULL,
	`thumb_key` text NOT NULL,
	`content_type` text NOT NULL,
	`width` integer NOT NULL,
	`height` integer NOT NULL,
	`bytes` integer NOT NULL,
	`position` integer DEFAULT 0 NOT NULL,
	`created_at` integer DEFAULT (unixepoch('subsec') * 1000) NOT NULL,
	FOREIGN KEY (`artwork_id`) REFERENCES `artworks`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `artwork_images_artwork_idx` ON `artwork_images` (`artwork_id`);--> statement-breakpoint
CREATE UNIQUE INDEX `artwork_images_key_idx` ON `artwork_images` (`key`);--> statement-breakpoint
CREATE TABLE `artwork_tags` (
	`artwork_id` text NOT NULL,
	`tag_id` text NOT NULL,
	PRIMARY KEY(`artwork_id`, `tag_id`),
	FOREIGN KEY (`artwork_id`) REFERENCES `artworks`(`id`) ON UPDATE no action ON DELETE cascade,
	FOREIGN KEY (`tag_id`) REFERENCES `tags`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE TABLE `artworks` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`title` text NOT NULL,
	`artist` text,
	`year` integer,
	`date_text` text,
	`medium` text,
	`height_cm` real,
	`width_cm` real,
	`depth_cm` real,
	`dimension_unit` text DEFAULT 'cm' NOT NULL,
	`description` text,
	`status` text DEFAULT 'owned' NOT NULL,
	`location` text,
	`acquired_on` text,
	`acquired_from` text,
	`purchase_price` integer,
	`estimated_value` integer,
	`currency` text DEFAULT 'USD' NOT NULL,
	`provenance` text,
	`notes` text,
	`is_public` integer DEFAULT false NOT NULL,
	`primary_image_id` text,
	`created_at` integer DEFAULT (unixepoch('subsec') * 1000) NOT NULL,
	`updated_at` integer NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `artworks_user_idx` ON `artworks` (`user_id`);--> statement-breakpoint
CREATE INDEX `artworks_user_status_idx` ON `artworks` (`user_id`,`status`);--> statement-breakpoint
CREATE TABLE `detector_models` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`model_key` text NOT NULL,
	`labels` text NOT NULL,
	`sha256` text NOT NULL,
	`bytes` integer NOT NULL,
	`input_size` integer DEFAULT 320 NOT NULL,
	`metrics` text,
	`is_active` integer DEFAULT true NOT NULL,
	`created_at` integer DEFAULT (unixepoch('subsec') * 1000) NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `detector_models_user_idx` ON `detector_models` (`user_id`);--> statement-breakpoint
CREATE TABLE `sessions` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`expires_at` integer NOT NULL,
	`created_at` integer DEFAULT (unixepoch('subsec') * 1000) NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `sessions_user_idx` ON `sessions` (`user_id`);--> statement-breakpoint
CREATE TABLE `tags` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`name` text NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `tags_user_name_idx` ON `tags` (`user_id`,`name`);--> statement-breakpoint
CREATE TABLE `users` (
	`id` text PRIMARY KEY NOT NULL,
	`email` text NOT NULL,
	`password_hash` text NOT NULL,
	`name` text NOT NULL,
	`handle` text,
	`gallery_public` integer DEFAULT false NOT NULL,
	`gallery_title` text,
	`gallery_intro` text,
	`created_at` integer DEFAULT (unixepoch('subsec') * 1000) NOT NULL,
	`updated_at` integer NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `users_email_idx` ON `users` (`email`);--> statement-breakpoint
CREATE UNIQUE INDEX `users_handle_idx` ON `users` (`handle`);