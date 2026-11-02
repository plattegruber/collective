CREATE TABLE `artwork_prototypes` (
	`id` text PRIMARY KEY NOT NULL,
	`artwork_id` text NOT NULL,
	`image_id` text,
	`source` text NOT NULL,
	`release_id` text NOT NULL,
	`embedding` blob NOT NULL,
	`box` text,
	`created_at` integer DEFAULT (unixepoch('subsec') * 1000) NOT NULL,
	FOREIGN KEY (`artwork_id`) REFERENCES `artworks`(`id`) ON UPDATE no action ON DELETE cascade,
	FOREIGN KEY (`image_id`) REFERENCES `artwork_images`(`id`) ON UPDATE no action ON DELETE cascade,
	FOREIGN KEY (`release_id`) REFERENCES `recognizer_releases`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `artwork_prototypes_artwork_idx` ON `artwork_prototypes` (`artwork_id`);--> statement-breakpoint
CREATE INDEX `artwork_prototypes_image_idx` ON `artwork_prototypes` (`image_id`);--> statement-breakpoint
CREATE TABLE `recognizer_releases` (
	`id` text PRIMARY KEY NOT NULL,
	`version` text NOT NULL,
	`detector_key` text NOT NULL,
	`detector_sha256` text NOT NULL,
	`detector_bytes` integer NOT NULL,
	`detector_input_size` integer NOT NULL,
	`embedder_key` text NOT NULL,
	`embedder_sha256` text NOT NULL,
	`embedder_bytes` integer NOT NULL,
	`embedder_input_size` integer NOT NULL,
	`embedder_dim` integer NOT NULL,
	`config` text NOT NULL,
	`metrics` text,
	`notes` text,
	`is_active` integer DEFAULT false NOT NULL,
	`created_at` integer DEFAULT (unixepoch('subsec') * 1000) NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `recognizer_releases_version_unique` ON `recognizer_releases` (`version`);