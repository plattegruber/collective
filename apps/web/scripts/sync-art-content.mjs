#!/usr/bin/env node
/**
 * Fetch artwork metadata from the canonical Google Sheet and emit the JSON map
 * consumed by the web client + ML tooling.
 *
 * Usage:
 *   node scripts/sync-art-content.mjs \
 *     [--sheet <sheetId>] \
 *     [--gid <tabGid>] \
 *     [--out <outputPath>]
 *
 * Environment overrides:
 *   SHEET_ID, SHEET_GID, ART_OUT_PATH
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import https from 'node:https';

const DEFAULT_SHEET_ID = '1iO9Ci_UNETwIoLlR-wYpZ-fJy4PmrcF8WvgSnv6zK6I';
const DEFAULT_GID = '0';
const __dirname = dirname(fileURLToPath(import.meta.url));
const DEFAULT_OUT_PATH = join(__dirname, '..', 'public', 'data', 'art-content.v2.json');

function parseArgs(argv) {
  const config = {};
  argv.forEach((arg) => {
    if (!arg.startsWith('--')) return;
    const [rawKey, rawValue] = arg.slice(2).split('=');
    const key = rawKey.trim();
    if (!key) return;
    const value = rawValue !== undefined ? rawValue : null;
    config[key] = value;
  });
  return config;
}

function fetchText(url, redirectCount = 0) {
  return new Promise((resolve, reject) => {
    https
      .get(url, (res) => {
        const isRedirect = res.statusCode && [301, 302, 303, 307, 308].includes(res.statusCode);
        if (isRedirect && res.headers.location) {
          if (redirectCount > 5) {
            reject(new Error('Too many redirects while fetching sheet.'));
            res.resume();
            return;
          }
          const nextUrl = res.headers.location.startsWith('http')
            ? res.headers.location
            : new URL(res.headers.location, url).toString();
          res.resume();
          resolve(fetchText(nextUrl, redirectCount + 1));
          return;
        }
        if (res.statusCode && res.statusCode >= 400) {
          reject(new Error(`Request failed with status ${res.statusCode}`));
          res.resume();
          return;
        }
        let data = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => {
          data += chunk;
        });
        res.on('end', () => resolve(data));
      })
      .on('error', reject);
  });
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        current += '"';
        i += 1;
        continue;
      }
      inQuotes = !inQuotes;
      continue;
    }

    if (char === ',' && !inQuotes) {
      row.push(current);
      current = '';
      continue;
    }

    if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && next === '\n') {
        i += 1;
      }
      row.push(current);
      rows.push(row);
      row = [];
      current = '';
      continue;
    }

    current += char;
  }

  if (current.length > 0 || row.length > 0) {
    row.push(current);
    rows.push(row);
  }

  return rows.map((cells) => cells.map((cell) => cell.trim()));
}

function normalizeHeader(value) {
  return value.replace(/\ufeff/g, '').toLowerCase().replace(/\s+/g, ' ').trim();
}

function sanitizePieceId(value) {
  if (!value) return '';
  return value.replace(/\*/g, '').replace(/\\/g, '').trim();
}

function parseYear(value) {
  if (!value) return undefined;
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  const numeric = Number(trimmed);
  if (Number.isFinite(numeric)) {
    return numeric;
  }
  return trimmed;
}

async function main() {
  const rawArgs = parseArgs(process.argv.slice(2));
  const sheetId = process.env.SHEET_ID ?? rawArgs.sheet ?? DEFAULT_SHEET_ID;
  const gid = process.env.SHEET_GID ?? rawArgs.gid ?? DEFAULT_GID;
  const outPath = process.env.ART_OUT_PATH ?? rawArgs.out ?? DEFAULT_OUT_PATH;

  if (!sheetId) {
    console.error('[sync-art-content] Missing sheet id (--sheet or SHEET_ID).');
    process.exit(1);
  }

  const url = `https://docs.google.com/spreadsheets/d/${sheetId}/export?format=csv&gid=${gid}`;
  console.log(`[sync-art-content] Fetching CSV from ${url}`);

  let csvRaw;
  try {
    csvRaw = await fetchText(url);
  } catch (error) {
    console.error('[sync-art-content] Failed to download sheet:', error.message);
    process.exit(1);
  }

  const rows = parseCsv(csvRaw).filter((cells) => cells.some((cell) => cell !== ''));
  if (rows.length < 2) {
    console.error('[sync-art-content] Sheet appears empty.');
    process.exit(1);
  }

  const headers = rows[0].map(normalizeHeader);
  const indexOf = (label) => {
    const idx = headers.findIndex((header) => header === label);
    if (idx === -1) {
      console.error(`[sync-art-content] Missing required column "${label}".`);
      process.exit(1);
    }
    return idx;
  };

  const columnIndex = {
    pieceId: indexOf('piece id'),
    title: indexOf('piece title'),
    artist: indexOf('artist name'),
    materials: indexOf('medium'),
    year: indexOf('date created'),
    description: indexOf('description'),
  };

  const output = {};
  const warnings = [];

  rows.slice(1).forEach((cells, rowIndex) => {
    const pieceId = sanitizePieceId(cells[columnIndex.pieceId]);
    if (!pieceId) {
      warnings.push(`Row ${rowIndex + 2}: missing Piece ID; skipped.`);
      return;
    }
    const entry = {
      title: cells[columnIndex.title]?.trim() ?? '',
      artist: cells[columnIndex.artist]?.trim() ?? '',
      materials: cells[columnIndex.materials]?.trim() ?? '',
      description: cells[columnIndex.description]?.trim() ?? '',
    };
    const yearValue = parseYear(cells[columnIndex.year]);
    if (yearValue !== undefined && yearValue !== '') {
      entry.year = yearValue;
    }
    output[pieceId] = entry;
  });

  if (Object.keys(output).length === 0) {
    console.error('[sync-art-content] No valid rows found in sheet.');
    process.exit(1);
  }

  warnings.forEach((message) => console.warn('[sync-art-content]', message));
  const dir = dirname(outPath);
  mkdirSync(dir, { recursive: true });
  writeFileSync(outPath, JSON.stringify(output, null, 2));
  console.log(`[sync-art-content] Wrote ${Object.keys(output).length} entries to ${outPath}`);
}

main().catch((error) => {
  console.error('[sync-art-content] Unexpected failure:', error);
  process.exit(1);
});
