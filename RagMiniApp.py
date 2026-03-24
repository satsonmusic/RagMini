import argparse
import hashlib
import json
import random
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ----------------------------
# Config & Paths
# ----------------------------

def _lab_root() -> Path:
	"""Hardcoded to your WebSum folder so the database always saves here."""
	return Path(r"C:\\Users\\scott\\WebSum")


DB_FILE = str(_lab_root() / "pipeline.sqlite")
DEFAULT_TIMEOUT_S = 20


def utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def log(event: str, **fields: Any) -> None:
	rec = {"ts": utc_now_iso(), "event": event, **fields}
	print(json.dumps(rec, ensure_ascii=False))


# ----------------------------
# Storage (SQLite)
# ----------------------------

def connect(db_path: Path) -> sqlite3.Connection:
	db_path.parent.mkdir(parents=True, exist_ok=True)
	con = sqlite3.connect(str(db_path))  # single-machine
	con.row_factory = sqlite3.Row
	con.execute("PRAGMA journal_mode=WAL")
	con.execute("PRAGMA foreign_keys=ON")
	return con


def init_db(con: sqlite3.Connection) -> None:
	con.executescript(
		"""
		CREATE TABLE IF NOT EXISTS jobs (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			url TEXT NOT NULL,
			status TEXT NOT NULL,              -- queued|running|done|failed
			priority INTEGER NOT NULL DEFAULT 0,
			attempts INTEGER NOT NULL DEFAULT 0,
			max_attempts INTEGER NOT NULL DEFAULT 5,
			next_run_at TEXT NOT NULL,
			locked_at TEXT,
			locked_by TEXT,
			last_error TEXT,
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL
		);

		CREATE INDEX IF NOT EXISTS idx_jobs_pick
			ON jobs(status, next_run_at, priority, id);

		CREATE TABLE IF NOT EXISTS cache (
			cache_key TEXT PRIMARY KEY,
			url TEXT NOT NULL,
			content_hash TEXT NOT NULL,
			fetched_at TEXT NOT NULL,
			extractor_version TEXT NOT NULL,
			model_version TEXT NOT NULL,
			summary TEXT NOT NULL
		);
		"""
	)
	con.commit()


# ----------------------------
# Fetch + extract
# ----------------------------

def fetch_url(url: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> Tuple[str, str]:
	"""Return (final_url, html_text)."""
	req = urllib.request.Request(
		url,
		headers={
			"User-Agent": "SummarizationPipeline/1.0 (+https://example.local)",
			"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
		},
	)
	with urllib.request.urlopen(req, timeout=timeout_s) as resp:
		final_url = resp.geturl()
		charset = "utf-8"
		ct = resp.headers.get("Content-Type", "")
		# crude charset detection
		if "charset=" in ct:
			charset = ct.split("charset=")[-1].split(";")[0].strip() or charset
		data = resp.read()
		return final_url, data.decode(charset, errors="replace")


def extract_title(html: str) -> str:
	"""Extract <title>...</title> if present."""
	low = html.lower()
	s = low.find("<title")
	if s == -1:
		return ""
	s = low.find(">", s)
	if s == -1:
		return ""
	e = low.find("</title>", s)
	if e == -1:
		return ""
	t = html[s + 1 : e]
	return " ".join(t.replace("\n", " ").split()).strip()


def extract_text(html: str) -> str:
	"""HTML->text extractor (stdlib-only).

	This is still intentionally lightweight, but it does two important things:
	- Removes obvious boilerplate tags (script/style/noscript)
	- Inserts newlines around block-ish tags so we can later prefer paragraph-like lines
	"""
	lower = html.lower()
	for tag in ("script", "style", "noscript"):
		while True:
			start = lower.find(f"<{tag}")
			if start == -1:
				break
			end = lower.find(f"</{tag}>", start)
			if end == -1:
				break
			end = end + len(f"</{tag}>")
			html = html[:start] + " " + html[end:]
			lower = html.lower()

	# Add rough newlines around common block tags
	for t in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"):
		html = html.replace(f"</{t}>", f"</{t}>\n")

	# Strip tags (naive)
	out = []
	in_tag = False
	for ch in html:
		if ch == "<":
			in_tag = True
			continue
		if ch == ">":
			in_tag = False
			out.append(" ")
			continue
		if in_tag:
			continue
		out.append(ch)

	text = "".join(out)
	# Normalize whitespace but preserve newlines
	lines = [" ".join(ln.split()).strip() for ln in text.splitlines()]
	lines = [ln for ln in lines if ln]
	return "\n".join(lines)


# ----------------------------
# Summarization (heuristic, stdlib-only)
# ----------------------------

def _tokenize(s: str) -> list:
	word = []
	out = []
	for ch in s.lower():
		if ch.isalnum() or ch in ("_", "-"):
			word.append(ch)
		else:
			if len(word) >= 3:
				out.append("".join(word))
			word = []
	if len(word) >= 3:
		out.append("".join(word))
	return out


_STOP = {
	"the",
	"and",
	"for",
	"with",
	"that",
	"this",
	"from",
	"are",
	"was",
	"were",
	"has",
	"have",
	"had",
	"not",
	"but",
	"you",
	"your",
	"they",
	"their",
	"into",
	"about",
	"also",
	"can",
	"may",
	"will",
	"one",
	"two",
	"new",
	"use",
	"used",
	"using",
	"more",
	"most",
	"some",
	"such",
	"other",
	"than",
	"these",
	"those",
	"which",
	"what",
	"when",
	"where",
	"who",
	"how",
	"why",
	"wikipedia",
	"jump",
	"content",
	"menu",
	"navigation",
	"search",
	"donate",
	"create",
	"account",
	"log",
	"main",
	"page",
	"contents",
	"edit",
	"community",
	"portal",
	"recent",
	"changes",
	"upload",
	"file",
	"special",
	"pages",
}


def _is_boilerplate_line(line: str) -> bool:
	low = line.lower()
	bad_phrases = (
		"jump to content",
		"main menu",
		"navigation",
		"create account",
		"log in",
		"donate",
		"recent changes",
		"community portal",
		"special pages",
		"contact us",
		"this material may not be published",
		"may not be published, broadcast",
		"rewritten, or redistributed",
		"all rights reserved",
		"terms of use",
		"privacy policy",
	)
	if any(p in low for p in bad_phrases):
		return True
	# short UI-ish fragments
	if len(low) < 25 and ("menu" in low or "navigation" in low):
		return True
	return False


def _looks_like_site_heading(sentence: str, page_title: str) -> bool:
	"""Filter out title/branding-y lines like:
	"Fox News - Breaking News | Latest Headlines ..."
	"CNN: Breaking News, Latest News..."
	"Home | SiteName"
	"""
	s = (sentence or "").strip()
	if not s:
		return True
	low = s.lower()

	# Lots of separators is usually nav/title spam
	if low.count("|") >= 1:
		return True
	if low.count(" - ") >= 2:
		return True
	if low.count(":") >= 2:
		return True

	# If it starts with the page title (or vice versa), likely branding
	t = (page_title or "").strip()
	if t:
		t_low = t.lower()
		if low.startswith(t_low) or t_low.startswith(low[: min(len(low), 40)]):
			return True
		# If most tokens are shared with title, also treat as heading
		s_tokens = set(_tokenize(low))
		t_tokens = set(_tokenize(t_low))
		if s_tokens and t_tokens:
			shared = len(s_tokens & t_tokens)
			if shared / max(1, len(s_tokens)) >= 0.7:
				return True

	return False


def summarize(text: str, title: str = "", max_input_chars: int = 12000, max_items: int = 7) -> str:
	"""Return numbered key-insight items (5-10), 1-2 sentences each.

	Important behaviors:
	- Output is plain text (not JSON).
	- Does NOT prefix output with the page title.
	- Filters out branding / site-heading style sentences.
	"""
	text = (text or "").strip()
	if not text:
		return "(empty)"

	# Keep more input for better scoring
	text = text[:max_input_chars]

	lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
	lines = [ln for ln in lines if not _is_boilerplate_line(ln)]

	# Build a working body by selecting medium-length lines (skip nav-y one-liners)
	body_lines = []
	for ln in lines:
		if 60 <= len(ln) <= 400:
			body_lines.append(ln)
		if len(body_lines) >= 60:
			break

	body = " ".join(body_lines) if body_lines else " ".join(lines[:30])
	body = " ".join(body.split())

	# Split into sentences (naive but ok)
	sents = []
	buf = []
	for ch in body:
		buf.append(ch)
		if ch in ".!?":
			s = "".join(buf).strip()
			buf = []
			if 40 <= len(s) <= 280:
				sents.append(s)
	if buf:
		s = "".join(buf).strip()
		if 40 <= len(s) <= 280:
			sents.append(s)

	if not sents:
		# fallback: first 1-2 reasonable lines (avoid title-like junk)
		fallback_lines = [ln for ln in lines if not _looks_like_site_heading(ln, title)]
		fallback = " ".join(fallback_lines[:3] if fallback_lines else lines[:3])
		return fallback[:800]

	freq: Dict[str, int] = {}
	for s in sents:
		for tok in _tokenize(s):
			if tok in _STOP:
				continue
			freq[tok] = freq.get(tok, 0) + 1

	def score_sentence(s: str) -> float:
		toks = [t for t in _tokenize(s) if t not in _STOP]
		if not toks:
			return 0.0
		# downweight very common words across the page
		score = 0.0
		for t in toks:
			score += 1.0 / (1.0 + (freq.get(t, 0) - 1))
		# prefer earlier sentences slightly
		return score

	scored = [(score_sentence(s), i, s) for i, s in enumerate(sents)]
	scored.sort(reverse=True)
	picked = []
	seen = set()
	for _sc, _i, s in scored:
		if _looks_like_site_heading(s, title):
			continue
		key = s.lower()[:80]
		if key in seen:
			continue
		seen.add(key)
		picked.append(s)
		if len(picked) >= max_items:
			break

	picked.sort(key=lambda s: body.find(s))

	# Preserve original reading order
	picked.sort(key=lambda s: body.find(s))

	# Ensure each item ends with punctuation and format as numbered list
	items = []
	for s in picked:
		s2 = s.strip()
		if s2 and s2[-1] not in ".!?":
			s2 += "."
		items.append(s2)

	# Clamp to 5-10 items
	items = items[: max(5, min(10, max_items))]
	return "\n".join([f"{i+1}. {it}" for i, it in enumerate(items)]).strip()


# ----------------------------
# Caching
# ----------------------------

def sha256_hex(s: str) -> str:
	return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def cache_key_for(url: str) -> str:
	return sha256_hex(url.strip())


# ----------------------------
# Queue operations
# ----------------------------

def enqueue(con: sqlite3.Connection, url: str, priority: int = 0, max_attempts: int = 5) -> int:
	now = utc_now_iso()
	con.execute(
		"""
		INSERT INTO jobs(url, status, priority, attempts, max_attempts, next_run_at, created_at, updated_at)
		VALUES(?, 'queued', ?, 0, ?, ?, ?, ?)
		""",
		(url, priority, max_attempts, now, now, now),
	)
	con.commit()
	# With row_factory=sqlite3.Row, fetchone() returns a Row (not a tuple)
	row = con.execute("SELECT last_insert_rowid() AS id").fetchone()
	job_id = int(row["id"]) if row is not None else -1
	log("job_enqueued", job_id=job_id, url=url, priority=priority)
	return job_id


@dataclass
class Job:
	id: int
	url: str
	attempts: int
	max_attempts: int


def claim_next_job(con: sqlite3.Connection, worker_id: str) -> Optional[Job]:
	"""Atomically claim the next available job."""
	now = utc_now_iso()
	cur = con.cursor()
	cur.execute("BEGIN IMMEDIATE")
	row = cur.execute(
		"""
		SELECT id, url, attempts, max_attempts
		FROM jobs
		WHERE status = 'queued'
			AND next_run_at <= ?
		ORDER BY priority DESC, id ASC
		LIMIT 1
		""",
		(now,),
	).fetchone()
	if row is None:
		cur.execute("COMMIT")
		return None

	cur.execute(
		"""
		UPDATE jobs
		SET status='running', locked_at=?, locked_by=?, updated_at=?
		WHERE id=?
		""",
		(now, worker_id, now, row["id"]),
	)
	cur.execute("COMMIT")
	return Job(
		id=int(row["id"]),
		url=str(row["url"]),
		attempts=int(row["attempts"]),
		max_attempts=int(row["max_attempts"]),
	)


def backoff_seconds(attempts: int) -> int:
	base = min(60 * 60, int(2 ** min(attempts, 10)))
	return int(base * (0.5 + random.random()))


def mark_done(con: sqlite3.Connection, job_id: int) -> None:
	now = utc_now_iso()
	con.execute(
		"UPDATE jobs SET status='done', updated_at=?, last_error=NULL WHERE id=?",
		(now, job_id),
	)
	con.commit()
	log("job_done", job_id=job_id)


def mark_failed(con: sqlite3.Connection, job_id: int, err: str, attempts: int, max_attempts: int) -> None:
	now = utc_now_iso()
	attempts2 = attempts + 1
	if attempts2 >= max_attempts:
		con.execute(
			"""
			UPDATE jobs
			SET status='failed', attempts=?, updated_at=?, last_error=?
			WHERE id=?
			""",
			(attempts2, now, err, job_id),
		)
		con.commit()
		log("job_failed", job_id=job_id, attempts=attempts2, max_attempts=max_attempts, error=err)
		return

	delay = backoff_seconds(attempts2)
	next_run = datetime.now(timezone.utc).timestamp() + delay
	next_run_iso = datetime.fromtimestamp(next_run, tz=timezone.utc).isoformat()
	con.execute(
		"""
		UPDATE jobs
		SET status='queued', attempts=?, next_run_at=?, updated_at=?, last_error=?, locked_at=NULL, locked_by=NULL
		WHERE id=?
		""",
		(attempts2, next_run_iso, now, err, job_id),
	)
	con.commit()
	log("job_retry_scheduled", job_id=job_id, attempts=attempts2, next_run_at=next_run_iso, error=err)


def store_cache(
	con: sqlite3.Connection,
	url: str,
	html: str,
	summary_text: str,
	extractor_version: str = "extract_v1",
	model_version: str = "stub_v1",
) -> None:
	ck = cache_key_for(url)
	h = sha256_hex(html)
	con.execute(
		"""
		INSERT INTO cache(cache_key, url, content_hash, fetched_at, extractor_version, model_version, summary)
		VALUES(?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(cache_key) DO UPDATE SET
			content_hash=excluded.content_hash,
			fetched_at=excluded.fetched_at,
			extractor_version=excluded.extractor_version,
			model_version=excluded.model_version,
			summary=excluded.summary
		""",
		(ck, url, h, utc_now_iso(), extractor_version, model_version, summary_text),
	)
	con.commit()


def get_cached_summary(con: sqlite3.Connection, url: str) -> Optional[Dict[str, str]]:
	ck = cache_key_for(url)
	row = con.execute(
		"SELECT url, content_hash, fetched_at, extractor_version, model_version, summary FROM cache WHERE cache_key=?",
		(ck,),
	).fetchone()
	if row is None:
		return None
	return {k: str(row[k]) for k in row.keys()}


# ----------------------------
# Worker loop
# ----------------------------

def process_one(con: sqlite3.Connection, job: Job, timeout_s: int) -> None:
	start_t = time.time()
	cached = get_cached_summary(con, job.url)
	if cached is not None:
		log("cache_hit", job_id=job.id, url=job.url, fetched_at=cached["fetched_at"])
		mark_done(con, job.id)
		return

	try:
		final_url, html = fetch_url(job.url, timeout_s=timeout_s)
		text = extract_text(html)
		sum_text = summarize(text)
		store_cache(con, final_url, html, sum_text)
		mark_done(con, job.id)
		log(
			"job_processed",
			job_id=job.id,
			url=job.url,
			final_url=final_url,
			ms=int((time.time() - start_t) * 1000),
		)
	except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
		mark_failed(con, job.id, f"network:{type(e).__name__}:{e}", job.attempts, job.max_attempts)
	except Exception as e:
		mark_failed(con, job.id, f"error:{type(e).__name__}:{e}", job.attempts, job.max_attempts)


def worker(db_path: Path, worker_id: str, poll_s: float, timeout_s: int) -> None:
	con = connect(db_path)
	init_db(con)
	log("worker_start", worker_id=worker_id, db=str(db_path), poll_s=poll_s, timeout_s=timeout_s)
	while True:
		job = claim_next_job(con, worker_id)
		if job is None:
			time.sleep(poll_s)
			continue
		log("job_claimed", worker_id=worker_id, job_id=job.id, url=job.url, attempts=job.attempts)
		process_one(con, job, timeout_s=timeout_s)


# ----------------------------
# CLI
# ----------------------------

def cmd_enqueue(args: argparse.Namespace) -> int:
	con = connect(Path(args.db))
	init_db(con)
	enqueue(con, args.url, priority=args.priority, max_attempts=args.max_attempts)
	return 0


def cmd_worker(args: argparse.Namespace) -> int:
	worker(Path(args.db), args.worker_id, poll_s=args.poll_s, timeout_s=args.timeout_s)
	return 0


def cmd_get(args: argparse.Namespace) -> int:
	con = connect(Path(args.db))
	init_db(con)
	cached = get_cached_summary(con, args.url)
	if cached is None:
		print("(no cache)")
		return 1
	print(json.dumps(cached, indent=2, ensure_ascii=False))
	return 0


def cmd_summarize(args: argparse.Namespace) -> int:
	"""One-shot command: fetch -> extract -> insights -> print.

	Prints ONLY the numbered insight list (no metadata JSON).
	"""
	con = connect(Path(args.db))
	init_db(con)

	# Fast path: return cached summary if present (print summary only)
	cached = get_cached_summary(con, args.url)
	if cached is not None and not args.no_cache:
		print(cached.get("summary", ""))
		return 0

	final_url, html = fetch_url(args.url, timeout_s=args.timeout_s)
	title = extract_title(html)
	text = extract_text(html)
	sum_text = summarize(text, title=title, max_items=args.items)
	store_cache(con, final_url, html, sum_text)

	print(sum_text)
	return 0


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Production-ish web summarization pipeline (SQLite queue + cache)")
	p.add_argument("--db", default=DB_FILE, help="SQLite db file")
	sp = p.add_subparsers(dest="cmd", required=True)

	p_sum = sp.add_parser("summarize", help="One command: fetch + print numbered key insights")
	p_sum.add_argument("url")
	p_sum.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
	p_sum.add_argument("--no-cache", action="store_true", help="Ignore cache and re-fetch")
	p_sum.add_argument("--items", type=int, default=7, help="Number of insight items (5-10)")
	p_sum.set_defaults(func=cmd_summarize)

	p_enq = sp.add_parser("enqueue", help="Add a URL as a job")
	p_enq.add_argument("url")
	p_enq.add_argument("--priority", type=int, default=0)
	p_enq.add_argument("--max-attempts", type=int, default=5)
	p_enq.set_defaults(func=cmd_enqueue)

	p_w = sp.add_parser("worker", help="Run a worker loop")
	p_w.add_argument("--worker-id", default=f"worker-{int(time.time())}")
	p_w.add_argument("--poll-s", type=float, default=1.0)
	p_w.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
	p_w.set_defaults(func=cmd_worker)

	p_g = sp.add_parser("get", help="Print cached summary for a URL")
	p_g.add_argument("url")
	p_g.set_defaults(func=cmd_get)

	return p


def main(argv: Optional[list] = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	return int(args.func(args))


if __name__ == "__main__":
	raise SystemExit(main())
