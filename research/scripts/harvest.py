#!/usr/bin/env python3
"""Harvest Firecrawl tool results from Claude Code session transcripts to disk.

Every Firecrawl MCP call made in this session (by the main agent or any subagent)
is recorded verbatim in the session's JSONL transcript. This script pairs each
call with its result and writes the payload under research/raw/, so scraped
content never has to be retyped.

Usage: python3 harvest.py [--quiet]
"""
import glob
import hashlib
import json
import os
import re
import sys
from urllib.parse import urlparse

TRANSCRIPTS = os.path.expanduser("~/.claude/projects/-home-user-fleet")
RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw")


def slugify(s, n=110):
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")
    return (s[:n] or "index").lower()


def sha(obj, n=10):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:n]


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(data)


def main():
    uses, results = {}, {}
    for fn in glob.glob(os.path.join(TRANSCRIPTS, "**", "*.jsonl"), recursive=True):
        with open(fn, errors="replace") as f:
            for line in f:
                if "firecrawl" not in line and "tool_result" not in line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                msg = d.get("message")
                if not isinstance(msg, dict) or not isinstance(msg.get("content"), list):
                    continue
                for b in msg["content"]:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_use" and "firecrawl" in b.get("name", ""):
                        uses[b["id"]] = (b["name"].split("__")[-1], b.get("input") or {})
                    elif b.get("type") == "tool_result":
                        results[b["tool_use_id"]] = (
                            text_of(b.get("content")),
                            bool(b.get("is_error")),
                            d.get("timestamp", ""),
                        )

    counts = {"pages": 0, "alexandria": 0, "search": 0, "map": 0, "failed": 0}
    reddit_urls, failures = set(), []
    for tid, (name, inp) in uses.items():
        if tid not in results:
            continue
        text, is_err, ts = results[tid]
        # Oversized results are replaced by a pointer to a file on disk; follow it.
        m = re.search(r"saved to:? (/\S+?\.(?:json|txt))", text)
        if m and os.path.exists(m.group(1)):
            text, is_err = open(m.group(1), errors="replace").read(), False
        try:
            payload = json.loads(text)
            if isinstance(payload, list):  # persisted as a list of content blocks
                payload = json.loads(text_of(payload))
        except Exception:
            payload = None
        if is_err or payload is None:
            counts["failed"] += 1
            failures.append(f"{name}\t{inp.get('url') or json.dumps(inp.get('alexandria') or inp.get('query'))}\t{text[:200]!r}")
            continue

        if name == "firecrawl_scrape" and inp.get("url"):
            url = inp["url"]
            u = urlparse(url if "://" in url else "https://" + url)
            slug = slugify(u.path.strip("/") + ("-" + u.query if u.query else ""))
            base = os.path.join(RAW, "pages", u.netloc.lower(), slug)
            md = payload.get("markdown")
            meta = payload.get("metadata") or {}
            if md:
                head = (
                    f"---\nsource_url: {url}\nfinal_url: {meta.get('url', '')}\n"
                    f"title: {json.dumps(meta.get('title', ''))}\nstatus_code: {meta.get('statusCode', '')}\n"
                    f"scraped_at: {ts}\n---\n\n"
                )
                write(base + ".md", head + md + "\n")
                counts["pages"] += 1
            extra = {k: v for k, v in payload.items() if k not in ("markdown", "metadata", "conversationId")}
            if extra:
                write(base + ".json", json.dumps({"source_url": url, "scraped_at": ts, **extra}, indent=1))
                if not md:
                    counts["pages"] += 1
        elif name == "firecrawl_scrape" and inp.get("alexandria"):
            calls = inp["alexandria"]
            calls = calls if isinstance(calls, list) else [calls]
            outs = ((payload.get("data") or {}).get("alexandria")) or []
            for i, call in enumerate(calls):
                out = outs[i] if i < len(outs) else None
                if call.get("provider") == "firecrawl":
                    continue
                d = os.path.join(RAW, "alexandria", call.get("provider", "x"), slugify(call.get("capability", "x")))
                write(
                    os.path.join(d, sha(call) + ".json"),
                    json.dumps({"request": call, "fetched_at": ts, "response": out}, indent=1),
                )
                counts["alexandria"] += 1
                if call.get("capability") == "reddit/post-comments":
                    reddit_urls.add((call.get("options") or {}).get("url", ""))
        elif name == "firecrawl_search":
            write(
                os.path.join(RAW, "search", sha(inp) + ".json"),
                json.dumps({"request": inp, "fetched_at": ts, "response": payload}, indent=1),
            )
            counts["search"] += 1
        elif name == "firecrawl_map":
            host = urlparse(inp["url"] if "://" in inp["url"] else "https://" + inp["url"]).netloc
            write(
                os.path.join(RAW, "map", f"{host}-{sha(inp, 6)}.json"),
                json.dumps({"request": inp, "response": payload}, indent=1),
            )
            counts["map"] += 1

    write(os.path.join(RAW, "reddit_threads_fetched.txt"), "\n".join(sorted(reddit_urls)) + "\n")
    write(os.path.join(RAW, "failed_calls.tsv"), "\n".join(failures) + "\n")
    if "--quiet" not in sys.argv:
        print("harvested:", json.dumps(counts))
        for root, _, files in os.walk(os.path.join(RAW, "pages")):
            if files:
                print(f"  {os.path.relpath(root, RAW)}: {len(files)} file(s)")
        for root, _, files in os.walk(os.path.join(RAW, "alexandria")):
            if files:
                print(f"  {os.path.relpath(root, RAW)}: {len(files)} file(s)")


if __name__ == "__main__":
    main()
