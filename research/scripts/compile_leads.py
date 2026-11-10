#!/usr/bin/env python3
"""Merge, check and rank lead records from 05-leads/raw/*.jsonl into leads.csv.

Checks each web-sourced lead's evidence URL against the pages and search results
actually fetched in this session (research/raw/), so nothing rests on recall.
"""
import csv
import glob
import html
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEADS = os.path.join(ROOT, "05-leads")
FIELDS = ["priority", "segment", "company", "website", "location", "sector", "size_hint", "signal", "signal_date",
          "named_sponsor", "fit_note", "evidence_quote", "evidence_url", "evidence_checked"]

NOT_A_SYSTEM = re.compile(r"apprentice|training|qualitative|research|survey|course|bootcamp|licen[cs]|\bsub\b|surface pro|\bpcs?\b|adoption|rollout|enablement programme|value realisation", re.I)
LIVE_SYSTEM = re.compile(r"chatbot|assistant|copilot|agent|transcription|summaris|virtual|generative", re.I)
RELIABILITY = re.compile(r"eval|monitor|hallucin|reliab|guardrail|quality|production|maintain|cost", re.I)


def norm(url):
    u = re.sub(r"^https?://(www\.)?", "", (url or "").strip().lower())
    return u.split("#")[0].split("?")[0].rstrip("/")


def seen_urls():
    urls = set()
    for f in glob.glob(os.path.join(ROOT, "raw", "search", "*.json")):
        try:
            data = (json.load(open(f)).get("response") or {}).get("data") or {}
        except Exception:
            continue
        for group in ("web", "news"):
            for hit in data.get(group) or []:
                urls.add(norm(hit.get("url")))
    for f in glob.glob(os.path.join(ROOT, "raw", "pages", "*", "*.md")):
        head = open(f, errors="replace").read(600)
        for m in re.finditer(r"^(?:source_url|final_url): (\S+)", head, re.M):
            urls.add(norm(m.group(1)))
    urls.discard("")
    return urls


def priority(r):
    seg, text = r.get("segment"), f"{r.get('signal', '')} {r.get('evidence_quote', '')}"
    if seg == "public-contract":
        if NOT_A_SYSTEM.search(r.get("signal", "")) or not LIVE_SYSTEM.search(r.get("signal", "")):
            return None
        return "A" if (r.get("value_gbp") or 0) <= 350_000 else "B"
    if seg == "funded-project":
        return "B" if (r.get("value_gbp") or 0) >= 100_000 else "C"
    if seg == "hiring":
        return "A" if RELIABILITY.search(text) else "B"
    if seg == "agency":
        return "A" if re.search(r"\bfor\b.+|client|case study", r.get("signal", ""), re.I) else "B"
    if seg == "launched-ai":
        return "A" if r.get("named_sponsor") else "B"
    return "B"


def main():
    fetched = seen_urls()
    rows, seen, dropped = [], set(), 0
    for f in sorted(glob.glob(os.path.join(LEADS, "raw", "*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                dropped += 1
                continue
            r = {k: (html.unescape(v) if isinstance(v, str) else v) for k, v in r.items()}
            key = (norm(r.get("website")) or re.sub(r"[^a-z0-9]", "", (r.get("company") or "").lower()), r.get("segment"))
            if not r.get("company") or key in seen:
                dropped += 1
                continue
            p = priority(r)
            if p is None:
                dropped += 1
                continue
            seen.add(key)
            r["priority"] = p
            if r.get("segment") in ("public-contract", "funded-project"):
                r["evidence_checked"] = "official record"
            else:
                r["evidence_checked"] = "yes" if norm(r.get("evidence_url")) in fetched else "no"
            rows.append(r)
    order = {"A": 0, "B": 1, "C": 2}
    rows.sort(key=lambda r: (order[r["priority"]], r["segment"], str(r.get("signal_date"))[::-1]))
    with open(os.path.join(LEADS, "leads.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} leads written, {dropped} dropped")
    table = {}
    for r in rows:
        table.setdefault(r["segment"], {}).setdefault(r["priority"], 0)
        table[r["segment"]][r["priority"]] += 1
    for seg, t in sorted(table.items()):
        unchecked = sum(1 for r in rows if r["segment"] == seg and r["evidence_checked"] == "no")
        print(f"  {seg:16} A={t.get('A', 0):3} B={t.get('B', 0):3} C={t.get('C', 0):3}  evidence not matched: {unchecked}")


if __name__ == "__main__":
    main()
