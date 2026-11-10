#!/usr/bin/env python3
"""Add the two calls Motley Fool's listing lacked, then rebuild INDEX.md and agent-mentions.md.

  GOOGL Q2 2026: Alphabet's official transcript PDF (text extracted with pdftotext)
  NET   Q2 2026: Yahoo Finance / Quartr transcript, as returned in a Firecrawl search result
Usage: add_extra_earnings.py <googl_q2.txt> <firecrawl_search_result.json>
"""
import json
import os
import re
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02-earnings-calls")
AGENT_RE = re.compile(r"\bagent(s|ic)?\b", re.I)


def save(ticker, company, quarter, date, title, url, body):
    fn = os.path.join(OUT, ticker, f"{date}_{ticker}_{quarter}.md")
    with open(fn, "w") as f:
        f.write(f"---\ncompany: {company}\nticker: {ticker}\nfiscal_quarter: {quarter}\npublished: {date}\n"
                f"title: \"{title}\"\nsource_url: {url}\n---\n\n# {title}\n\n{body.strip()}\n")


if len(sys.argv) == 3:
    txt = open(sys.argv[1]).read().replace("\f", "\n")
    # pdftotext wraps lines; re-join wrapped lines into paragraphs
    txt = re.sub(r"(?<![\n.:?!])\n(?!\n)", " ", txt)
    save("GOOGL", "Alphabet", "Q2-FY2026", "2026-07-22", "Alphabet (GOOGL) Q2 2026 Earnings Call Transcript",
         "https://s206.q4cdn.com/479360582/files/doc_events/2026/Jul/22/2026_Q2_Earnings_Transcript.pdf", txt)
    web = json.load(open(sys.argv[2]))["data"]["web"]
    hit = next(w for w in web if "NET-Q2-2026-earnings_call" in w["url"])
    body = hit["description"].split("\n", 1)[1].replace("Yahoo Finance Pushly", "")
    # put "Speaker / role / timestamp" headers on one bold line
    body = re.sub(r"\n\n([^\n]{2,60})\n\n(?:([^\n]{2,60})\n\n)?(\d+:\d\d:\d\d)\n\n", lambda m: f"\n\n**{m.group(1)}{' (' + m.group(2) + ')' if m.group(2) else ''}, {m.group(3)}:** ", body)
    save("NET", "Cloudflare", "Q2-FY2026", "2026-08-06", "Cloudflare (NET) Q2 2026 Earnings Call Transcript", hit["url"], body)

rows, mentions = [], ["# Agent mentions across earnings calls\n",
                      "Every paragraph containing 'agent', 'agents' or 'agentic', pulled verbatim from the full transcripts in this folder.\n"]
for ticker in sorted(os.listdir(OUT)):
    d = os.path.join(OUT, ticker)
    if not os.path.isdir(d):
        continue
    for fn in sorted(os.listdir(d), reverse=True):
        text = open(os.path.join(d, fn)).read()
        meta = dict(l.split(": ", 1) for l in text.split("---")[1].strip().splitlines())
        body = text.split("---", 2)[2]
        turns = [p.strip() for p in body.split("\n\n") if AGENT_RE.search(p)]
        rows.append((ticker, meta["fiscal_quarter"], meta["published"], len(body.split()), len(turns), f"{ticker}/{fn}", meta["source_url"]))
        mentions.append(f"\n## {meta['company']} ({ticker}) {meta['fiscal_quarter']}, published {meta['published']}\n\nSource: {meta['source_url']}\n")
        mentions.extend(f"\n{t}\n" for t in turns)
open(os.path.join(OUT, "agent-mentions.md"), "w").write("\n".join(mentions))
with open(os.path.join(OUT, "INDEX.md"), "w") as f:
    f.write("# Earnings calls\n\n| Ticker | Fiscal quarter | Published | Words | Paragraphs mentioning agents | File | Source |\n|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | [{r[5]}]({r[5]}) | [link]({r[6]}) |\n")
        print(*r[:5])
