#!/usr/bin/env python3
"""Download full earnings call transcripts (Motley Fool) and save them as markdown.

Also writes agent-mentions.md: every speaker turn that mentions agents, so the
relevant parts of ~12 long calls can be read in one place.
"""
import os
import re
import sys
import time
import urllib.request

from bs4 import BeautifulSoup
from markdownify import markdownify

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02-earnings-calls")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
BASE = "https://www.fool.com/earnings/call-transcripts/"

CALLS = {
    "MSFT": ("Microsoft", [
        "2026/10/29/microsoft-msft-q1-2027-earnings-call-transcript/",
        "2026/08/07/microsoft-msft-q4-2026-earnings-call-transcript/",
        "2026/04/29/microsoft-msft-q3-2026-earnings-transcript/",
    ]),
    "GOOGL": ("Alphabet", [
        "2026/10/29/alphabet-googl-q3-2026-earnings-call-transcript/",
        "2026/04/29/alphabet-googl-q1-2026-earnings-call-transcript/",
        "2026/02/04/alphabet-googl-q4-2025-earnings-call-transcript/",
    ]),
    "AMZN": ("Amazon", [
        "2026/10/30/amazon-amzn-q3-2026-earnings-call-transcript/",
        "2026/08/07/amazon-amzn-q2-2026-earnings-call-transcript/",
        "2026/04/29/amazon-amzn-q1-2026-earnings-call-transcript/",
    ]),
    "NET": ("Cloudflare", [
        "2026/11/06/cloudflare-net-q3-2026-earnings-call-transcript/",
        "2026/05/07/cloudflare-net-q1-2026-earnings-call-transcript/",
        "2026/02/10/cloudflare-net-q4-2025-earnings-call-transcript/",
    ]),
}
# extra transcript paths can be appended as TICKER=path arguments
for arg in sys.argv[1:]:
    t, p = arg.split("=", 1)
    CALLS[t][1].append(p)

AGENT_RE = re.compile(r"\bagent(s|ic)?\b", re.I)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def main():
    mentions = ["# Agent mentions across earnings calls\n",
                "Every speaker turn containing 'agent', 'agents' or 'agentic', pulled verbatim from the full transcripts in this folder.\n"]
    index = []
    for ticker, (company, paths) in CALLS.items():
        for path in paths:
            url = BASE + path
            html = fetch(url)
            soup = BeautifulSoup(html, "html.parser")
            title = (soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else path)
            body = soup.find(id="article-body-transcript") or soup.find(class_="article-body")
            for junk in body.select(".article-body-promobox, .image-container, script, style, .pitch-container, [class*=ad-]"):
                junk.decompose()
            md = markdownify(str(body), heading_style="ATX")
            md = re.sub(r"\n{3,}", "\n\n", md).strip()
            date = "-".join(path.split("/")[:3])
            m = re.search(r"-(q[1-4])-(\d{4})-", path)
            quarter = f"{m.group(1).upper()}-FY{m.group(2)}" if m else "call"
            fn = os.path.join(OUT, ticker, f"{date}_{ticker}_{quarter}.md")
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            with open(fn, "w") as f:
                f.write(f"---\ncompany: {company}\nticker: {ticker}\nfiscal_quarter: {quarter}\n"
                        f"published: {date}\ntitle: \"{title}\"\nsource_url: {url}\n---\n\n# {title}\n\n{md}\n")
            words = len(md.split())
            turns = [p.strip() for p in md.split("\n\n") if AGENT_RE.search(p)]
            index.append((ticker, quarter, date, words, len(turns), os.path.relpath(fn, OUT), url))
            mentions.append(f"\n## {company} ({ticker}) {quarter}, published {date}\n\nSource: {url}\n")
            mentions.extend(f"\n{t}\n" for t in turns)
            print(f"{ticker} {quarter} {date}: {words} words, {len(turns)} agent turns")
            time.sleep(1)
    with open(os.path.join(OUT, "agent-mentions.md"), "w") as f:
        f.write("\n".join(mentions))
    with open(os.path.join(OUT, "INDEX.md"), "w") as f:
        f.write("# Earnings calls\n\n| Ticker | Fiscal quarter | Published | Words | Turns mentioning agents | File | Source |\n|---|---|---|---|---|---|---|\n")
        for r in index:
            f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | [{r[5]}]({r[5]}) | [link]({r[6]}) |\n")


if __name__ == "__main__":
    main()
