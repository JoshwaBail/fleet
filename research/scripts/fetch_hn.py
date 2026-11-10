#!/usr/bin/env python3
"""Fetch Hacker News threads about agent frameworks/SDKs via the public Algolia API.

Finds stories whose TITLE matches a category keyword and that drew real
discussion, then saves each full comment tree as markdown.
"""
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "04-forums", "hackernews")
API = "https://hn.algolia.com/api/v1/"
MIN_COMMENTS = 25
PER_QUERY = 12
MAX_THREADS = 90

QUERIES = [
    "langchain", "langgraph", "langsmith", "llamaindex", "crewai", "autogen", "semantic kernel",
    "agent framework", "agent frameworks", "agents sdk", "agent sdk", "agent development kit",
    "agentkit", "agentcore", "strands agents", "vercel ai sdk", "pydantic ai", "mastra", "dspy",
    "smolagents", "letta", "memgpt", "haystack llm", "building agents", "ai agents production",
    "agents are", "agentic", "llm framework", "llm frameworks", "llm abstraction", "mcp agents",
    "multi-agent", "agent orchestration", "cloudflare agents",
]
VENT = re.compile(
    r"\bwhy (we|i)\b|no longer|stop(ped)?\b|don.t|doesn.t|problem|harmful|pointless|\bbad\b|\bmess|hype|overhyped|"
    r"regret|leaving|ditch|without|you (don.t|do not) need|considered harmful|fail|broken|\bhard\b|myth|truth|lessons|"
    r"wrong|worse|hate|sucks|nightmare|bloat|too complex|complexity|unnecessary|rethink|against|skeptic|disappoint|trap|"
    r"mistake|struggl|difficult|not ready|reality|deleted|overrated|useless|lying|scam|bubble",
    re.I,
)


# named frameworks / SDKs: always on-topic
NAMED = re.compile(
    r"langchain|langgraph|langsmith|llamaindex|llama index|crewai|autogen|semantic kernel|agent frameworks?|"
    r"agents? sdk|agent development kit|agentkit|agentcore|strands agents|vercel ai sdk|ai sdk|pydantic ?ai|mastra|"
    r"dspy|smolagents|letta|memgpt|haystack|llm frameworks?|llm abstraction|agent orchestration|cloudflare agents",
    re.I,
)
# generic agent talk: only kept when the title itself is critical
GENERIC = re.compile(r"\bagents?\b|agentic|multi-agent|\bmcp\b", re.I)
NOISE = re.compile(r"coding agent|agentic coding|\bqwen|\bglm\b|gemini \d|starcraft|user.agent|\bclaws?\b|agentic ide", re.I)


def on_topic(title):
    if NAMED.search(title):
        return True
    ai = re.search(r"\\bAI\\b|LLM|agentic|multi-agent|\\bMCP\\b|autonomous|production|framework", title, re.I)
    return bool(GENERIC.search(title) and ai and VENT.search(title) and not NOISE.search(title))


def get(path):
    req = urllib.request.Request(API + path, headers={"User-Agent": "research-script"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def clean(text):
    if not text:
        return ""
    t = re.sub(r"<p>", "\n\n", text)
    t = re.sub(r'<a href="([^"]+)"[^>]*>.*?</a>', r"\1", t)
    t = re.sub(r"<pre><code>(.*?)</code></pre>", lambda m: "\n```\n" + m.group(1) + "\n```\n", t, flags=re.S)
    t = re.sub(r"</?i>", "*", t)
    t = re.sub(r"<[^>]+>", "", t)
    return html.unescape(t).strip()


def render(node, depth, out):
    for c in node.get("children") or []:
        if not c.get("text"):
            continue
        body = clean(c["text"]).replace("\n", "\n" + "> " * depth)
        out.append(f"{'> ' * depth}**{c.get('author')}** ({(c.get('created_at') or '')[:10]}):\n{'> ' * depth}{body}\n")
        render(c, depth + 1, out)


def count(node):
    return sum(1 + count(c) for c in node.get("children") or [])


def main():
    os.makedirs(OUT, exist_ok=True)
    found = {}
    for q in QUERIES:
        params = urllib.parse.urlencode({
            "query": q, "tags": "story", "restrictSearchableAttributes": "title",
            "numericFilters": f"num_comments>={MIN_COMMENTS}", "hitsPerPage": 40,
        })
        hits = get("search?" + params)["hits"]
        hits = [h for h in hits if on_topic(h.get("title") or "")]
        hits.sort(key=lambda h: -(h.get("num_comments") or 0))
        for h in hits[:PER_QUERY]:
            found.setdefault(h["objectID"], h)
        time.sleep(0.3)
    # critical-sounding titles first, then by comment volume
    ranked = sorted(found.values(), key=lambda h: (not VENT.search(h["title"]), -(h.get("num_comments") or 0)))[:MAX_THREADS]
    rows = []
    for h in ranked:
        item = get(f"items/{h['objectID']}")
        out = []
        render(item, 0, out)
        date = (item.get("created_at") or "")[:10]
        slug = re.sub(r"[^a-z0-9]+", "-", item["title"].lower()).strip("-")[:70]
        fn = f"{date}_{h['objectID']}_{slug}.md"
        n = count(item)
        with open(os.path.join(OUT, fn), "w") as f:
            f.write(f"---\nsource: hackernews\ntitle: {json.dumps(item['title'])}\n"
                    f"hn_url: https://news.ycombinator.com/item?id={h['objectID']}\nlinked_url: {item.get('url') or ''}\n"
                    f"date: {date}\npoints: {item.get('points')}\ncomments_saved: {n}\n"
                    f"critical_title: {bool(VENT.search(item['title']))}\n---\n\n# {item['title']}\n\n")
            if item.get("text"):
                f.write(clean(item["text"]) + "\n\n")
            f.write("## Comments\n\n" + "\n".join(out))
        rows.append((date, item["title"], item.get("points"), n, bool(VENT.search(item["title"])), fn, h["objectID"]))
        time.sleep(0.2)
    rows.sort(key=lambda r: -r[3])
    with open(os.path.join(OUT, "INDEX.md"), "w") as f:
        f.write("# Hacker News threads\n\n| Date | Title | Points | Comments | Critical title | File |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | [{r[1].replace('|', '/')}](https://news.ycombinator.com/item?id={r[6]}) | {r[2]} | {r[3]} | {'yes' if r[4] else ''} | [{r[5]}]({r[5]}) |\n")
    print(f"{len(rows)} threads, {sum(r[3] for r in rows)} comments, {sum(1 for r in rows if r[4])} with critical titles")
    for r in rows[:25]:
        print(f"  {r[0]} {r[3]:>4}c {'!' if r[4] else ' '} {r[1][:90]}")


if __name__ == "__main__":
    main()
