#!/usr/bin/env python3
"""Organise harvested raw data into the readable corpus folders.

  01-websites/<company>/*.md      marketing, pricing and docs-overview pages
  03-reviews/reviews.jsonl + .md  one record per customer review
  04-forums/reddit/*.md           one file per Reddit thread

Safe to re-run; it rebuilds these folders from research/raw/ each time.
"""
import datetime
import glob
import hashlib
import json
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")

HOSTS = {
    "langchain.com": "langchain", "llamaindex.ai": "llamaindex", "crewai.com": "crewai", "mastra.ai": "mastra",
    "ai.pydantic.dev": "pydantic-ai", "pydantic.dev": "pydantic-ai", "agno.com": "agno",
    "haystack.deepset.ai": "haystack-deepset", "deepset.ai": "haystack-deepset",
    "letta.com": "letta", "docs.letta.com": "letta", "dspy.ai": "dspy",
    "openai.github.io": "openai-agents-sdk", "developers.openai.com": "openai-agents-sdk", "openai.com": "openai-agents-sdk",
    "platform.claude.com": "claude-agent-sdk", "claude.com": "claude-agent-sdk",
    "google.github.io": "google-adk", "adk.dev": "google-adk", "cloud.google.com": "google-adk",
    "learn.microsoft.com": "microsoft-agent-framework", "azure.microsoft.com": "microsoft-agent-framework",
    "strandsagents.com": "aws-strands-agentcore", "aws.amazon.com": "aws-strands-agentcore",
    "ai-sdk.dev": "vercel-ai-sdk", "vercel.com": "vercel-ai-sdk",
    "agents.cloudflare.com": "cloudflare-agents", "developers.cloudflare.com": "cloudflare-agents",
    "huggingface.co": "huggingface-smolagents", "github.com": "huggingface-smolagents",
}


def front_matter(text):
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    meta = dict(l.split(": ", 1) for l in m.group(1).splitlines() if ": " in l) if m else {}
    return meta, text[m.end():] if m else text


def reset(d):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)


# ---------------------------------------------------------------- websites
def build_websites():
    out = os.path.join(ROOT, "01-websites")
    reset(out)
    rows = []
    for host_dir in sorted(glob.glob(os.path.join(RAW, "pages", "*"))):
        host = os.path.basename(host_dir).removeprefix("www.")
        company = HOSTS.get(host)
        if not company:
            continue
        for f in sorted(glob.glob(os.path.join(host_dir, "*.md"))):
            text = open(f).read()
            meta, body = front_matter(text)
            if meta.get("status_code", "200") not in ("200", "") or len(body.split()) < 40:
                continue
            name = f"{host}__{os.path.basename(f)}"
            os.makedirs(os.path.join(out, company), exist_ok=True)
            shutil.copy(f, os.path.join(out, company, name))
            rows.append((company, meta.get("source_url", ""), len(body.split()), f"{company}/{name}"))
    with open(os.path.join(out, "INDEX.md"), "w") as f:
        f.write("# Website scrapes (via Firecrawl)\n\n| Company / SDK | Page | Words | File |\n|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]} | {r[2]} | [{r[3]}]({r[3]}) |\n")
    return len(rows), len({r[0] for r in rows})


# ---------------------------------------------------------------- reviews
def g2_reviews(path):
    meta, body = front_matter(open(path).read())
    product = re.search(r"products/([^/]+)/", meta.get("source_url", ""))
    product = product.group(1) if product else "?"
    parts = re.split(r"- \[Report a Concern\]\(https://www\.g2\.com/survey_responses/(\d+)/concerns/new\)", body)
    for i in range(1, len(parts), 2):
        sid, chunk = parts[i], parts[i + 1]
        head = parts[i - 1].split("Show More")[-1] if i > 1 else parts[0].split("Here's how.")[-1]
        head_lines = [l.strip() for l in head.splitlines() if l.strip() and not l.startswith(("[", "!", "\\")) and "](" not in l and "Reviewer" not in l and l.strip() != "More Options"]
        review = chunk.split("Show More")[0]
        title = re.search(r'^"(.+)"$', review, re.M)
        rating = re.search(r"^(\d(?:\.\d)?)/5$", review, re.M)
        date = re.search(r"\d{1,2}/\d{1,2}/\d{4}", head)
        text = review[rating.end():] if rating else review
        text = text.replace("Review collected by and hosted on G2.com.", "").strip()
        size = next((l for l in head_lines if "emp." in l), "")
        role = next((l for l in head_lines[1:] if "emp." not in l and not re.search(r"\d/\d", l) and "submitted" not in l), "")
        if not text:
            continue
        yield {
            "source": "G2", "product": product, "rating": float(rating.group(1)) if rating else None, "rating_scale": 5,
            "date": date.group(0) if date else "", "title": title.group(1) if title else "", "reviewer_role": role,
            "company_size": size, "text": re.sub(r"\n{3,}", "\n\n", text),
            "url": f"https://www.g2.com/products/{product}/reviews#survey-response-{sid}", "id": f"g2-{sid}",
        }


MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def trustpilot_reviews(path):
    meta, body = front_matter(open(path).read())
    product = meta.get("source_url", "").split("/review/")[-1].split("?")[0]
    pat = re.compile(
        r"!\[Rated (\d) out of 5 stars\]\([^)]*\)\s*\n\s*\[\*\*(.+?)\*\*\]\((https://www\.trustpilot\.com/reviews/([0-9a-f]+))\)\s*\n(.*?)\n\s*((?:%s) \d{1,2}, \d{4})\s*\n" % MONTHS,
        re.S,
    )
    for m in pat.finditer(body):
        yield {
            "source": "Trustpilot", "product": product, "rating": float(m.group(1)), "rating_scale": 5,
            "date": m.group(6), "title": m.group(2), "reviewer_role": "", "company_size": "",
            "text": m.group(5).strip(), "url": m.group(3), "id": f"tp-{m.group(4)}",
        }


def capterra_reviews(path):
    meta, body = front_matter(open(path).read())
    product = re.search(r"/p/\d+/([^/]+)/", meta.get("source_url", ""))
    product = product.group(1) if product else "?"
    parts = re.split(r'^### "(.+)"\s*$', body, flags=re.M)
    for i in range(1, len(parts), 2):
        title, chunk = parts[i], parts[i + 1].split("Review source")[0]
        rating = re.search(r"^(\d\.\d)$", chunk, re.M)
        date = re.search(r"^((?:%s) \d{1,2}, \d{4})$" % MONTHS, chunk, re.M)
        text = chunk[date.end():].strip() if date else chunk.strip()
        if len(text.split()) < 15:
            continue
        yield {
            "source": "Capterra", "product": product, "rating": float(rating.group(1)) if rating else None, "rating_scale": 5,
            "date": date.group(1) if date else "", "title": title, "reviewer_role": "", "company_size": "",
            "text": text, "url": meta.get("source_url", ""), "id": "cap-" + hashlib.sha1((title + text).encode()).hexdigest()[:10],
        }


def peerspot_reviews(path):
    d = json.load(open(path))
    data = ((d.get("response") or {}).get("data")) or {}
    product = (d["request"].get("options") or {}).get("slug") or (data.get("product") or {}).get("slug", "?")
    for r in data.get("reviews") or []:
        sections = "\n\n".join(f"**{s.get('heading')}**\n{s.get('text')}" for s in r.get("sections") or [] if s.get("text"))
        yield {
            "source": "PeerSpot", "product": product, "rating": r.get("rating"), "rating_scale": r.get("rating_scale"),
            "date": r.get("date_label", ""), "title": r.get("title", ""), "reviewer_role": r.get("author_job_title", ""),
            "company_size": "", "text": sections or r.get("text", ""), "url": r.get("url", ""), "id": f"ps-{r.get('id')}",
        }


def build_reviews():
    out = os.path.join(ROOT, "03-reviews")
    reset(out)
    seen, reviews = set(), []
    sources = (
        [(g2_reviews, f) for f in sorted(glob.glob(os.path.join(RAW, "pages", "www.g2.com", "*.md")))]
        + [(trustpilot_reviews, f) for f in sorted(glob.glob(os.path.join(RAW, "pages", "www.trustpilot.com", "*.md")))]
        + [(capterra_reviews, f) for f in sorted(glob.glob(os.path.join(RAW, "pages", "www.capterra.com", "*.md")))]
        + [(peerspot_reviews, f) for f in sorted(glob.glob(os.path.join(RAW, "alexandria", "peerspot-com", "software-reviews-reviews", "*.json")))
           + sorted(glob.glob(os.path.join(RAW, "manual", "peerspot*.json")))]
    )
    for fn, path in sources:
        for r in fn(path):
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            reviews.append(r)
    with open(os.path.join(out, "reviews.jsonl"), "w") as f:
        for r in reviews:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    for r in reviews:
        counts[(r["source"], r["product"])] = counts.get((r["source"], r["product"]), 0) + 1
    by_src = {}
    for r in reviews:
        by_src.setdefault(r["source"], []).append(r)
    for src, rs in by_src.items():
        with open(os.path.join(out, f"{src.lower()}.md"), "w") as f:
            f.write(f"# {src} reviews ({len(rs)})\n")
            for r in rs:
                f.write(f"\n---\n\n## {r['product']}: {r['title'] or '(untitled)'}\n\n"
                        f"- Rating: {r['rating']}/{r['rating_scale']}\n- Date: {r['date']}\n"
                        + (f"- Reviewer: {r['reviewer_role']} {r['company_size']}\n" if r["reviewer_role"] or r["company_size"] else "")
                        + f"- Source: {r['url']}\n\n{r['text']}\n")
    with open(os.path.join(out, "INDEX.md"), "w") as f:
        f.write(f"# Customer reviews ({len(reviews)})\n\nStructured records are in `reviews.jsonl`; readable copies are in one file per source.\n\n"
                "| Source | Product | Reviews |\n|---|---|---|\n")
        for (s, p), n in sorted(counts.items()):
            f.write(f"| {s} | {p} | {n} |\n")
    return len(reviews), counts


# ---------------------------------------------------------------- reddit
def render_comments(items, depth, out):
    n = 0
    for c in items or []:
        body = (c.get("body") or "").strip()
        if not body:
            continue
        n += 1
        pre = "> " * depth
        out.append(f"{pre}**u/{c.get('author')}** ({c.get('score')} pts, {(c.get('created_at_iso') or '')[:10]}):\n{pre}"
                   + body.replace("\n", "\n" + pre) + "\n")
        n += render_comments(((c.get("replies") or {}).get("items")), depth + 1, out)
    return n


def build_reddit():
    out = os.path.join(ROOT, "04-forums", "reddit")
    reset(out)
    rows, seen = [], set()
    files = glob.glob(os.path.join(RAW, "alexandria", "scrapecreators", "reddit-post-comments", "*.json")) + \
        glob.glob(os.path.join(RAW, "manual", "*.json"))
    for path in sorted(files):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        data = ((d.get("response") or {}).get("data")) if "response" in d else None
        if data is None:  # a hand-copied raw tool result
            try:
                data = d["data"]["alexandria"][0]["data"]
            except Exception:
                continue
        post = (data or {}).get("post") or {}
        if not post.get("id") or post["id"] in seen:
            continue
        seen.add(post["id"])
        lines = []
        n = render_comments(data.get("comments"), 0, lines)
        date = datetime.datetime.fromtimestamp(post.get("created_utc") or 0, datetime.timezone.utc).strftime("%Y-%m-%d")
        slug = re.sub(r"[^a-z0-9]+", "-", (post.get("title") or "").lower()).strip("-")[:60]
        fn = f"{date}_{post.get('subreddit')}_{post['id']}_{slug}.md"
        url = post.get("url") or (d.get("request", {}).get("options", {}) or {}).get("url", "")
        with open(os.path.join(out, fn), "w") as f:
            f.write(f"---\nsource: reddit\nsubreddit: r/{post.get('subreddit')}\ntitle: {json.dumps(post.get('title'))}\n"
                    f"url: {url}\ndate: {date}\nscore: {post.get('score')}\ncomments_total: {post.get('num_comments')}\n"
                    f"comments_saved: {n}\n---\n\n# {post.get('title')}\n\n**u/{post.get('author')}** in r/{post.get('subreddit')}:\n\n"
                    f"{post.get('selftext') or ''}\n\n## Comments\n\n" + "\n".join(lines))
        rows.append((date, post.get("subreddit"), post.get("title"), post.get("num_comments"), n, fn, url))
    rows.sort(key=lambda r: -(r[3] or 0))
    with open(os.path.join(out, "INDEX.md"), "w") as f:
        f.write(f"# Reddit threads ({len(rows)})\n\n| Date | Subreddit | Title | Comments (total) | Saved | File |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | r/{r[1]} | [{(r[2] or '').replace('|', '/')}]({r[6]}) | {r[3]} | {r[4]} | [{r[5][:40]}...]({r[5]}) |\n")
    return len(rows), sum(r[4] for r in rows)


if __name__ == "__main__":
    pages, companies = build_websites()
    print(f"websites: {pages} pages across {companies} companies")
    n, counts = build_reviews()
    print(f"reviews: {n}")
    for k, v in sorted(counts.items()):
        print("   ", k, v)
    threads, comments = build_reddit()
    print(f"reddit: {threads} threads, {comments} comments")
