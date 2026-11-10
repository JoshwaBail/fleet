#!/usr/bin/env python3
"""Leads from UK public data: awarded AI contracts and Innovate UK funded AI projects.

  Contracts Finder (gov.uk)       -> 05-leads/raw/public-contracts.jsonl
  UKRI Gateway to Research        -> 05-leads/raw/funded-projects.jsonl
"""
import datetime
import json
import os
import re
import time
import urllib.parse
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "05-leads", "raw")
UA = {"User-Agent": "Mozilla/5.0 (research script)", "Accept": "application/json", "Content-Type": "application/json"}
AI = re.compile(
    r"generative ai|gen ?ai\b|large language model|\bllms?\b|chatbot|chat bot|ai assistant|virtual assistant|"
    r"copilot|conversational ai|ai agent|agentic|ai[- ]powered|artificial intelligence",
    re.I,
)
ACADEMIC = re.compile(r"universit|college|school of|institute|nhs|research council|hospital", re.I)


def http(url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def contracts():
    seen, rows = set(), []
    for kw in ["generative AI", "chatbot", "large language model", "AI assistant", "conversational AI", "Copilot", "AI agent"]:
        page = 1
        while page <= 4:
            res = http("https://www.contractsfinder.service.gov.uk/api/rest/2/search_notices/json",
                       {"searchCriteria": {"keyword": kw, "statuses": ["Awarded"], "publishedFrom": "2025-03-01"}, "size": 100, "page": page})
            items = res.get("noticeList") or []
            for it in items:
                n = it["item"]
                text = f"{n.get('title') or ''} {n.get('description') or ''}"
                value = n.get("awardedValue") or 0
                if n["id"] in seen or not AI.search(n.get("title") or "") and len(AI.findall(text)) < 2:
                    continue
                if not 10_000 <= value <= 1_500_000:
                    continue
                seen.add(n["id"])
                m = AI.search(text)
                quote = re.sub(r"\s+", " ", text[max(0, m.start() - 120): m.start() + 160]).strip()
                rows.append({
                    "company": (n.get("organisationName") or "").strip(), "website": "", "location": n.get("postcode") or "",
                    "sector": "public sector", "size_hint": "unknown", "segment": "public-contract",
                    "signal": f"Awarded '{(n.get('title') or '').strip()}' to {n.get('awardedSupplier')} for £{value:,.0f}"
                              + (" (SME supplier)" if n.get("awardedToSme") else ""),
                    "signal_date": (n.get("awardedDate") or n.get("publishedDate") or "")[:10],
                    "evidence_url": f"https://www.contractsfinder.service.gov.uk/notice/{n['id']}",
                    "evidence_quote": quote[:280],
                    "fit_note": "A paid AI project with an outside supplier; someone must own quality and cost once the contract ends.",
                    "named_sponsor": "", "supplier": n.get("awardedSupplier") or "", "value_gbp": value,
                    "supplier_is_sme": bool(n.get("awardedToSme")),
                })
            if len(items) < 100:
                break
            page += 1
            time.sleep(0.5)
    rows.sort(key=lambda r: r["signal_date"], reverse=True)
    return rows


def funded():
    seen, rows = set(), []
    since = datetime.datetime(2024, 6, 1).timestamp() * 1000
    for term in ['"large language model"', '"generative AI"', "LLM", "chatbot", '"AI agent"', '"AI assistant"', "agentic"]:
        for page in range(1, 9):
            url = "https://gtr.ukri.org/api/search/project?" + urllib.parse.urlencode({"term": term, "page": page, "fetchSize": 100})
            try:
                bean = http(url)["facetedSearchResultBean"]
            except Exception as e:
                print("gtr", term, page, e)
                break
            results = bean.get("results") or []
            for r in results:
                pc = r["projectComposition"]
                p, org = pc["project"], (pc.get("leadResearchOrganisation") or {})
                fund = p.get("fund") or {}
                funder = ((fund.get("funder") or {}).get("name")) or ""
                name = (org.get("name") or "").strip()
                if p["id"] in seen or funder != "Innovate UK" or (fund.get("start") or 0) < since:
                    continue
                if not name or ACADEMIC.search(name) or not AI.search(p.get("title") or ""):
                    continue
                seen.add(p["id"])
                start = datetime.datetime.fromtimestamp(fund["start"] / 1000).strftime("%Y-%m-%d")
                end = datetime.datetime.fromtimestamp((fund.get("end") or fund["start"]) / 1000).strftime("%Y-%m-%d")
                rows.append({
                    "company": name.title(), "website": "", "location": "UK", "sector": "unknown", "size_hint": "unknown",
                    "segment": "funded-project",
                    "signal": f"Innovate UK {p.get('grantCategory') or 'grant'} of £{fund.get('valuePounds') or 0:,} for '{p.get('title')}' ({start} to {end})",
                    "signal_date": start, "evidence_url": f"https://gtr.ukri.org/projects?ref={p.get('grantReference')}",
                    "evidence_quote": p.get("title") or "",
                    "fit_note": "Grant-funded AI build with a fixed end date; the system needs an owner once the funding stops.",
                    "named_sponsor": "", "value_gbp": fund.get("valuePounds") or 0, "project_end": end,
                })
            if len(results) < 100:
                break
            time.sleep(0.4)
    rows.sort(key=lambda r: r["signal_date"], reverse=True)
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("public-contracts", contracts), ("funded-projects", funded)):
        rows = fn()
        with open(os.path.join(OUT, name + ".jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows))
        for r in rows[:12]:
            print("  ", r["signal_date"], "|", r["company"][:40], "|", r["signal"][:120])
