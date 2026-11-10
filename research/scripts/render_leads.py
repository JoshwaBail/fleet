#!/usr/bin/env python3
"""Render 05-leads/leads.csv as a searchable, filterable HTML table (05-leads/leads.html)."""
import csv
import json
import os

LEADS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "05-leads")
rows = list(csv.DictReader(open(os.path.join(LEADS, "leads.csv"))))
data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lead list</title>
<style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--muted:#6b6b66;--line:#e2e1dc;--card:#fff;--a:#0f6b4f;--b:#8a5a00;--c:#6b6b66;--link:#1a5fb4;--tag:#ecebe6}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a09f99;--line:#33322f;--card:#1f1f1d;--a:#5fd0a5;--b:#e2b45a;--c:#a09f99;--link:#8ab4f8;--tag:#2a2a27}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;z-index:2}
h1{font-size:17px;margin:0 0 8px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
input,select{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 8px}
input{flex:1;min-width:160px}
#count{color:var(--muted);margin-left:auto}
.wrap{overflow-x:auto;padding:0 16px 24px}
table{border-collapse:collapse;width:100%;min-width:1080px}
th,td{text-align:left;vertical-align:top;padding:8px 10px;border-bottom:1px solid var(--line)}
th{font-size:12px;color:var(--muted);font-weight:600;white-space:nowrap;cursor:pointer}
td.co{font-weight:600;min-width:170px}td.build{min-width:300px}td.sig{min-width:280px}
.sub{color:var(--muted);font-weight:400;font-size:12.5px}
.p{display:inline-block;width:22px;height:22px;line-height:22px;text-align:center;border-radius:50%;font-weight:700;font-size:12px;border:1.5px solid currentColor}
.pA{color:var(--a)}.pB{color:var(--b)}.pC{color:var(--c)}
.tag{display:inline-block;background:var(--tag);border-radius:4px;padding:1px 7px;font-size:12px;font-weight:600;margin-bottom:4px}
.tag.off{color:var(--muted);font-weight:400}
a{color:var(--link)}q{color:var(--muted);font-size:12.5px;display:block;margin-top:4px}
.flag{color:var(--b);font-size:12px}
</style></head><body>
<header><h1>Lead list: what we would build for each organisation</h1>
<div class="controls">
<input id="q" type="search" placeholder="Search company, sector, signal, build">
<select id="pri"><option value="">All priorities</option><option>A</option><option>B</option><option>C</option></select>
<select id="bt"><option value="">All build types</option></select>
<select id="seg"><option value="">All sources</option></select>
<span id="count"></span></div></header>
<div class="wrap"><table><thead><tr>
<th data-k="priority">Priority</th><th data-k="company">Organisation</th><th data-k="build_type">What we would build</th>
<th data-k="signal">Signal</th><th data-k="signal_date">Date</th><th>Evidence</th></tr></thead><tbody id="tb"></tbody></table></div>
<script>
const rows = __DATA__;
const SEG = {"public-contract":"Public contract","hiring":"Job advert","launched-ai":"AI launch","agency":"Agency","funded-project":"Innovate UK grant"};
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fill = (id, vals, label) => vals.forEach(v => { const o = document.createElement("option"); o.value = v; o.textContent = label ? (label[v] || v) : v; document.getElementById(id).appendChild(o); });
fill("seg", [...new Set(rows.map(r => r.segment))], SEG);
fill("bt", [...new Set(rows.map(r => r.build_type).filter(Boolean))].sort());
let sortKey = null, asc = true;
function draw() {
  const q = document.getElementById("q").value.toLowerCase(), p = document.getElementById("pri").value,
        s = document.getElementById("seg").value, b = document.getElementById("bt").value;
  let out = rows.filter(r => (!p || r.priority === p) && (!s || r.segment === s) && (!b || r.build_type === b) && (!q || Object.values(r).join(" ").toLowerCase().includes(q)));
  if (sortKey) out = [...out].sort((x, y) => String(x[sortKey]).localeCompare(String(y[sortKey])) * (asc ? 1 : -1));
  document.getElementById("count").textContent = out.length + " of " + rows.length;
  document.getElementById("tb").innerHTML = out.map(r => {
    const site = r.website && r.website !== "unknown" ? `<a href="https://${esc(r.website.replace(/^https?:\/\//, ""))}" target="_blank" rel="noopener">${esc(r.website)}</a>` : "";
    const meta = [SEG[r.segment] || r.segment, r.sector, r.location, r.size_hint !== "unknown" ? r.size_hint : ""].filter(x => x && x !== "unknown").map(esc).join(" · ");
    const safeUrl = /^https?:\/\//.test(r.evidence_url) ? esc(r.evidence_url) : "#";
    const off = r.build_type === "Skip" || r.build_type === "Low fit";
    return `<tr><td><span class="p p${esc(r.priority)}">${esc(r.priority)}</span></td>
<td class="co">${esc(r.company)}<div class="sub">${meta}</div><div class="sub">${site}</div></td>
<td class="build"><span class="tag${off ? " off" : ""}">${esc(r.build_type)}</span><div>${esc(r.proposed_build)}</div></td>
<td class="sig">${esc(r.signal)}${r.named_sponsor ? `<div class="sub">Named: ${esc(r.named_sponsor)}</div>` : ""}<q>${esc(r.evidence_quote)}</q></td>
<td style="white-space:nowrap">${esc(r.signal_date)}</td>
<td><a href="${safeUrl}" target="_blank" rel="noopener">source</a>${r.evidence_checked === "no" ? '<div class="flag">not confirmed</div>' : ""}</td></tr>`;
  }).join("");
}
document.querySelectorAll("th[data-k]").forEach(th => th.onclick = () => { asc = sortKey === th.dataset.k ? !asc : true; sortKey = th.dataset.k; draw(); });
["q", "pri", "seg", "bt"].forEach(id => document.getElementById(id).addEventListener("input", draw));
draw();
</script></body></html>'''

with open(os.path.join(LEADS, "leads.html"), "w") as f:
    f.write(PAGE.replace("__DATA__", data))
print(len(rows), "rows rendered")
