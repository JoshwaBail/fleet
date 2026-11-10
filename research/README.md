# Agent framework market research corpus

Raw material for analysing the AI agent framework / SDK category (LangChain and its peers).
Collected 10 November 2026.

## What is here

| Folder | Contents | Volume |
|---|---|---|
| `01-websites/` | Home, pricing, product, customers and about pages for 17 companies and SDKs, scraped with Firecrawl. One folder per company; `INDEX.md` lists every page. | 60 pages, ~118k words |
| `02-earnings-calls/` | Full earnings call transcripts for Microsoft, Alphabet, Amazon and Cloudflare, plus `agent-mentions.md` (every paragraph that mentions agents, in one file). | 14 calls, ~137k words |
| `03-reviews/` | Customer reviews as structured records (`reviews.jsonl`) and as one readable file per source. | 165 reviews, ~38k words |
| `04-forums/reddit/` | Reddit threads where developers complain about agent frameworks, one file per thread. | 73 threads, ~1,400 comments |
| `04-forums/hackernews/` | Hacker News threads about the named frameworks and about agents failing in production, with full comment trees. | 44 threads, ~4,750 comments |
| `raw/` | Unprocessed Firecrawl output that the folders above are built from, including the Reddit search results. | |
| `scripts/` | The collection and build scripts. | |

## Coverage

**Websites (17):** LangChain, LlamaIndex, CrewAI, Mastra, Pydantic AI, Agno, Haystack/deepset, Letta, DSPy,
OpenAI Agents SDK, Claude Agent SDK, Google ADK, Microsoft Agent Framework, AWS Strands/AgentCore,
Vercel AI SDK, Cloudflare Agents, Hugging Face smolagents.

**Earnings calls:** the public incumbents in the category. OpenAI, Anthropic, LangChain, Vercel and the rest are private and hold no earnings calls.

| Company | Calls |
|---|---|
| Microsoft | Q3 FY26, Q4 FY26, Q1 FY27 |
| Alphabet | Q4 2025, Q1 2026, Q2 2026, Q3 2026 |
| Amazon | Q1 2026, Q2 2026, Q3 2026 |
| Cloudflare | Q4 2025, Q1 2026, Q2 2026, Q3 2026 |

**Reviews by source:** G2 74, Trustpilot 41, Capterra 37, PeerSpot 13.
Products: LangChain, LangSmith, Amazon Bedrock, Gemini platform (Vertex AI), Microsoft Copilot Studio,
Azure AI Foundry, Vercel, Cloudflare Workers, CrewAI, n8n, Botpress. See `03-reviews/INDEX.md` for counts.

## Known gaps

- **G2 gives 10 reviews per product.** Pagination and star filters return the first page again, so each G2 product is capped at its 10 default-sorted reviews. Several G2 profiles (LlamaIndex, Dify, Copilot Studio) render no review text at all.
- **Reviews skew to adjacent products.** Pure developer frameworks have few reviews anywhere: LangChain has 12 in total here, CrewAI 2, and LlamaIndex, Mastra, Pydantic AI, Agno, Letta and DSPy have none. Most of the volume is n8n, Vercel, Bedrock and Copilot Studio. Trustpilot reviews for Vercel and n8n are mostly about billing and support, not agent tooling.
- **No app store reviews.** These are developer libraries with no app store listings.
- **Reddit is a sample, not every thread.** Threads were found through 29 searches and hold the first page of comments only (deeper reply pages were not followed). Coverage is strong for LangChain, LangGraph, CrewAI, Bedrock and the category in general; thin for Mastra, Vercel AI SDK, Cloudflare Agents and the OpenAI Agents SDK.
- **Hacker News selection is by title match**, so it includes launch threads (where the criticism is in the comments) and a few off-topic hits.
- **Websites are marketing pages, not full sites.** Docs were limited to one overview page each. Letta's product page timed out; CrewAI has no about or customers page.
- **One PeerSpot page was lost** (the first 10 Amazon Bedrock reviews), because the result was returned out-of-band and expired before it was saved.
- No bot-check or login walls were bypassed. Sites that refused a plain request were skipped.

## Rebuilding

```bash
python3 scripts/harvest.py            # Firecrawl results from the session transcript -> raw/
python3 scripts/build.py              # raw/ -> 01-websites, 03-reviews, 04-forums/reddit
python3 scripts/fetch_earnings.py     # Motley Fool transcripts -> 02-earnings-calls
python3 scripts/fetch_hn.py           # Hacker News via the Algolia API -> 04-forums/hackernews
```

`harvest.py` only works inside the Claude Code session that made the Firecrawl calls, since it reads that session's transcript.
`add_extra_earnings.py` adds the two calls Motley Fool's listing lacked (Alphabet and Cloudflare Q2 2026) from files downloaded separately.

## Sources and reuse

Earnings transcripts are from Motley Fool, Alphabet investor relations and Yahoo Finance. Reviews, Reddit
threads and Hacker News comments are third-party content written by named or pseudonymous individuals.
Reviewer names are left out of `03-reviews/`, but remain in `raw/`. Treat the corpus as private research
material rather than something to republish.
