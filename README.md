<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/version-0.1.0-green.svg" alt="Version 0.1.0">
  <img src="https://img.shields.io/badge/tests-125%20passed-brightgreen.svg" alt="Tests 125 passed">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="MIT License">
</p>

<h1 align="center">Cache Guard</h1>

<p align="center">
  <strong>Stop overpaying for LLM API calls you've already made.</strong>
</p>

<p align="center">
  A drop-in Python middleware that wraps Anthropic, OpenAI, and Gemini SDKs to<br>
  automatically optimize prompt caching and show you exactly how much money you're saving.
</p>

---

## The Problem

Every major LLM provider offers prompt caching — cached tokens cost **10-90% less** than regular input tokens. But developers silently break their cache due to non-obvious pitfalls:

- Non-deterministic tool ordering
- System prompt mutations between requests
- Model switches mid-session
- Dynamic content placed before static content
- And [dozens of other subtle mistakes](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)

The result? You pay **full price** for tokens the provider *already computed*, and you don't even know it's happening. The only signal you get is a number buried in the API response.

## The Solution

Cache Guard wraps your existing SDK client in **one line of code**. It detects cache breaks locally in **< 1 millisecond**, automatically fixes the most common mistakes, and logs exactly how much money you're saving — or wasting — on every single call.

```python
import cache_guard
import anthropic

# Before: client = anthropic.Anthropic()
client = cache_guard.wrap(anthropic.Anthropic())

# Everything else stays exactly the same.
# Cache Guard works silently in the background.
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a helpful assistant.",
    tools=[...],
    messages=[...]
)
```

```
[cache-guard] L1 HIT | Cache hit 94.2% | Saved $0.0340 | Session total: $1.24 saved
```

---

## Installation

```bash
pip install cache-guard
```

Then install it alongside the provider(s) you use:

```bash
# Anthropic Claude
pip install cache-guard anthropic

# OpenAI GPT / o-series
pip install cache-guard openai

# Google Gemini
pip install cache-guard google-genai

# Optional: Redis for distributed caching (L2)
pip install cache-guard redis
```

**Requirements:** Python 3.10+

---

## Quick Start

### Anthropic

```python
import cache_guard
import anthropic

client = cache_guard.wrap(anthropic.Anthropic())

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a helpful coding assistant.",
    tools=[
        {
            "name": "search_code",
            "description": "Search the codebase",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ],
    messages=[{"role": "user", "content": "Find all TODO comments"}],
)
```

**What Cache Guard does automatically:**

| Optimization | What it does |
|---|---|
| Injects `cache_control` | Adds top-level auto-caching if you forgot it |
| Sorts tools | Alphabetical by name — prevents ordering-based misses |
| Stabilizes JSON keys | Sorts all dict keys recursively for consistent hashing |
| Intermediate breakpoints | Adds `cache_control` markers every 15 messages when you exceed 20 blocks |
| Smart TTL | Switches from 5-minute to 1-hour TTL when your request intervals are > 5 min |

### OpenAI

```python
import cache_guard
import openai

client = cache_guard.wrap(openai.OpenAI())

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain prompt caching."},
    ],
)
```

**What Cache Guard does automatically:**

| Optimization | What it does |
|---|---|
| Derives `prompt_cache_key` | Generates a stable routing key so your requests hit the same physical cache hardware |
| Reorders content | Moves system messages before user messages for better prefix overlap |
| Sorts tools | Same as Anthropic — deterministic ordering |
| Smart retention | Sets `prompt_cache_retention="24h"` when your request intervals are > 10 min |
| 1024-token threshold | Suppresses false-positive cache warnings for prompts under 1024 tokens |

### Google Gemini

```python
import cache_guard
from google import genai

client = cache_guard.wrap(genai.Client())

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Summarize this document.",
    config={"system_instruction": "You are an expert analyst."},
)
```

**What Cache Guard does automatically:**

| Optimization | What it does |
|---|---|
| Implicit → Explicit promotion | Creates a `CachedContent` object when the cost-benefit math says it saves money |
| TTL optimization | Calculates optimal TTL from your request frequency to minimize storage costs |
| Zombie cache cleanup | Persists a cache registry to disk — cleans up orphaned caches even after crashes |
| Storage cost tracking | Tracks Gemini's per-hour storage fees separately so you see real ROI |

### Async Clients

Cache Guard supports async clients out of the box. All diffing and metric extraction runs via `asyncio.to_thread()` so the event loop is never blocked.

```python
import cache_guard
import anthropic

client = cache_guard.wrap(anthropic.AsyncAnthropic())

# Use await as normal
response = await client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

Works with `AsyncAnthropic`, `AsyncOpenAI`, and Gemini's async methods.

---

## How It Works

Cache Guard uses a **tiered L1 / L2 / L3 architecture** inspired by CPU cache hierarchies and production inference gateways:

```
┌───────────────────────────────────────────────────────────────┐
│                     Your Application                          │
├───────────────────────────────────────────────────────────────┤
│                    cache_guard.wrap()                          │
├──────────┬────────────────────────────────────┬───────────────┤
│          │                                    │               │
│    L1    │  Local Python Dict                 │    < 1 ms     │
│          │  Rolling segment fingerprints      │               │
│          │  Instant divergence detection      │               │
│          │                                    │               │
├──────────┼────────────────────────────────────┼───────────────┤
│          │                                    │               │
│    L2    │  Redis (optional)                  │   2 – 10 ms   │
│          │  Cross-worker prefix sharing       │               │
│          │  Gemini CachedContent coordination │               │
│          │                                    │               │
├──────────┼────────────────────────────────────┼───────────────┤
│          │                                    │               │
│    L3    │  Provider API                      │  100 ms – 2s  │
│          │  The actual transformer KV-cache   │               │
│          │                                    │               │
└──────────┴────────────────────────────────────┴───────────────┘
```

### L1: The Secret Sauce

Instead of hashing your entire prompt, Cache Guard hashes it in **segments**:

```
system prompt  →  sha256  →  "a3f1..."
tools block    →  sha256  →  "b7e2..."
message[0]     →  sha256  →  "c9d4..."
message[1]     →  sha256  →  "e1a6..."
```

When your next request comes in, it compares segment hashes sequentially. The moment one doesn't match, it knows exactly **which segment diverged** — without scanning the full content. This is how it achieves < 1ms detection even on 100k-token prompts.

### What L1 Enables

- **Pre-emptive warnings** — Detect a 1-character typo in a 50,000-token system prompt *before* the request is sent and the bill is generated
- **`dry_run` mode** — Test your prompt structure against the cache without spending a cent
- **Actionable suggestions** — Not just "cache missed" but "your system prompt changed — use a system-reminder message instead"

---

## Dry Run Mode

Test whether your prompt would hit the cache — without making an API call.

```python
import cache_guard

client = cache_guard.wrap(anthropic.Anthropic())

# First, make a real call to establish the cache
client.messages.create(model="claude-sonnet-4-20250514", ...)

# Now test a new prompt against the cache for free
result = cache_guard.dry_run(
    client,
    model="claude-sonnet-4-20250514",
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "New question"}],
)

print(result.would_hit_cache)       # True / False
print(result.prefix_match_depth)    # "75% — 3/4 segments match (diverged at message[1])"
print(result.estimated_savings)     # 0.034
print(result.warnings)              # [CacheBreakWarning(...)]
```

```
[cache-guard] DRY RUN — would HIT cache | Estimated savings: $0.0340
```

Zero cost. Instant feedback. Iterate on your prompt structure before spending a cent.

---

## Configuration

```python
client = cache_guard.wrap(
    anthropic.Anthropic(),

    # Automatically fix safe issues (tool sorting, cache_control injection)
    auto_fix=True,                  # default: True

    # TTL strategy for Anthropic
    ttl_strategy="auto",            # "auto" | "5m" | "1h" — default: "auto"

    # Strict mode: raise exceptions instead of warnings on cache breaks
    strict_mode=False,              # default: False

    # Logging verbosity
    log_level="INFO",               # "DEBUG" | "INFO" | "WARNING" | "ERROR"

    # Alert when cache hit rate drops below this threshold
    min_cache_hit_rate=0.7,         # default: 0.7

    # OpenAI: custom function to derive prompt_cache_key
    cache_key_fn=lambda session: f"user_{session.session_id}",

    # L2: Redis URL for distributed environments
    l2_backend="redis://localhost:6379",

    # Privacy: add timing jitter to prevent cache-timing side-channel attacks
    privacy_mode=False,             # default: False
    privacy_jitter_ms=(50, 200),    # jitter range in ms — default: (50, 200)
)
```

### Pricing Overrides

Cache Guard ships with default pricing tables for all supported models. If prices change, override them:

```python
from cache_guard.config import PricingConfig

client = cache_guard.wrap(
    anthropic.Anthropic(),
    pricing_overrides={
        "anthropic": {
            "claude-sonnet-4-20250514": PricingConfig(
                base_input=3.00,        # $ per million tokens
                cache_read=0.30,        # 90% discount
                cache_write_5m=3.75,    # 25% premium (5-min TTL)
                cache_write_1h=6.00,    # 100% premium (1-hour TTL)
                output=15.00,
            ),
        },
    },
)
```

---

## The Promotion Formula

For Gemini (explicit `CachedContent`) and Anthropic (1-hour TTL), Cache Guard uses a break-even formula to decide when upgrading is worth the cost:

```
N  >  (C_write + S × T) / (C_input − C_cache_read)
```

| Symbol | Meaning |
|---|---|
| **N** | Expected number of requests reusing this content |
| **C_write** | One-time cost to write the cache |
| **S** | Storage cost per hour (Gemini only; 0 for Anthropic) |
| **T** | TTL in hours |
| **C_input** | Standard input token cost |
| **C_cache_read** | Discounted cache read cost |

Cache Guard tracks your request frequency automatically and promotes when **N** crosses the break-even threshold.

---

## System Prompt Templates

Dynamic content in system prompts (dates, user names, config values) is one of the most common cache killers. Cache Guard provides a template pattern that keeps the cache-friendly static part as the system prompt and injects dynamic values as messages instead:

```python
from cache_guard.core.optimizer import SystemPromptTemplate

template = SystemPromptTemplate(
    "You are a helpful assistant. The current date is {date}. User: {user_name}."
)

# Use the static template as your system prompt (never changes → always cached)
system_prompt = template.static_part

# Inject dynamic values as a system-reminder message (appended, not prefixed)
reminder = template.render_dynamic(date="2026-02-20", user_name="Alice")
# → "Updated context: date=2026-02-20, user_name=Alice"
```

---

## Distributed Caching with Redis (L2)

In serverless or multi-worker environments, different workers may not know about each other's sessions. The optional L2 cache solves this:

```python
client = cache_guard.wrap(
    anthropic.Anthropic(),
    l2_backend="redis://localhost:6379",
)
```

**What L2 enables:**

- **Cross-worker prefix sharing** — Worker B knows that Worker A already warmed the cache for a given prefix
- **Gemini cache coordination** — Prevents redundant `CachedContent` creation fees when multiple workers process the same content
- **Rate limit coordination** — Shared request counting across workers
- **Graceful degradation** — If Redis is unavailable, Cache Guard falls back to L1-only with zero errors

---

## Gemini Safety Lock

Gemini's explicit caches incur storage fees of **$4.50 per million tokens per hour**. If your process crashes without cleaning up, those caches keep billing.

Cache Guard solves this with a **disk-persisted cache registry**:

```python
from cache_guard.providers.gemini import GeminiProvider

# On startup, clean up any zombie caches from previous crashes
provider = GeminiProvider(config, gemini_client=client)
cleaned = provider.cleanup_stale_caches(max_age_hours=2.0)
print(f"Cleaned {cleaned} orphaned caches")
```

The registry is written to `~/.cache/cache_guard/gemini_registry.json` on every cache creation. On next startup, any caches not accessed within the threshold are automatically deleted.

---

## Architecture

```
cache_guard/
├── __init__.py                  # Public API: wrap(), dry_run(), configure()
├── types.py                     # Core data types (9 dataclasses)
├── config.py                    # Configuration + pricing tables
│
├── cache/
│   ├── fingerprint.py           # Normalize → segment → rolling SHA-256 hash
│   ├── l1.py                    # Local dict: <1ms fingerprint comparison
│   └── l2.py                    # Optional Redis: cross-worker coordination
│
├── core/
│   ├── session.py               # Session state tracking across API calls
│   ├── optimizer.py             # Transforms: sort tools, stabilize JSON, templates
│   ├── differ.py                # Segment-level diff engine with cost estimation
│   ├── metrics.py               # Cost formulas for all 3 providers
│   ├── promoter.py              # Break-even promotion logic
│   └── logger.py                # Rich colored terminal output
│
├── providers/
│   ├── base.py                  # Abstract provider interface
│   ├── anthropic.py             # cache_control, breakpoints, TTL, JSON stabilization
│   ├── openai.py                # prompt_cache_key, retention, content reordering
│   └── gemini.py                # CachedContent lifecycle, promotion, safety lock
│
├── middleware/
│   ├── interceptor.py           # Sync wrapper: L1 → transform → L3 → metrics
│   └── async_interceptor.py     # Async wrapper: non-blocking via asyncio.to_thread()
│
└── persistence/
    └── cache_registry.py        # Disk-persisted Gemini cache safety lock
```

---

## Design Principles

1. **Exact prefix matching only.** No semantic or embedding-based caching. Cache Guard guarantees 100% accuracy — it will never serve a "similar" cached result for a different question.

2. **Never modify the response.** Cache Guard transforms the *request* (sorting tools, injecting cache_control) and *logs* after the response. The response object you receive is identical to what the raw SDK would return.

3. **Composition over inheritance.** SDK clients are wrapped, not subclassed. This makes Cache Guard resilient to SDK version changes.

4. **Optional everything.** Install only the provider SDKs you use. Redis is optional. Privacy mode is optional. Every feature degrades gracefully when its dependency is absent.

---

## Contributing

Contributions are welcome! Here's how to set up the development environment:

```bash
git clone https://github.com/kclaka/Cache-Guard.git
cd Cache-Guard

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"

# Run the test suite
pytest -v
```

**125 tests** cover all modules: fingerprinting, L1 cache, transforms, session tracking, cost calculations, promotion logic, and all three providers.

---

## License

MIT

---

<p align="center">
  <strong>Cache Guard</strong> — because the best API call is the one you don't pay full price for.
</p>
