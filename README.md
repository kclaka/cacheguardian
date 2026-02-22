<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/version-0.3.0-green.svg" alt="Version 0.3.0">
  <img src="https://img.shields.io/badge/tests-244%20passed-brightgreen.svg" alt="Tests 244 passed">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="MIT License">
</p>

<h1 align="center">CacheGuardian</h1>

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
- Using expensive models when a cheaper one would cache the same prompt
- And [dozens of other subtle mistakes](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)

The result? You pay **full price** for tokens the provider *already computed*, and you don't even know it's happening. The only signal you get is a number buried in the API response.

## The Solution

CacheGuardian wraps your existing SDK client in **one line of code**. It detects cache breaks locally in **< 1 millisecond**, automatically fixes the most common mistakes, recommends cheaper models when your prompt doesn't meet cache thresholds, and logs exactly how much money you're saving — or wasting — on every single call.

```python
import cacheguardian
import anthropic

# Before: client = anthropic.Anthropic()
client = cacheguardian.wrap(anthropic.Anthropic())

# Everything else stays exactly the same.
# CacheGuardian works silently in the background.
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a helpful assistant.",
    tools=[...],
    messages=[...]
)
```

```
[cacheguardian] L1 HIT | Cache hit 94.2% | Saved $0.0340 | Session total: $1.24 saved
```

---

## Installation

```bash
pip install cacheguardian
```

Then install it alongside the provider(s) you use:

```bash
# Anthropic Claude
pip install cacheguardian anthropic

# OpenAI GPT / o-series
pip install cacheguardian openai

# Google Gemini
pip install cacheguardian google-genai

# All providers
pip install "cacheguardian[all]"

# Optional: Redis for distributed caching (L2)
pip install "cacheguardian[redis]"
```

**Requirements:** Python 3.10+

---

## Quick Start

### Anthropic

```python
import cacheguardian
import anthropic

client = cacheguardian.wrap(anthropic.Anthropic())

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

**What CacheGuardian does automatically:**

| Optimization | What it does |
|---|---|
| Injects `cache_control` | Adds top-level auto-caching if you forgot it |
| Sorts tools | Alphabetical by name — prevents ordering-based misses |
| Stabilizes JSON keys | Sorts all dict keys recursively for consistent hashing |
| Intermediate breakpoints | Adds `cache_control` markers every 15 messages when you exceed 20 blocks |
| Smart TTL | Switches from 5-minute to 1-hour TTL when your request intervals are > 5 min |

### OpenAI

```python
import cacheguardian
import openai

client = cacheguardian.wrap(openai.OpenAI())

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain prompt caching."},
    ],
)
```

**What CacheGuardian does automatically:**

| Optimization | What it does |
|---|---|
| Derives `prompt_cache_key` | Generates a stable routing key so your requests hit the same physical cache hardware |
| Reorders content | Moves system messages before user messages for better prefix overlap |
| Sorts tools | Same as Anthropic — deterministic ordering |
| Bucket padding | Pads system content to the next 128-token cache boundary |
| Smart retention | Sets `prompt_cache_retention="24h"` when your request intervals are > 10 min |

### Google Gemini

```python
import cacheguardian
from google import genai

client = cacheguardian.wrap(genai.Client())

response = client.models.generate_content(
    model="gemini-2.5-flash",
    system_instruction="You are an expert analyst.",
    contents=[{"role": "user", "parts": [{"text": "Summarize this document."}]}],
)
```

**What CacheGuardian does automatically:**

| Optimization | What it does |
|---|---|
| Normalizes kwargs | Moves `system_instruction` and `tools` into `config` dict for SDK v1.x compatibility |
| Implicit to Explicit promotion | Creates a `CachedContent` object when the cost-benefit math says it saves money |
| TTL optimization | Calculates optimal TTL from your request frequency to minimize storage costs |
| Zombie cache cleanup | Persists a cache registry to disk — cleans up orphaned caches even after crashes |
| Storage cost tracking | Tracks Gemini's per-hour storage fees separately so you see real ROI |

### Async Clients

CacheGuardian supports async clients out of the box. All diffing and metric extraction runs via `asyncio.to_thread()` so the event loop is never blocked.

```python
import cacheguardian
import anthropic

client = cacheguardian.wrap(anthropic.AsyncAnthropic())

# Use await as normal
response = await client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

Works with `AsyncAnthropic`, `AsyncOpenAI`, and Gemini's async methods.

---

## Model Recommendations

CacheGuardian detects when your prompt is **below a model's cache threshold** and recommends a cheaper alternative that would actually cache the content.

```python
import cacheguardian

# Example: 1,500 tokens on Opus (4,096-token cache minimum)
# → Won't cache. Sonnet (1,024-token min) would cache and save ~60%.
rec = cacheguardian.recommend(
    provider="anthropic",
    model="claude-opus-4-6",
    token_count=1500,
)

if rec:
    print(rec.recommended_model)           # "claude-sonnet-4-6"
    print(f"{rec.savings_percentage:.0f}%") # "60%"
    print(rec.capability_note)             # "claude-sonnet-4 is less capable than claude-opus-4"
```

### Automatic hints in the pipeline

When `log_level` is `DEBUG` or `INFO`, CacheGuardian logs model hints automatically:

```
[cacheguardian] MODEL HINT — 1,500 tokens < claude-opus-4-6's cache min (4,096)
  Consider: claude-sonnet-4-6 (cache min: 1,024 tokens, saves ~$0.0070/req, 94% reduction)
  Note: claude-sonnet-4 (high-capability) is less capable than claude-opus-4 (frontier).
```

### In dry-run mode

```python
dr = cacheguardian.dry_run(
    client,
    model="claude-opus-4-6",
    max_tokens=1024,
    system="Your system prompt here...",
    messages=[{"role": "user", "content": "Hello"}],
)

if dr.model_recommendation:
    print(dr.model_recommendation.recommended_model)
    print(dr.model_recommendation.savings_percentage)
```

### With a wrapped client

```python
client = cacheguardian.wrap(anthropic.Anthropic())

# Inherits config and pricing overrides from the wrapped client
rec = cacheguardian.recommend(
    client=client,
    model="claude-opus-4-6",
    token_count=1500,
)
```

### Key behaviors

- **Same-provider only** — never recommends cross-provider models (no Opus to GPT-4o suggestions)
- **Capability warnings** — tiered severity based on how far down you're stepping:
  - 1-tier drop (Opus to Sonnet): mild note
  - 2+ tier drop (Opus to Haiku): stern warning about reduced reasoning capability
- **Output cost awareness** — factors in output token costs, not just input savings
- **Threshold: >10% savings** — won't recommend marginal switches
- **Zero overhead in production** — skipped when `log_level` is `WARNING` or `ERROR`

---

## How It Works

CacheGuardian uses a **tiered L1 / L2 / L3 architecture** inspired by CPU cache hierarchies and production inference gateways:

```
┌───────────────────────────────────────────────────────────────┐
│                     Your Application                          │
├───────────────────────────────────────────────────────────────┤
│                    cacheguardian.wrap()                          │
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

Instead of hashing your entire prompt, CacheGuardian hashes it in **segments**:

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
- **Model recommendations** — Detect when a cheaper model would cache content that your current model can't

---

## Dry Run Mode

Test whether your prompt would hit the cache — without making an API call.

```python
import cacheguardian

client = cacheguardian.wrap(anthropic.Anthropic())

# First, make a real call to establish the cache
client.messages.create(model="claude-sonnet-4-20250514", ...)

# Now test a new prompt against the cache for free
result = cacheguardian.dry_run(
    client,
    model="claude-sonnet-4-20250514",
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "New question"}],
)

print(result.would_hit_cache)        # True / False
print(result.prefix_match_depth)     # "75% — 3/4 segments match (diverged at message[1])"
print(result.estimated_savings)      # 0.034
print(result.warnings)               # [CacheBreakWarning(...)]
print(result.model_recommendation)   # ModelRecommendation or None
```

```
[cacheguardian] DRY RUN — would HIT cache | Estimated savings: $0.0340
```

Zero cost. Instant feedback. Iterate on your prompt structure before spending a cent.

---

## Configuration

```python
client = cacheguardian.wrap(
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

    # Model recommendations (suggest cheaper models when below cache threshold)
    model_recommendations=True,     # default: True

    # OpenAI: custom function to derive prompt_cache_key
    cache_key_fn=lambda session: f"user_{session.session_id}",

    # L2: Redis URL for distributed environments
    l2_backend="redis://localhost:6379",

    # Privacy: add timing jitter to prevent cache-timing side-channel attacks
    privacy_mode=False,             # default: False
    privacy_jitter_ms=(50, 200),    # jitter range in ms — default: (50, 200)
    privacy_jitter_mode="fixed",    # "fixed" | "adaptive" — default: "fixed"
)
```

### Pricing Overrides

CacheGuardian ships with default pricing tables for all supported models. If prices change, override them:

```python
from cacheguardian.config import PricingConfig

client = cacheguardian.wrap(
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

For Gemini (explicit `CachedContent`) and Anthropic (1-hour TTL), CacheGuardian uses a break-even formula to decide when upgrading is worth the cost:

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

CacheGuardian tracks your request frequency automatically and promotes when **N** crosses the break-even threshold.

---

## System Prompt Templates

Dynamic content in system prompts (dates, user names, config values) is one of the most common cache killers. CacheGuardian provides a template pattern that keeps the cache-friendly static part as the system prompt and injects dynamic values as messages instead:

```python
from cacheguardian.core.optimizer import SystemPromptTemplate

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
client = cacheguardian.wrap(
    anthropic.Anthropic(),
    l2_backend="redis://localhost:6379",
)
```

**What L2 enables:**

- **Cross-worker prefix sharing** — Worker B knows that Worker A already warmed the cache for a given prefix
- **Gemini cache coordination** — Prevents redundant `CachedContent` creation fees when multiple workers process the same content
- **Rate limit coordination** — Shared request counting across workers
- **Graceful degradation** — If Redis is unavailable, CacheGuardian falls back to L1-only with zero errors

---

## Gemini Safety Lock

Gemini's explicit caches incur storage fees of **$4.50 per million tokens per hour**. If your process crashes without cleaning up, those caches keep billing.

CacheGuardian solves this with a **disk-persisted cache registry**:

```python
from cacheguardian.providers.gemini import GeminiProvider

# On startup, clean up any zombie caches from previous crashes
provider = GeminiProvider(config, gemini_client=client)
cleaned = provider.cleanup_stale_caches(max_age_hours=2.0)
print(f"Cleaned {cleaned} orphaned caches")
```

The registry is written to `~/.cache/cacheguardian/gemini_registry.json` on every cache creation. On next startup, any caches not accessed within the threshold are automatically deleted.

---

## Architecture

```
cacheguardian/
├── __init__.py                  # Public API: wrap(), dry_run(), recommend()
├── types.py                     # Core data types (10 dataclasses)
├── config.py                    # Configuration + pricing tables
├── py.typed                     # PEP 561 typed package marker
│
├── cache/
│   ├── fingerprint.py           # Normalize → segment → rolling SHA-256 hash
│   ├── l1.py                    # Local dict: <1ms fingerprint comparison
│   └── l2.py                    # Optional Redis: cross-worker coordination
│
├── core/
│   ├── advisor.py               # Model recommendation engine
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

1. **Exact prefix matching only.** No semantic or embedding-based caching. CacheGuardian guarantees 100% accuracy — it will never serve a "similar" cached result for a different question.

2. **Never modify the response.** CacheGuardian transforms the *request* (sorting tools, injecting cache_control) and *logs* after the response. The response object you receive is identical to what the raw SDK would return.

3. **Composition over inheritance.** SDK clients are wrapped, not subclassed. This makes CacheGuardian resilient to SDK version changes.

4. **Optional everything.** Install only the provider SDKs you use. Redis is optional. Privacy mode is optional. Every feature degrades gracefully when its dependency is absent.

---

## Contributing

Contributions are welcome! Here's how to set up the development environment:

```bash
git clone https://github.com/kclaka/cacheguardian.git
cd cacheguardian

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,all]"

# Run the unit tests
pytest -v

# Run live integration tests (requires API keys in .env)
pytest tests/test_live.py -v -s
```

**244 tests** cover all modules: fingerprinting, L1 cache, transforms, session tracking, cost calculations, promotion logic, model recommendations, and live end-to-end tests across all three providers.

---

## License

MIT

---

<p align="center">
  <strong>CacheGuardian</strong> — because the best API call is the one you don't pay full price for.
</p>
