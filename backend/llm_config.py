"""
Multi-Model LLM Configuration: task-based provider routing with automatic fallback.

Why this exists (and why it's more than "pick a provider"):
Earlier versions of this module picked one provider for all text tasks and one for vision.
That's simple, but it wastes an important fact: this app's tasks are NOT equally demanding.
grade_documents_node runs a trivial yes/no relevance classification on every single query;
generate_node produces the actual "elite financial advisor" answer -- citations, financial
reasoning, tone -- which is the task quality genuinely matters for; the vision tool does
multimodal page reading. Routing all three to the same model either overpays for the cheap
task or underpays for the expensive one.

This module assigns each task a *role* (grader / synthesis / vision), each role an ordered
chain of providers, and automatically falls back to the next provider in the chain if the
current one errors (rate limit, quota exhaustion, auth failure, etc.) -- instead of hanging
through a provider's own retry/backoff or crashing outright. This is exactly the failure mode
that hit this project directly: three vision calls in quick succession exhausted Gemini's
free-tier per-minute quota (RESOURCE_EXHAUSTED, limit 5/min) and the request sat retrying for
nearly a minute. With a fallback chain, that third call would have gone straight to Claude
instead.

Configuration (comma-separated provider chains, tried left to right):
    GRADER_PROVIDERS=openai,groq        (default if unset)
    SYNTHESIS_PROVIDERS=openai,gemini   (default if unset)
    VISION_PROVIDERS=openai,gemini      (default if unset)

Within a provider, the model is resolved per role, so the grader and the synthesiser do not
have to share one model id:
    OPENAI_GRADER_MODEL / OPENAI_SYNTHESIS_MODEL   (role-specific, wins if set)
    OPENAI_MODEL                                    (provider-wide)
    built-in defaults: gpt-4.1-mini for grading, gpt-4.1 for synthesis and vision
The same {PROVIDER}_{ROLE}_MODEL pattern works for every provider (GEMINI_GRADER_MODEL,
GROQ_SYNTHESIS_MODEL, ...).

Valid provider names are 'gemini', 'claude', 'groq' and 'ollama'. Groq is deliberately absent
from the vision defaults because the Groq models this app uses have no vision capability;
Ollama is absent from *all* defaults because it only works if someone is actually running a
local Ollama daemon -- it is an opt-in you turn on by naming it in a *_PROVIDERS var, never
something you inherit by accident.

Backward compatible with the older single-provider vars (LLM_PROVIDER, VISION_PROVIDER): if
the role-specific *_PROVIDERS var isn't set but the corresponding old var is, that becomes a
single-item chain (i.e. the old exact behavior, no fallback) rather than silently introducing
fallback someone didn't ask for. Set the new *_PROVIDERS vars to opt into multi-model routing.


-------------------------------------------------------------------------------------------
Why fallback alone was not enough (the hardening pass)
-------------------------------------------------------------------------------------------
A fallback chain only helps if the *chain itself* is sane, and in practice it usually isn't:
the deployment's env is half-configured, a provider name is misspelled, or the local Ollama
that the chain depends on isn't running. The original implementation treated every one of
those as "provider raised an exception -> try the next one", which turned a one-line config
mistake into a slow, confusing, multi-provider error cascade. Five specific behaviours were
added to fix that, and each is worth understanding because they are the difference between
"the app is down and nobody knows why" and "the log says which env var to set":

1. Pre-flight usability checks. A provider whose API key env var is empty cannot possibly
   succeed. Trying it anyway costs a client construction, a DNS lookup and an HTTP round trip
   *per request* to learn something we already knew from os.getenv(). Unusable providers are
   therefore skipped before any network I/O, with one deduplicated log line explaining why.
   The same applies to Ollama: a ~0.3s TCP connect to its port (result cached for
   _PROBE_TTL_SECONDS) is far cheaper than letting an HTTP client discover ECONNREFUSED on
   every call, and it lets us say "Ollama isn't running" instead of surfacing a raw
   ConnectError from three layers down.

2. Up-front chain validation. `GRADER_PROVIDERS=gemni,groq` used to raise ValueError from the
   client builder, which the generic `except Exception` swallowed as if Gemini itself had
   failed -- so a typo silently degraded to the next provider and the log blamed the wrong
   thing. Unknown names are now detected when the chain is resolved and reported as typos,
   naming the valid options.

3. Never `raise last_error` when nothing ran. If a chain resolves to zero providers (blank
   env var, all-typo'd names, every provider unusable) the fallback loop never executes and
   `last_error` is still None -- `raise None` produces "TypeError: exceptions must derive from
   BaseException", perfectly masking the actual cause. Every one of those cases now raises a
   RuntimeError that names the exact env var to set and which providers are currently usable.

4. One bounded retry, on the last provider only. max_retries=0 on the underlying clients is
   deliberate: a provider's own exponential backoff can hang a request for a minute, and the
   whole point of a chain is that falling over to the next provider is faster than waiting.
   But the *last* provider has nothing to fall over to, so a transient 503/overload there used
   to fail the entire request instantly. It now gets exactly one retry after a short sleep,
   and only for errors that are clearly transient (overloaded / unavailable / timeout) --
   never for rate limits (whose reset window is far longer than we can afford to block on),
   auth failures or connection refusals, none of which get better in 1.5 seconds.

5. Caches that can recover, and are thread-safe. The client caches are read and written from
   FastAPI's threadpool, so they are guarded by a lock. They are also keyed by the resolved
   model/base-url rather than just the provider name, and a client is evicted when it fails
   with an auth or connection error, so a client constructed during a transient bad state
   (empty key at import time, Ollama not yet up) does not stay poisoned for the process's
   lifetime.

get_provider_status() exposes all of the above for /health, cheaply and without ever raising,
because "which provider is actually serving my requests, and why did the other one get
skipped" is the single most common operational question this module has to answer.
"""
import os
import base64
import logging
import socket
import threading
import time
from urllib.parse import urlparse

logger = logging.getLogger("LLMConfig")

ROLES = ("grader", "synthesis", "vision")

# Ollama is intentionally NOT in any default chain: it only works when a local daemon is
# running, so inheriting it by default would mean every deployment without Ollama silently
# carries a dead link in its chain. Name it explicitly in a *_PROVIDERS var to opt in.
DEFAULT_CHAINS = {
    "grader": ["openai", "groq"],
    "synthesis": ["openai", "gemini"],
    "vision": ["openai", "gemini"],
}

KNOWN_PROVIDERS = ("openai", "gemini", "claude", "groq", "ollama")

# Providers that can actually accept an image. Groq's chat models as used here cannot, so
# naming groq in VISION_PROVIDERS is a config error we should report rather than discover
# through a confusing API-side rejection.
VISION_CAPABLE_PROVIDERS = ("openai", "gemini", "claude", "ollama")

# API-key env vars per provider. Gemini accepts either name (the google-genai client itself
# checks both), so it gets a tuple; a provider is "configured" if any of its vars is non-empty.
PROVIDER_KEY_ENV = {
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "claude": ("ANTHROPIC_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "ollama": (),  # local daemon, no key
}

# OpenAI's reasoning-family models (o1/o3/o4, and the GPT-5 line) reject `temperature` with a
# 400: "Unsupported value: 'temperature' does not support 0 with this model". Every other
# model this app talks to is set to temperature=0 deliberately -- a financial-advisory answer
# and a yes/no relevance grade both want determinism, not creativity -- so rather than drop
# that setting globally we omit the parameter only for the families that refuse it.
#
# This is a prefix match on purpose: OpenAI ships dated snapshots (gpt-5.1-2026-01-30) and
# size variants (gpt-5-mini, o4-mini) under the same family, and a hardcoded list of exact
# ids would silently stop matching the moment a new snapshot is pinned.
OPENAI_NO_TEMPERATURE_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _openai_supports_temperature(model: str) -> bool:
    m = (model or "").strip().lower()
    return not m.startswith(OPENAI_NO_TEMPERATURE_PREFIXES)

# Bounded transient retry (see point 4 in the module docstring). One retry, short sleep.
_TRANSIENT_RETRIES = 1
_TRANSIENT_RETRY_SLEEP = 1.5

# Reachability probe for Ollama: short timeouts, result cached briefly so that /health and a
# burst of requests don't each pay for a connect.
_PROBE_TIMEOUT = 0.35
_TAGS_TIMEOUT = 0.8
_PROBE_TTL_SECONDS = 30.0

_client_lock = threading.RLock()
_text_llm_cache = {}
_vision_llm_cache = {}

_probe_lock = threading.RLock()
_ollama_probe = {
    "checked_at": 0.0, "ok": False, "detail": "not probed", "url": None, "models": None,
}

# Deduplication for "skipping provider X because Y" lines. Without this, a provider that is
# unusable for a structural reason (empty API key) would emit an identical warning on every
# single request -- which is noise, and noise is how real warnings get missed.
_log_once_lock = threading.RLock()
_logged_once = {}


def _log_once(key: str, message: str, level: int = logging.WARNING):
    """Logs `message` only when it differs from the last message logged under `key`."""
    with _log_once_lock:
        if _logged_once.get(key) == message:
            return
        _logged_once[key] = message
    logger.log(level, message)


# ---------------------------------------------------------------------------
# Provider chain resolution + validation
# ---------------------------------------------------------------------------

def _chain_env_var(role: str) -> str:
    return f"{role.upper()}_PROVIDERS"


def _validate_role(role: str) -> str:
    if not isinstance(role, str) or role.lower() not in ROLES:
        raise ValueError(
            f"Unknown LLM role {role!r}. Valid roles are: {', '.join(ROLES)}."
        )
    return role.lower()


def _get_role_provider_chain(role: str) -> list:
    """Returns the raw, *unvalidated* provider chain for a role (order preserved, duplicates
    removed). Kept as-is for backward compatibility -- validation happens in _resolve_chain()
    so that callers who just want to know 'what did the operator ask for' can still see typos
    rather than having them silently filtered out."""
    role = _validate_role(role)
    env_key = _chain_env_var(role)
    override = os.getenv(env_key)
    raw = None

    if override is not None:
        # The role-specific var, once set at all, governs -- including when it is set to
        # blank. See the NOTE below: an explicit blank must not silently fall through to
        # either the legacy var or the built-in default.
        raw = [p.strip().lower() for p in override.split(",") if p.strip()]
    elif role == "vision" and (os.getenv("VISION_PROVIDER") or "").strip():
        # Backward compatibility with the old single-provider vars.
        raw = [os.getenv("VISION_PROVIDER").strip().lower()]
    elif role in ("grader", "synthesis") and (os.getenv("LLM_PROVIDER") or "").strip():
        raw = [os.getenv("LLM_PROVIDER").strip().lower()]

    if raw is None:
        # NOTE: an env var that is *set but blank* (VISION_PROVIDERS=) deliberately does NOT
        # fall through to the default. Blanking a chain is something an operator does on
        # purpose, and quietly substituting the default would hide that. It resolves to an
        # empty chain, which _plan_chain() turns into an explicit, actionable error.
        raw = list(DEFAULT_CHAINS[role])

    seen = set()
    chain = []
    for p in raw:
        if p not in seen:
            seen.add(p)
            chain.append(p)
    return chain


def _chain_source(role: str) -> str:
    """Human-readable description of where a role's chain came from -- the first thing you
    want to know when the chain isn't what you expected."""
    role = _validate_role(role)
    env_key = _chain_env_var(role)
    if os.getenv(env_key) is not None:
        return f"env {env_key}"
    if role == "vision" and (os.getenv("VISION_PROVIDER") or "").strip():
        return "env VISION_PROVIDER (legacy)"
    if role in ("grader", "synthesis") and (os.getenv("LLM_PROVIDER") or "").strip():
        return "env LLM_PROVIDER (legacy)"
    return "built-in default"


def _resolve_chain(role: str) -> dict:
    """Splits a role's configured chain into what we can actually use and why the rest was
    dropped. Pure env/socket inspection -- no LLM API calls, so it is safe to call from
    /health as well as from the request path."""
    role = _validate_role(role)
    requested = _get_role_provider_chain(role)

    known, unknown = [], []
    for p in requested:
        (known if p in KNOWN_PROVIDERS else unknown).append(p)

    usable, unusable = [], {}
    for p in known:
        ok, reason = _provider_usable(p, role)
        if ok:
            usable.append(p)
        else:
            unusable[p] = reason

    return {
        "role": role,
        "source": _chain_source(role),
        "requested": requested,
        "unknown": unknown,
        "usable": usable,
        "unusable": unusable,
    }


# ---------------------------------------------------------------------------
# Usability checks -- "can this provider possibly work right now?", answered without
# spending a request against it.
# ---------------------------------------------------------------------------

def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _fetch_ollama_tags(url: str):
    """Returns the list of locally pulled model names, or None if we couldn't find out.

    This is a single local GET /api/tags -- it loads no model, spends no tokens and hits no
    third party. It is worth the milliseconds because "Ollama is running but the model you
    configured was never pulled" is otherwise indistinguishable, from the outside, from every
    other 404, and it is a *very* common state: OLLAMA_MODEL defaults to a name the operator
    may never have pulled. Returning None on any failure is deliberate -- an unreadable tag
    list must never be treated as "the model is missing", because that would wrongly disable a
    working provider."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/tags", timeout=_TAGS_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        models = payload.get("models") or []
        names = [m.get("name") or m.get("model") for m in models if isinstance(m, dict)]
        return [n for n in names if n]
    except Exception as e:
        logger.debug(f"Could not read Ollama tag list from {url}: {type(e).__name__}: {e}")
        return None


def _probe_ollama(force: bool = False) -> tuple:
    """Cheap reachability check for the Ollama daemon. Returns (ok, detail).

    A plain socket connect (plus the local tag list, see _fetch_ollama_tags): no model is
    loaded, no tokens are spent, and the timeouts are fractions of a second. It exists because
    ECONNREFUSED discovered by an HTTP client three libraries deep produces a stack trace
    nobody can act on, whereas "Ollama is not reachable at http://localhost:11434" tells you
    exactly what to do. The result is cached for _PROBE_TTL_SECONDS so a burst of requests --
    or a monitoring system hammering /health -- costs one probe, not one per call."""
    url = _ollama_base_url()
    now = time.monotonic()
    with _probe_lock:
        fresh = (
            not force
            and _ollama_probe["url"] == url
            and (now - _ollama_probe["checked_at"]) < _PROBE_TTL_SECONDS
        )
        if fresh:
            return _ollama_probe["ok"], _ollama_probe["detail"]

    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 11434)

    models = None
    try:
        with socket.create_connection((host, port), timeout=_PROBE_TIMEOUT):
            ok, detail = True, f"reachable at {url}"
        models = _fetch_ollama_tags(url)
    except Exception as e:
        ok = False
        detail = (
            f"not reachable at {url} ({type(e).__name__}). Start it with `ollama serve` "
            f"(and `ollama pull {os.getenv('OLLAMA_MODEL', 'llama3.2')}`), point "
            f"OLLAMA_BASE_URL at a running instance, or drop 'ollama' from the chain."
        )

    _mark_ollama(ok, detail, models)
    return ok, detail


def _mark_ollama(ok: bool, detail: str, models=None):
    """Records the outcome of a probe *or* of a real call. A successful call is the strongest
    possible evidence that Ollama is up, and a connection error during a call is the strongest
    evidence it went away -- both should update the probe cache so the next request doesn't
    have to rediscover it."""
    with _probe_lock:
        _ollama_probe.update({
            "checked_at": time.monotonic(),
            "ok": ok,
            "detail": detail,
            "url": _ollama_base_url(),
            "models": models if models is not None else (_ollama_probe.get("models") if ok else None),
        })


def _ollama_models() -> list:
    with _probe_lock:
        return list(_ollama_probe.get("models") or [])


def _ollama_has_model(wanted: str) -> bool:
    """Tag names carry a `:tag` suffix (`llava` is stored as `llava:latest`), so a configured
    name without a colon matches any tag of that model. Unknown tag list -> assume present."""
    with _probe_lock:
        models = _ollama_probe.get("models")
    if models is None:
        return True
    wanted = (wanted or "").strip()
    if not wanted:
        return True
    for name in models:
        if name == wanted:
            return True
        if ":" not in wanted and name.split(":", 1)[0] == wanted:
            return True
    return False


def _provider_usable(provider: str, role: str = None) -> tuple:
    """Returns (usable, reason_if_not). Never raises, never calls a hosted LLM API."""
    try:
        if provider not in KNOWN_PROVIDERS:
            return False, (
                f"unknown provider name (valid: {', '.join(KNOWN_PROVIDERS)})"
            )

        if role == "vision" and provider not in VISION_CAPABLE_PROVIDERS:
            return False, (
                f"'{provider}' has no vision-capable model in this app -- remove it from "
                f"VISION_PROVIDERS (usable vision providers: "
                f"{', '.join(VISION_CAPABLE_PROVIDERS)})"
            )

        if provider == "ollama":
            ok, detail = _probe_ollama()
            if not ok:
                return False, f"Ollama {detail}"
            if role is not None:
                wanted = (
                    os.getenv("OLLAMA_VISION_MODEL", "llava") if role == "vision"
                    else os.getenv("OLLAMA_MODEL", "llama3.2")
                )
                if not _ollama_has_model(wanted):
                    available = _ollama_models()
                    return False, (
                        f"Ollama is running at {_ollama_base_url()} but model '{wanted}' is "
                        f"not pulled. Run `ollama pull {wanted}`, or point "
                        f"{'OLLAMA_VISION_MODEL' if role == 'vision' else 'OLLAMA_MODEL'} at "
                        f"one you have ({', '.join(available) if available else 'none pulled'})."
                    )
            return True, None

        key_vars = PROVIDER_KEY_ENV.get(provider, ())
        if key_vars and not any((os.getenv(v) or "").strip() for v in key_vars):
            return False, (
                f"no API key -- set {' or '.join(key_vars)} in backend/.env"
            )
        return True, None
    except Exception as e:  # defensive: usability checks must never break a request
        logger.debug(f"usability check for '{provider}' raised {type(e).__name__}: {e}")
        return True, None


# ---------------------------------------------------------------------------
# Error classification -- log wording *and* retry policy. Getting "which provider failed and
# why" into one readable line is the main operational need this module has.
# ---------------------------------------------------------------------------

def _classify_error(e: Exception) -> str:
    """Rough classification used for log wording and for deciding whether a retry could
    possibly help. Fallback to the next provider happens on ANY error regardless of this."""
    msg = f"{type(e).__name__}: {e}".lower()

    # Connection problems first: a refused/unreachable endpoint often *also* mentions the
    # URL, model name or a numeric code that would otherwise match a later rule.
    if any(k in msg for k in (
        "connection refused", "econnrefused", "connect call failed", "failed to establish",
        "max retries exceeded", "name or service not known", "nodename nor servname",
        "getaddrinfo", "connectionerror", "connecterror", "network is unreachable",
    )):
        return "connection / provider unreachable"
    if any(k in msg for k in (
        "401", "403", "unauthenticated", "unauthorized", "permission denied",
        "api key", "api_key", "authentication", "invalid_api_key", "credential",
    )):
        return "auth / API key"
    if any(k in msg for k in (
        "429", "rate limit", "rate_limit", "resource_exhausted", "quota", "too many requests",
    )):
        return "rate limit / quota"
    if any(k in msg for k in (
        "overloaded", "529", "503", "502", "504", "service unavailable", "unavailable",
        "internal server error", "try again later",
    )):
        return "provider overloaded"
    if any(k in msg for k in ("timeout", "timed out", "deadline exceeded")):
        return "timeout"
    if any(k in msg for k in ("model not found", "not found, try pulling", "404", "no such model")):
        return "model not found / not pulled"
    if any(k in msg for k in (
        "400", "invalid_request", "unsupported", "does not support", "not supported",
        "validationerror", "schema",
    )):
        return "bad request / unsupported feature"
    return "error"


# Only these classes of failure can plausibly be fixed by waiting ~1.5 seconds. Rate limits
# reset on a much longer window (Gemini free tier: per-minute), auth and connection failures
# are structural, and "model not found" needs an `ollama pull`. Retrying any of those would
# just add latency to a request that is already going to fail.
_TRANSIENT_KINDS = ("provider overloaded", "timeout")


def _is_transient(e: Exception) -> bool:
    return _classify_error(e) in _TRANSIENT_KINDS


def _failure_hint(provider: str, kind: str) -> str:
    """The 'so what do I do about it' half of an error message."""
    if kind == "auth / API key":
        keys = PROVIDER_KEY_ENV.get(provider, ())
        if keys:
            return f"Check {' / '.join(keys)} in backend/.env."
        return "Check the provider credentials."
    if kind == "rate limit / quota":
        return (
            f"'{provider}' is out of quota for now -- add another provider to this chain so "
            f"the next request can fall over instead of failing."
        )
    if kind == "connection / provider unreachable":
        if provider == "ollama":
            return f"Is `ollama serve` running at {_ollama_base_url()}?"
        return "Check network egress from this host."
    if kind == "model not found / not pulled":
        if provider == "ollama":
            return (
                f"Run `ollama pull {os.getenv('OLLAMA_MODEL', 'llama3.2')}` "
                f"(vision: `ollama pull {os.getenv('OLLAMA_VISION_MODEL', 'llava')}`)."
            )
        if provider == "openai":
            return (
                "Check OPENAI_MODEL / OPENAI_GRADER_MODEL / OPENAI_SYNTHESIS_MODEL / "
                "OPENAI_VISION_MODEL in backend/.env. Note that access to a given model also "
                "depends on the account tier -- `curl https://api.openai.com/v1/models -H "
                "\"Authorization: Bearer $OPENAI_API_KEY\"` lists what this key can actually use."
            )
        return "Check the *_MODEL env var for this provider -- the model name looks wrong."
    if kind == "bad request / unsupported feature" and provider == "openai":
        return (
            "If this names `temperature`, the configured model is a reasoning-family model; "
            "add its prefix to OPENAI_NO_TEMPERATURE_PREFIXES in llm_config.py."
        )
    if kind == "provider overloaded":
        return "Transient on the provider side; retried once already."
    return ""


# ---------------------------------------------------------------------------
# Per-provider client construction. Cached per (provider, model, base_url) rather than per
# provider name alone, so that changing GEMINI_MODEL (or pointing OLLAMA_BASE_URL somewhere
# else) yields a new client instead of silently reusing the old one.
# ---------------------------------------------------------------------------

def _role_model(provider: str, role: str, default_env: str, fallback: str) -> str:
    """Resolves the model id for a (provider, role) pair.

    Role-specific override first (`OPENAI_GRADER_MODEL`), then the provider-wide var
    (`OPENAI_MODEL`), then the built-in default.

    Why the per-role layer exists at all: this module's entire premise is that the grader
    (a yes/no relevance classification that runs on *every* query) and the synthesiser (the
    actual advisory answer) have completely different quality requirements. Routing them to
    separate provider *chains* achieved that only when the chains named different providers.
    Once both roles point at OpenAI -- which is the common setup -- a single `OPENAI_MODEL`
    would put a frontier model on the cheap high-frequency task and quietly multiply the bill
    for no quality gain. A role-aware model id restores the original intent within one
    provider.
    """
    role_specific = os.getenv(f"{provider.upper()}_{role.upper()}_MODEL", "").strip()
    if role_specific:
        return role_specific
    return os.getenv(default_env, "").strip() or fallback


def _text_llm_spec(provider: str, role: str = "synthesis") -> dict:
    if provider == "openai":
        # gpt-4.1-mini for grading, gpt-4.1 for synthesis. Both support tool calling, native
        # structured output and images, and both accept temperature=0 (unlike the o-series /
        # gpt-5 reasoning line -- see OPENAI_NO_TEMPERATURE_PREFIXES). Override with
        # OPENAI_MODEL, or per role with OPENAI_GRADER_MODEL / OPENAI_SYNTHESIS_MODEL.
        default = "gpt-4.1-mini" if role == "grader" else "gpt-4.1"
        spec = {
            "provider": "openai",
            "model": _role_model("openai", role, "OPENAI_MODEL", default),
        }
        base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
        if base_url:
            # Set this to point at Azure OpenAI, a self-hosted gateway, or any
            # OpenAI-compatible endpoint. Part of the cache key, so switching it rebuilds
            # the client instead of silently reusing one aimed at the old host.
            spec["base_url"] = base_url
        return spec
    if provider == "gemini":
        return {"provider": "gemini", "model": _role_model("gemini", role, "GEMINI_MODEL", "gemini-flash-latest")}
    if provider == "claude":
        return {"provider": "claude", "model": _role_model("anthropic", role, "ANTHROPIC_MODEL", "claude-sonnet-4-6")}
    if provider == "groq":
        return {"provider": "groq", "model": _role_model("groq", role, "GROQ_MODEL", "llama-3.3-70b-versatile")}
    if provider == "ollama":
        return {
            "provider": "ollama",
            "model": _role_model("ollama", role, "OLLAMA_MODEL", "llama3.2"),
            "base_url": _ollama_base_url(),
        }
    raise ValueError(
        f"Unknown provider '{provider}'. Expected one of: {', '.join(KNOWN_PROVIDERS)}."
    )


def _vision_llm_spec(provider: str) -> dict:
    if provider == "openai":
        spec = {
            "provider": "openai",
            "model": os.getenv("OPENAI_VISION_MODEL", "").strip()
            or os.getenv("OPENAI_MODEL", "").strip()
            or "gpt-4.1",
        }
        base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
        if base_url:
            spec["base_url"] = base_url
        return spec
    if provider == "gemini":
        return {"provider": "gemini", "model": os.getenv("GEMINI_VISION_MODEL", "gemini-flash-latest")}
    if provider == "claude":
        return {"provider": "claude", "model": os.getenv("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-6")}
    if provider == "ollama":
        return {
            "provider": "ollama",
            "model": os.getenv("OLLAMA_VISION_MODEL", "llava"),
            "base_url": _ollama_base_url(),
        }
    raise ValueError(
        f"Unknown vision provider '{provider}'. Expected one of: "
        f"{', '.join(VISION_CAPABLE_PROVIDERS)} (Groq has no vision model)."
    )


def _build_openai_llm(spec: dict):
    """Shared by the text and vision paths -- OpenAI's chat models are multimodal, so there is
    no separate vision client to build."""
    from langchain_openai import ChatOpenAI

    kwargs = {"model": spec["model"], "max_retries": 0}
    if _openai_supports_temperature(spec["model"]):
        kwargs["temperature"] = 0
    else:
        logger.info(
            f"Omitting temperature for OpenAI model '{spec['model']}' "
            f"(reasoning-family models reject it)."
        )
    if spec.get("base_url"):
        kwargs["base_url"] = spec["base_url"]
    return ChatOpenAI(**kwargs)


def _build_text_llm(spec: dict):
    provider = spec["provider"]
    # max_retries=0 everywhere is deliberate -- see point 4 of the module docstring. We want
    # to fall over to the next provider fast, not sit inside one client's backoff.
    if provider == "openai":
        return _build_openai_llm(spec)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=spec["model"], temperature=0, max_retries=0)
    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=spec["model"], temperature=0, max_retries=0)
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=spec["model"], temperature=0, max_retries=0)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=spec["model"], base_url=spec["base_url"], temperature=0, max_retries=0
        )
    raise ValueError(
        f"Unknown provider '{provider}'. Expected one of: {', '.join(KNOWN_PROVIDERS)}."
    )


def _build_vision_llm(spec: dict):
    provider = spec["provider"]
    if provider == "openai":
        return _build_openai_llm(spec)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=spec["model"], temperature=0, max_retries=0)
    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=spec["model"], temperature=0, max_retries=0)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=spec["model"], base_url=spec["base_url"], temperature=0, max_retries=0
        )
    raise ValueError(
        f"Unknown vision provider '{provider}'. Expected one of: "
        f"{', '.join(VISION_CAPABLE_PROVIDERS)} (Groq has no vision model)."
    )


def _spec_key(spec: dict) -> tuple:
    return tuple(sorted(spec.items()))


def _get_cached(cache: dict, spec: dict, builder, provider: str, label: str):
    """Double-checked, lock-guarded cache. The lock matters because FastAPI runs sync route
    handlers (and agent.py's ThreadPoolExecutor fans out summarisation batches) on a
    threadpool: without it two threads can race to construct the same client, and dict
    mutation from multiple threads is exactly the kind of bug that shows up once a month in
    production and never in dev."""
    key = _spec_key(spec)
    with _client_lock:
        client = cache.get(key)
        if client is not None:
            return client
    # Construct outside the lock: client construction can do file/env/network setup and we
    # don't want to serialise every other thread behind a slow provider SDK import.
    logger.info(f"Constructing {label} LLM client for provider '{provider}' (model={spec.get('model')})")
    client = builder(spec)
    with _client_lock:
        return cache.setdefault(key, client)


def _get_cached_text_llm(provider: str, role: str = "synthesis"):
    return _get_cached(
        _text_llm_cache, _text_llm_spec(provider, role), _build_text_llm, provider, "text"
    )


def _get_cached_vision_llm(provider: str):
    return _get_cached(_vision_llm_cache, _vision_llm_spec(provider), _build_vision_llm, provider, "vision")


def _evict_client(provider: str, vision: bool):
    """Drops a cached client so the next request builds a fresh one.

    Called after auth/connection failures specifically: those are the failures where the
    cached object may be holding on to state captured from a bad moment (an empty API key
    read at construction time, a dead HTTP connection pool). Without eviction a process that
    started before its env was fully populated stays broken until it is restarted, which is
    exactly the 'it works after a redeploy and nobody knows why' class of bug.

    Evicts *every* cached entry for the provider rather than one spec. Since models are now
    resolved per role, one provider can hold several clients (an OPENAI_GRADER_MODEL client
    and an OPENAI_SYNTHESIS_MODEL one). A bad API key invalidates all of them equally, and
    reconstructing a client is cheap next to leaving a stale one serving traffic."""
    cache = _vision_llm_cache if vision else _text_llm_cache
    with _client_lock:
        for key in [k for k in cache if ("provider", provider) in k]:
            cache.pop(key, None)


def reset_llm_clients():
    """Clears every cached client and the Ollama probe. Safe to call at any time; the next
    call rebuilds. Exposed for tests and for an operator-triggered 'reload config' path."""
    with _client_lock:
        _text_llm_cache.clear()
        _vision_llm_cache.clear()
    with _probe_lock:
        _ollama_probe.update(
            {"checked_at": 0.0, "ok": False, "detail": "not probed", "url": None, "models": None}
        )
    with _log_once_lock:
        _logged_once.clear()
    logger.info("LLM client caches cleared.")


# ---------------------------------------------------------------------------
# The chain runner: shared by the text roles and by vision so that chain validation, provider
# skipping, retry policy and error reporting can't drift apart between the two paths.
# ---------------------------------------------------------------------------

class LLMChainError(RuntimeError):
    """Raised when a role's whole provider chain is unusable or every provider failed.

    A distinct type so call sites can tell 'the config/infrastructure is wrong' apart from
    'the model said something we couldn't parse'. Subclasses RuntimeError so existing
    `except Exception` handlers keep working.

    `is_config_error` is True when nothing was ever attempted because the chain itself is
    broken (blank, all typos, no usable provider). That distinction is load-bearing: a
    degraded retry path can only help if there is *something* to retry against, so callers
    check the flag rather than pattern-matching on the message text."""

    def __init__(self, message: str, is_config_error: bool = False):
        super().__init__(message)
        self.is_config_error = is_config_error


def _unusable_chain_error(plan: dict) -> LLMChainError:
    """Builds the error we raise instead of `raise last_error` when nothing ever ran.

    Every branch here answers the same question -- 'which env var do I edit?' -- because the
    situations that get you here (blank chain, typo'd name, missing key, Ollama down) are all
    config mistakes, and a config mistake deserves a config-shaped error message."""
    role = plan["role"]
    env_var = _chain_env_var(role)
    lines = []

    if not plan["requested"]:
        lines.append(
            f"No providers configured for the '{role}' role: {env_var} resolved to an empty "
            f"chain (source: {plan['source']})."
        )
    else:
        lines.append(
            f"No usable providers for the '{role}' role. Configured chain "
            f"{plan['requested']} (source: {plan['source']})."
        )
    if plan["unknown"]:
        lines.append(
            f"Unrecognised provider name(s): {plan['unknown']} -- looks like a typo. "
            f"Valid names: {', '.join(KNOWN_PROVIDERS)}."
        )
    for provider, reason in plan["unusable"].items():
        lines.append(f"  - {provider}: {reason}")

    alternatives = [
        p for p in KNOWN_PROVIDERS
        if p not in plan["requested"] and _provider_usable(p, role)[0]
    ]
    if alternatives:
        lines.append(
            f"Currently usable alternative(s): {', '.join(alternatives)}. "
            f"Set {env_var}={','.join(alternatives)} in backend/.env to use them."
        )
    else:
        lines.append(
            f"No other provider is usable either. Set an API key (GOOGLE_API_KEY, "
            f"GROQ_API_KEY or ANTHROPIC_API_KEY) and put that provider in {env_var}."
        )
    return LLMChainError(" ".join(lines), is_config_error=True)


def _plan_chain(role: str) -> dict:
    """Resolves + validates a role's chain, raising an actionable error rather than letting
    an empty loop fall through to `raise last_error` (which would be `raise None`)."""
    plan = _resolve_chain(role)

    if plan["unknown"]:
        _log_once(
            f"unknown:{role}",
            f"[{role}] ignoring unrecognised provider name(s) {plan['unknown']} in "
            f"{_chain_env_var(role)} -- valid names are {', '.join(KNOWN_PROVIDERS)}. "
            f"Continuing with {plan['usable'] or 'nothing'}.",
        )
    for provider, reason in plan["unusable"].items():
        _log_once(
            f"unusable:{role}:{provider}",
            f"[{role}] skipping provider '{provider}': {reason}",
        )

    if not plan["usable"]:
        raise _unusable_chain_error(plan)
    return plan


def _run_chain(role: str, attempt, vision: bool = False):
    """Runs `attempt(provider)` against each usable provider in the role's chain, in order.

    Falls back on any error. The last provider in the chain -- and only the last, since the
    others have somewhere better to go than a sleep -- gets one bounded retry for errors that
    are clearly transient."""
    plan = _plan_chain(role)
    usable = plan["usable"]
    last_error = None
    last_provider = None

    for index, provider in enumerate(usable):
        is_last = index == len(usable) - 1
        attempts_left = 1 + (_TRANSIENT_RETRIES if is_last else 0)
        tries = 0
        while tries < attempts_left:
            tries += 1
            try:
                result = attempt(provider)
                if provider == "ollama":
                    _mark_ollama(True, f"reachable at {_ollama_base_url()}")
                return result
            except Exception as e:
                kind = _classify_error(e)
                hint = _failure_hint(provider, kind)

                if kind == "connection / provider unreachable":
                    _evict_client(provider, vision)
                    if provider == "ollama":
                        _mark_ollama(False, f"not reachable at {_ollama_base_url()} ({type(e).__name__})")
                elif kind == "auth / API key":
                    _evict_client(provider, vision)

                if tries < attempts_left and _is_transient(e):
                    logger.warning(
                        f"[{role}] provider '{provider}' failed ({kind}): {e} -- last provider "
                        f"in the chain, retrying once in {_TRANSIENT_RETRY_SLEEP}s."
                    )
                    time.sleep(_TRANSIENT_RETRY_SLEEP)
                    continue

                next_step = (
                    f"Trying next provider ({usable[index + 1]})..." if not is_last
                    else "No more providers in chain."
                )
                logger.warning(
                    f"[{role}] provider '{provider}' failed ({kind}): {e}. "
                    f"{hint + ' ' if hint else ''}{next_step}"
                )
                last_error = e
                last_provider = provider
                break

    detail = f"{_classify_error(last_error)}: {last_error}" if last_error else "unknown"
    raise LLMChainError(
        f"All providers failed for the '{role}' role. Chain tried: {usable} "
        f"(source: {plan['source']}). Last failure was '{last_provider}' -- {detail}. "
        f"{_failure_hint(last_provider, _classify_error(last_error)) if last_error else ''}"
    ) from last_error


# ---------------------------------------------------------------------------
# Response normalisation. Chat models return .content as either a plain string or a list of
# content blocks (Anthropic always does the latter when tools/thinking are involved, Gemini
# sometimes does). Every consumer in this app wants text, so normalise in one place.
# ---------------------------------------------------------------------------

def content_to_text(response) -> str:
    """Best-effort extraction of plain text from a chat-model response. Never raises."""
    try:
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") in (None, "text") and isinstance(block.get("text"), str):
                        parts.append(block["text"])
                    elif isinstance(block.get("text"), str):
                        parts.append(block["text"])
                else:
                    text = getattr(block, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(p for p in parts if p)
        return str(content)
    except Exception:
        return str(response)


# ---------------------------------------------------------------------------
# Structured output. This is the fiddliest part of multi-provider routing: .with_structured_
# output() is not one feature, it's three (function calling / json_mode / json_schema) with
# different support per provider AND per model. Groq's Llama models handle tool-call-based
# schemas well; Ollama's smaller local models frequently emit JSON-ish prose, ignore the
# schema, or return None. Letting any of that propagate would break grade_documents_node.
# ---------------------------------------------------------------------------

def _is_pydantic_model(schema) -> bool:
    try:
        from pydantic import BaseModel
        return isinstance(schema, type) and issubclass(schema, BaseModel)
    except Exception:
        return False


def _schema_fields(schema) -> dict:
    """{field_name: description} for a pydantic schema, used to build the plain-text prompt."""
    try:
        json_schema = schema.model_json_schema()
        return {
            name: (spec.get("description") or spec.get("title") or name)
            for name, spec in (json_schema.get("properties") or {}).items()
        }
    except Exception:
        return {}


def _extract_json_object(text: str):
    """Pulls the first balanced {...} out of a model response, tolerating markdown fences and
    the "Sure! Here's the JSON:" preamble small models love to emit."""
    import json

    if not text:
        return None
    cleaned = text.strip()
    if "```" in cleaned:
        chunks = cleaned.split("```")
        for chunk in chunks:
            candidate = chunk.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                cleaned = candidate
                break

    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start:i + 1])
                except Exception:
                    return None
    return None


def _coerce_to_schema(schema, text: str):
    """Last-resort parse of a plain-text response into the requested schema.

    Three escalating strategies, because "the model ignored the schema" has three flavours:
      1. It emitted JSON (possibly fenced, possibly with a preamble)  -> parse it.
      2. It emitted `field: value` lines                              -> scrape them.
      3. It emitted a bare answer ("yes")                             -> for a single-field
         schema, that bare answer IS the value. This is the case that actually matters here:
         GradeDocuments has exactly one field and small models answer it in one word."""
    import re

    if not _is_pydantic_model(schema):
        parsed = _extract_json_object(text)
        if parsed is not None:
            return parsed
        raise LLMChainError(
            f"Could not coerce plain-text response into the requested schema. "
            f"Response began: {text[:200]!r}"
        )

    parsed = _extract_json_object(text)
    if isinstance(parsed, dict):
        try:
            return schema.model_validate(parsed)
        except Exception:
            pass  # fall through to the scraping strategies

    fields = _schema_fields(schema)
    values = {}
    for name in fields:
        match = re.search(rf"[\"']?{re.escape(name)}[\"']?\s*[:=]\s*[\"']?([^\"'\n,}}]+)", text, re.I)
        if match:
            values[name] = match.group(1).strip()
    if values:
        try:
            return schema.model_validate(values)
        except Exception:
            pass

    if len(fields) == 1:
        name = next(iter(fields))
        bare = text.strip().strip("`\"' .")
        # Prefer an explicit yes/no token when the field is a binary score: models frequently
        # answer "Yes, the document is relevant." rather than the bare token.
        lowered = bare.lower()
        token = None
        if re.search(r"\byes\b", lowered):
            token = "yes"
        elif re.search(r"\bno\b", lowered):
            token = "no"
        for candidate in ([token] if token else []) + [bare, bare.split("\n")[0].strip()]:
            if not candidate:
                continue
            try:
                return schema.model_validate({name: candidate})
            except Exception:
                continue

    raise LLMChainError(
        f"Could not coerce plain-text response into {getattr(schema, '__name__', schema)}. "
        f"Response began: {text[:200]!r}"
    )


def _structured_output_prompt(schema, prompt):
    """Appends explicit JSON instructions to the original prompt for the degraded path."""
    fields = _schema_fields(schema)
    if fields:
        example = ", ".join(f'"{name}": "<{desc}>"' for name, desc in fields.items())
        instruction = (
            "\n\nRespond with ONLY a single JSON object and nothing else -- no prose, no "
            "markdown code fences, no explanation. Use exactly this shape:\n"
            "{" + example + "}"
        )
    else:
        instruction = "\n\nRespond with ONLY a single JSON object and nothing else."

    if isinstance(prompt, str):
        return prompt + instruction
    if isinstance(prompt, list):
        from langchain_core.messages import HumanMessage
        return list(prompt) + [HumanMessage(content=instruction.strip())]
    return prompt


class _StructuredFallbackLLM:
    """`.with_structured_output(schema).invoke(prompt)` with two layers of degradation.

    Layer 1 is the native structured-output API of each provider in the chain, in order.
    Layer 2 -- reached only when *every* provider's native support failed or returned
    something that doesn't validate -- re-asks in plain text with explicit JSON instructions
    and parses the answer ourselves. Layer 2 exists because grade_documents_node runs on every
    single query: a grader that raises takes the whole request down, whereas a grader that
    degrades to reading "yes" out of a sentence keeps the app answering."""

    def __init__(self, parent, schema):
        self._parent = parent
        self._schema = schema

    def _native(self, llm, prompt):
        result = llm.with_structured_output(self._schema).invoke(prompt)
        # Some providers/models return None or a stray dict instead of the schema type when
        # the model didn't emit a well-formed tool call. Treat that as a provider failure so
        # the chain moves on rather than handing agent.py a None to call .binary_score on.
        if result is None:
            raise LLMChainError("structured output returned None (model produced no parseable object)")
        if _is_pydantic_model(self._schema) and not isinstance(result, self._schema):
            if isinstance(result, dict):
                return self._schema.model_validate(result)
            raise LLMChainError(
                f"structured output returned {type(result).__name__}, expected "
                f"{getattr(self._schema, '__name__', self._schema)}"
            )
        return result

    def invoke(self, prompt):
        role = self._parent.role
        try:
            return _run_chain(role, lambda p: self._native(_get_cached_text_llm(p, role), prompt))
        except LLMChainError as chain_error:
            # If the chain itself is unusable (no key, nothing configured) the plain-text
            # retry cannot work either -- re-raise the actionable config error as-is rather
            # than burying it under a second, identical failure.
            if getattr(chain_error, "is_config_error", False):
                raise
            logger.warning(
                f"[{role}] native structured output unavailable across the whole chain "
                f"({chain_error}). Degrading to plain-text JSON prompt."
            )

        text_prompt = _structured_output_prompt(self._schema, prompt)
        response = _run_chain(role, lambda p: _get_cached_text_llm(p, role).invoke(text_prompt))
        return _coerce_to_schema(self._schema, content_to_text(response))


class _BoundFallbackLLM:
    def __init__(self, parent, tools):
        self._parent = parent
        self._tools = tools

    def invoke(self, messages):
        role = self._parent.role
        return _run_chain(
            role,
            lambda p: _get_cached_text_llm(p, role).bind_tools(self._tools).invoke(messages),
        )


class FallbackLLM:
    """Tries each provider in a role's chain in order, falling back to the next on any
    error. Exposes the same .invoke() / .with_structured_output() / .bind_tools() surface
    the underlying LangChain chat models do, so call sites don't need to change."""

    def __init__(self, role: str):
        self.role = _validate_role(role)

    def _provider_chain(self):
        return _get_role_provider_chain(self.role)

    def _call_with_fallback(self, fn):
        """Kept as a public-ish hook (it was part of this class's shape before the hardening
        pass); all the logic now lives in _run_chain so text and vision can't diverge."""
        return _run_chain(self.role, lambda provider: fn(_get_cached_text_llm(provider, self.role)))

    def invoke(self, *args, **kwargs):
        return self._call_with_fallback(lambda llm: llm.invoke(*args, **kwargs))

    def with_structured_output(self, schema):
        return _StructuredFallbackLLM(self, schema)

    def bind_tools(self, tools):
        return _BoundFallbackLLM(self, tools)


def get_llm_for_role(role: str) -> FallbackLLM:
    """
    Returns a FallbackLLM for the given role ("grader", "synthesis" or "vision"), which tries
    each provider in that role's chain in order and falls back automatically on error.
    Construction is lazy and network-free: nothing is validated or connected until a call is
    actually made, so importing this module never blocks server startup.
    """
    return FallbackLLM(role)


def get_llm() -> FallbackLLM:
    """Backward-compatible alias: returns the 'synthesis' role's FallbackLLM. Used by
    call sites (e.g. the retriever's unused llm parameter, /suggest_questions) that predate
    the grader/synthesis split and don't need the distinction."""
    return get_llm_for_role("synthesis")


# ---------------------------------------------------------------------------
# Vision: message-format differences between providers are handled here so vision_tool.py
# doesn't need to know which provider it's talking to.
# ---------------------------------------------------------------------------

def _build_vision_message(provider: str, prompt_text: str, b64_image: str):
    """Builds the provider-appropriate multimodal HumanMessage.

    Verified against the pinned integrations (langchain-anthropic 1.5.1,
    langchain-google-genai 4.3.1, langchain-ollama 1.1.0) by round-tripping a message through
    each package's own message converter:
      * Anthropic wants a native {"type": "image", "source": {...base64...}} block.
      * Gemini and Ollama both accept the OpenAI-style {"type": "image_url", ...} block, and
        both accept the data-URI either as a bare string or under {"url": ...}. The dict form
        is used here because it is the shape both libraries document; langchain_ollama splits
        it on the comma and forwards the base64 payload in its `images` array, which is what
        the Ollama /api/chat endpoint expects.
    If a future upgrade breaks one of these, the failure surfaces as a per-provider error in
    the chain log with the provider named, not as a silent wrong-format request."""
    from langchain_core.messages import HumanMessage

    if provider == "claude":
        image_block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64_image},
        }
    else:
        image_block = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
        }

    return HumanMessage(content=[{"type": "text", "text": prompt_text}, image_block])


def invoke_vision_with_fallback(png_bytes: bytes, prompt_text: str):
    """
    Sends a rendered page image + question to the vision provider chain, trying each in
    order and falling back automatically on error (rate limit, quota, auth, etc.).
    Returns (response_text, provider_name_that_succeeded).

    Raises LLMChainError (a RuntimeError) if the chain is unusable or every provider failed --
    vision_tool.py catches that and returns it to the agent as a readable string, because a
    LangGraph tool must never raise.
    """
    if not png_bytes:
        raise LLMChainError("No image bytes supplied to the vision provider chain.")

    b64_image = base64.b64encode(png_bytes).decode("utf-8")

    def attempt(provider):
        llm = _get_cached_vision_llm(provider)
        message = _build_vision_message(provider, prompt_text, b64_image)
        logger.info(
            f"[vision] sending {len(png_bytes) / 1024:.0f}KB page image to provider: {provider}"
        )
        response = llm.invoke([message])
        text = content_to_text(response)
        if not text.strip():
            # An empty body is a failure, not an answer -- fall through to the next provider
            # rather than handing the agent a blank "analysis".
            raise LLMChainError(f"provider '{provider}' returned an empty response")
        return text, provider

    return _run_chain("vision", attempt, vision=True)


# ---------------------------------------------------------------------------
# Operational visibility
# ---------------------------------------------------------------------------

def get_provider_status() -> dict:
    """
    Reports, per role, which providers are configured and which are actually usable right now.
    Surfaced by /health.

    Deliberately cheap and deliberately safe: it inspects env vars and (for Ollama) reuses the
    short-lived TCP probe cache. It never sends a request to an LLM API, never spends quota,
    and never raises -- a health endpoint that can 500 because the health check itself blew up
    is worse than no health endpoint.
    """
    try:
        providers = {}
        for provider in KNOWN_PROVIDERS:
            key_vars = PROVIDER_KEY_ENV.get(provider, ())
            configured = (
                any((os.getenv(v) or "").strip() for v in key_vars) if key_vars else True
            )
            usable, reason = _provider_usable(provider)
            entry = {
                "configured": bool(configured),
                "usable": bool(usable),
                "key_env": list(key_vars),
            }
            if reason:
                entry["reason"] = reason
            if provider == "ollama":
                entry["base_url"] = _ollama_base_url()
                entry["models_pulled"] = _ollama_models()
                entry["text_model"] = os.getenv("OLLAMA_MODEL", "llama3.2")
                entry["vision_model"] = os.getenv("OLLAMA_VISION_MODEL", "llava")
            providers[provider] = entry

        roles = {}
        all_ok = True
        for role in ROLES:
            try:
                plan = _resolve_chain(role)
            except Exception as e:
                roles[role] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                all_ok = False
                continue
            ok = bool(plan["usable"])
            all_ok = all_ok and ok
            roles[role] = {
                "ok": ok,
                "source": plan["source"],
                "configured": plan["requested"],
                "usable": plan["usable"],
                "active": plan["usable"][0] if plan["usable"] else None,
                "unusable": plan["unusable"],
                "unknown": plan["unknown"],
                "env_var": _chain_env_var(role),
            }
            if not ok:
                roles[role]["hint"] = str(_unusable_chain_error(plan))

        return {"ok": all_ok, "roles": roles, "providers": providers}
    except Exception as e:  # pragma: no cover -- last-resort guard, must never propagate
        logger.error(f"get_provider_status() failed unexpectedly: {e}", exc_info=True)
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "roles": {}, "providers": {}}
