You are working inside the LLM IntelliProxy codebase in VS Code.
Your task is to implement a context compression middleware module, a model fallback engine,
and a model availability monitoring system — and wire all three into the existing proxy pipeline.
Do NOT modify any unrelated files. Do NOT refactor existing logic unless directly required by this task.

━━━ TASK OVERVIEW ━━━

Implement three cooperating systems:

  1. ContextCompressionEngine — intercepts every incoming prompt before it reaches the
     decision engine or provider, compresses it according to runtime settings, then passes
     the result downstream transparently.

  2. ModelFallbackEngine — intercepts every provider response. If a provider returns an
     error, it automatically retries the same request against the next most similar model
     in the registry, continuing until a successful response is returned or all candidates
     are exhausted. This applies to ALL requests regardless of routing mode (auto-dispatch
     or direct passthrough).

  3. ModelAvailabilityMonitor — runs continuous background health checks against every
     registered model. Records precise timestamps when a model becomes unavailable and
     when it recovers. Uses historical downtime patterns to estimate and display predicted
     recovery times on the dashboard.

━━━ PHASE 0 — IMPLEMENTATION PLAN (MANDATORY GATE) ━━━

This phase is REQUIRED before any code is written, any file is created, or any existing
file is modified. Do not proceed to Phase 1 until this phase is complete and its output
has been written to disk.

─── 0.1 Codebase Analysis ───

Read and analyze the entire existing codebase. Produce a structured summary covering:

  EXISTING STRUCTURE:
    • Folder layout and purpose of each directory
    • Entry points (main file, server bootstrap, router)
    • Request pipeline — trace the full lifecycle of a single POST /v1/chat/completions
      request from receipt to provider response, naming every function and file involved
    • Provider system — how providers are currently defined, registered, and called
    • Model registry — current schema, how it is populated and queried
    • Decision engine — current routing logic and where it sits in the pipeline
    • Config system — how config.yaml and env vars are loaded and accessed
    • Database layer — ORM or query builder in use, migration strategy, existing tables
      and their full schemas
    • Background jobs — any existing schedulers, cron tasks, or event loops
    • Dashboard — frontend framework, how it receives data, existing views and endpoints
    • Test setup — test runner, conventions, existing test file locations
    • Error handling — how provider errors are currently caught and surfaced

  GAPS IDENTIFIED:
    For each of the three systems to be implemented (compression, fallback, availability),
    state explicitly:
      • What does not exist at all and must be created from scratch
      • What partially exists and must be extended
      • What already exists and must only be wired or configured

  CONFLICTS IDENTIFIED:
    • Any naming collisions between new modules and existing files or exports
    • Any existing error handling logic that the fallback engine must replace or wrap
    • Any existing model status fields in the registry that overlap with the new
      availability state schema
    • Any existing background job infrastructure that the monitor should reuse rather
      than duplicate

─── 0.2 Implementation Plan Document ───

Write the implementation plan to a file at:
  IMPLEMENTATION_PLAN.md

If this file already exists:
  • Read its current contents first
  • Append a new dated section: ## Plan Update — <ISO date>
  • Do NOT overwrite or remove any prior plan content
  • Note which prior items are now complete, in progress, or superseded

If this file does not exist:
  • Create it from scratch with the full structure below

The document must contain the following sections:

  # LLM IntelliProxy — Implementation Plan

  ## Codebase Snapshot
  Date analyzed: <ISO datetime>

  ### Folder Structure
  <annotated tree of the current project — one line per file/folder, with a short
  purpose annotation for each>

  ### Request Pipeline (current)
  <numbered trace of a request through the existing system, naming files and functions>

  ### Database Tables (current)
  <list of existing tables with their columns and types>

  ### Background Jobs (current)
  <list of any existing scheduled tasks, their interval, and their purpose>

  ## Gap Analysis

  ### ContextCompressionEngine
  | Item | Status | Notes |
  |------|--------|-------|
  <one row per required component — status: missing | partial | exists>

  ### ModelFallbackEngine
  | Item | Status | Notes |
  |------|--------|-------|

  ### ModelAvailabilityMonitor
  | Item | Status | Notes |
  |------|--------|-------|

  ## Conflict Register
  <numbered list of identified conflicts with proposed resolutions>

  ## Implementation Sequence
  <ordered list of every discrete implementation task across all three systems,
  grouped by step, with:
    - task description
    - file(s) to create or modify
    - estimated risk: low | medium | high
    - dependencies: which prior tasks must be complete first>

  ## Database Migration Plan
  <ordered list of all new tables and columns to be added, with migration filenames,
  confirming no existing table is altered destructively>

  ## Test Plan
  <list of all test cases to be added, grouped by system, with a one-line description
  of what each test verifies>

  ## Open Questions
  <any ambiguities discovered during analysis that require clarification before
  or during implementation — if none, write "None">

─── 0.3 Plan Review Gate ───

After writing IMPLEMENTATION_PLAN.md, output a concise summary to the chat containing:

  1. The full annotated folder structure (copied from the plan)
  2. The gap analysis tables for all three systems
  3. The conflict register
  4. The ordered implementation sequence (task list only, no file details)
  5. Any open questions

Then STOP and wait for explicit confirmation before proceeding to Phase 1.

If the user responds with approval (e.g. "looks good", "proceed", "continue"), begin
Phase 1 from the top of the implementation sequence.

If the user requests changes to the plan, update IMPLEMENTATION_PLAN.md accordingly
(append a "## Plan Revision" section with the changes), re-output the affected sections,
and wait for approval again.

Do NOT begin implementation under any circumstances until approval is received.

━━━ PHASE 1 — LOCATE INTEGRATION POINTS ━━━

Before writing any implementation code (config changes, new modules, pipeline wiring):
  1. Re-read the relevant sections of IMPLEMENTATION_PLAN.md to confirm the integration
     points identified in Phase 0 are still accurate
  2. If anything has changed or was missed, update the plan document before proceeding
  3. Confirm in the chat which files will be touched in the next step

─── Files to locate ───

  a. Where incoming prompts are received (POST /v1/chat/completions handler or equivalent)
  b. Where provider calls are made and responses are received
  c. How errors and HTTP status codes from providers are currently handled
  d. The structure of the unified model registry (fields available per model entry)
  e. Where background jobs or scheduled tasks are currently run (if any)

━━━ PHASE 2 — ADD CONFIGURATION ━━━

Extend config.yaml with the following new sections (do NOT remove existing keys):

  compression:
    enabled: true               # COMPRESSION_ENABLED
    level: medium               # low | medium | high
    max_tokens: 4096            # MAX_TOKENS
    mode: coding                # coding | chat | general

  fallback:
    enabled: true               # FALLBACK_ENABLED
    max_attempts: 3             # maximum models to try before returning error (includes original)
    retry_on:                   # HTTP status codes and error types that trigger fallback
      - 429                     # rate limited
      - 500                     # internal server error
      - 502                     # bad gateway
      - 503                     # service unavailable
      - 504                     # gateway timeout
      - timeout                 # request timeout
      - connection_error        # provider unreachable
    similarity_strategy: category_then_size
    excluded_models: []         # model IDs never used as fallback targets

  availability:
    enabled: true               # AVAILABILITY_ENABLED
    check_interval_seconds: 60  # how often to probe each model
    probe_timeout_seconds: 10   # max wait per probe before marking unresponsive
    recovery_window_hours: 168  # hours of history to use for ETA prediction (7 days)
    min_samples_for_eta: 3      # minimum past downtime events needed before showing ETA
    mark_unavailable_after: 2   # consecutive failures before marking unavailable
    mark_available_after: 1     # consecutive successes before marking recovered

Map each value to an environment variable override:
  COMPRESSION_ENABLED, COMPRESSION_LEVEL, COMPRESSION_MAX_TOKENS, COMPRESSION_MODE
  FALLBACK_ENABLED, FALLBACK_MAX_ATTEMPTS
  AVAILABILITY_ENABLED, AVAILABILITY_CHECK_INTERVAL, AVAILABILITY_PROBE_TIMEOUT

━━━ PHASE 3 — IMPLEMENT CONTEXT COMPRESSION ━━━

Create a new file at the appropriate path in the existing module structure, e.g.:
  src/middleware/contextCompression.ts   (or .js / .py — match the project language)

Implement and export a single middleware function:
  compressContext(messages: Message[], settings: CompressionSettings): Message[]

The function must implement this exact logic:

  IF settings.enabled = false
    → Return messages unchanged. Strip only: ANSI escape codes, exact duplicate messages.
    → Do NOT summarize, restructure, or remove any content.

  IF settings.enabled = true, apply by level:

    LOW    → Remove noise and duplicate messages.
             Preserve all structure and technical detail.

    MEDIUM → Summarize verbose content blocks.
             Retain all errors, identifiers, file references, and key values.
             Drop non-essential log lines.

    HIGH   → Retain ONLY: errors, failures, key results, critical code paths.
             Convert all other content to dense single-line summaries.
             Maximize token reduction.

  RELEVANCE RULES (apply at all compression levels):

    KEEP — never remove:
      • Errors, exceptions, failures, warnings
      • File names, line numbers, stack traces
      • Decision points, return values, state changes

    REMOVE — always:
      • Progress indicators and success confirmations
      • Repeated or duplicate output
      • Boilerplate, filler text, decorative separators

  MODE-SPECIFIC RULES:

    coding  → Errors first. git output: changed files only.
              Test output: failures only. Logs: errors + warnings only.
    chat    → Compress to: intent + constraints + key facts. Drop pleasantries.
    general → One sentence per concept. Remove all filler.

  TOKEN CONTROL:
    1. If compressed output exceeds settings.max_tokens, compress lower-priority
       sections further until it fits.
    2. Never truncate errors, stack traces, or failures regardless of token limit.
    3. If output still exceeds max_tokens after full compression, append:
       [TRUNCATED: <N> low-priority tokens removed]

  OUTPUT FORMAT — structure the compressed result as three labelled sections
  prepended to the final message array:
    [GOAL]             One-line description of the task or request.
    [KEY CONTEXT]      Critical prior state, decisions, or constraints. Omit if none.
    [COMPRESSED INPUT] Filtered and compressed content per rules above.

  HARD CONSTRAINTS — the function must never:
    ✗ Hallucinate or infer content that was not present in the input
    ✗ Include removed content in any form, including in comments or summaries
    ✗ Add output sections beyond the three listed above
    ✗ Explain, annotate, or justify its compression decisions in the output

━━━ PHASE 4 — IMPLEMENT MODEL FALLBACK ENGINE ━━━

Create a new file at the appropriate path, e.g.:
  src/middleware/modelFallback.ts   (or .js / .py — match the project language)

─── 4.1 Candidate Selection ───

Implement:
  rankFallbackCandidates(
    failedModelId: string,
    registry: ModelEntry[],
    settings: FallbackSettings
  ): ModelEntry[]

This function returns an ordered list of fallback candidates from the registry, excluding:
  • The model that just failed (never retry the same model)
  • Any model in settings.excluded_models
  • Any model with enabled: false
  • Any model whose current availability_status is "unavailable"

Rank candidates using settings.similarity_strategy:

  category_then_size (default):
    1. Same category as the failed model (e.g. both "coding")
    2. Within that group, order by closest context_window to the failed model
    3. Prefer same provider first, then other providers
    4. Fall back to different category models only if no same-category models remain

  provider_first:
    1. Same provider as the failed model
    2. Same category
    3. Any other enabled model

  any:
    1. Any enabled model in the registry, ordered by category match then provider match
    2. Use this as a last resort if other strategies return no candidates

─── 4.2 Fallback Execution ───

Implement:
  executeWithFallback(
    request: ChatRequest,
    initialModel: ModelEntry,
    registry: ModelEntry[],
    settings: FallbackSettings,
    providerCall: (model: ModelEntry, request: ChatRequest) => Promise<ChatResponse>
  ): Promise<FallbackResult>

Where FallbackResult is:
  {
    response:        ChatResponse
    model_used:      string
    provider_used:   string
    attempts:        FallbackAttempt[]
    fallback_used:   boolean
  }

And FallbackAttempt is:
  {
    model_id:        string
    provider:        string
    attempt_number:  number
    error_code:      string | null
    error_message:   string | null
    latency_ms:      number
    timestamp:       string             // ISO 8601 UTC
  }

Execution logic:

  1. Attempt the request against initialModel
  2. If the response is successful → return FallbackResult immediately
  3. If the response matches a code/type in settings.retry_on:
     a. Log the failed attempt to FallbackAttempt[]
     b. Report the failure to ModelAvailabilityMonitor via reportExternalFailure()
        so availability state is updated immediately without waiting for next probe
     c. Call rankFallbackCandidates() to get the next ordered candidate list
     d. Attempt the request against the next candidate
     e. Repeat until success OR attempts.length >= settings.max_attempts
  4. If all attempts fail:
     Return HTTP 502 with body:
       {
         "error": {
           "message": "All fallback models exhausted. Attempted: [model_a, model_b, model_c]",
           "type": "fallback_exhausted",
           "attempts": [ ...FallbackAttempt[] ]
         }
       }

  STREAMING BEHAVIOR:
    • If stream: true and a stream errors mid-flight:
      - Do NOT forward partial output to the client
      - Buffer internally, discard, and retry against the next candidate
      - Only begin streaming to the client once a model responds successfully

─── 4.3 Fallback Logging ───

Persist every FallbackAttempt to the database under a new table: fallback_log

  fallback_log
  ├── id               UUID / autoincrement
  ├── request_id       string    (correlate with decision backlog)
  ├── attempt_number   integer
  ├── model_id         string
  ├── provider         string
  ├── error_code       string | null
  ├── error_message    string | null
  ├── latency_ms       integer
  ├── fallback_used    boolean
  ├── timestamp        datetime  // ISO 8601 UTC — when this attempt was made

Add as an additive migration — do NOT alter existing tables.

━━━ PHASE 5 — IMPLEMENT MODEL AVAILABILITY MONITOR ━━━

Create a new file at the appropriate path, e.g.:
  src/monitor/modelAvailability.ts   (or .js / .py — match the project language)

─── 5.1 Database Schema ───

Add two new tables via additive migration. Do NOT alter existing tables.

  model_availability_log
  ├── id                    UUID / autoincrement
  ├── model_id              string       FK → registry model id
  ├── provider              string
  ├── status                string       "available" | "unavailable"
  ├── checked_at            datetime     ISO 8601 UTC — exact moment the probe ran
  ├── response_time_ms      integer | null
  ├── error_code            string | null
  ├── error_message         string | null
  ├── consecutive_failures  integer
  ├── consecutive_successes integer

  model_availability_state
  ├── model_id              string       PK — one row per model, upserted on each probe
  ├── provider              string
  ├── current_status        string       "available" | "unavailable" | "unknown"
  ├── unavailable_since     datetime | null   ISO 8601 UTC — set on transition to unavailable,
                                              cleared on recovery
  ├── last_available_at     datetime | null   ISO 8601 UTC — last confirmed successful probe
  ├── last_checked_at       datetime          ISO 8601 UTC — last probe of any kind
  ├── last_error_code       string | null
  ├── last_error_message    string | null
  ├── consecutive_failures  integer
  ├── consecutive_successes integer
  ├── estimated_recovery_at datetime | null   ISO 8601 UTC — computed ETA, null if unknown
  ├── eta_confidence        string | null     "low" | "medium" | "high" | null

─── 5.2 Health Probe Logic ───

Implement a background scheduler running on check_interval_seconds.
On each tick, probe every enabled model concurrently:

  probeModel(model: ModelEntry): Promise<ProbeResult>

  • Chat models: send single-message prompt "ping" with max_tokens: 1
  • Embedding models: send a single short string
  • Image models: skip probe — mark as "unknown" (cannot cheaply probe)

Apply hysteresis after each probe:
  • Mark UNAVAILABLE only after mark_unavailable_after consecutive failures
  • Mark AVAILABLE only after mark_available_after consecutive successes

On transition available → unavailable:
  • Set unavailable_since = probe timestamp (ISO 8601 UTC)
  • Set last_available_at = previous last_checked_at
  • Trigger ETA recalculation
  • Emit internal event for real-time dashboard update

On transition unavailable → available:
  • Set last_available_at = probe timestamp (ISO 8601 UTC)
  • Compute actual downtime = NOW() - unavailable_since
  • Persist completed downtime event to model_availability_log
  • Clear unavailable_since, estimated_recovery_at, eta_confidence (set all to null)
  • Emit internal recovery event

─── 5.3 ETA Prediction Engine ───

Implement:
  computeRecoveryETA(
    modelId: string,
    unavailableSince: Date,
    history: AvailabilityEvent[],
    settings: AvailabilitySettings
  ): ETAResult | null

Where ETAResult is:
  {
    estimated_recovery_at: Date         // ISO 8601 UTC
    confidence:            "low" | "medium" | "high"
    based_on_samples:      number
    avg_downtime_minutes:  number
  }

Algorithm:
  1. Query model_availability_log for all completed downtime events within
     recovery_window_hours
  2. If fewer than min_samples_for_eta events → return null
  3. Compute average downtime duration across all sampled events
  4. estimated_recovery_at = unavailable_since + avg_downtime_minutes
  5. Assign confidence from historical variance:
       high   → stddev < 20% of mean
       medium → stddev 20–50% of mean
       low    → stddev > 50% of mean
  6. Persist ETAResult to model_availability_state
  7. Recompute on every subsequent probe while model remains unavailable

─── 5.4 External Event Interface ───

Expose for use by the ModelFallbackEngine:

  reportExternalFailure(
    modelId: string,
    errorCode: string,
    errorMessage: string,
    timestamp: Date             // ISO 8601 UTC
  ): void

Must:
  • Increment consecutive_failures in model_availability_state
  • Insert a row into model_availability_log with status "unavailable"
    and checked_at = timestamp parameter
  • If consecutive_failures >= mark_unavailable_after and current_status
    is not already "unavailable":
    - Set unavailable_since = timestamp (ISO 8601 UTC)
    - Set current_status = "unavailable"
    - Trigger ETA recalculation immediately

─── 5.5 Startup Behavior ───

  1. Load existing model_availability_state from DB for all known models
  2. Set any models with no prior state to current_status: "unknown"
  3. Run an immediate full probe of all models before (or in parallel with) accepting traffic
  4. Log all probe results with checked_at = startup timestamp (ISO 8601 UTC)

━━━ PHASE 6 — WIRE INTO THE PIPELINE ━━━

In the existing POST /v1/chat/completions handler (or equivalent entry point):

  1. Load compression and fallback settings from config at request time (not startup)
  2. Call compressContext(req.messages, settings) — replace req.messages with result
  3. Resolve the target model (decision engine for intelliproxy-auto, or direct from request)
  4. If resolved model's current_status is "unavailable":
     → Skip attempt — go directly to rankFallbackCandidates() without a provider call
  5. Call executeWithFallback(request, resolvedModel, registry, fallbackSettings, providerCall)
  6. Return the successful response to the client

Add response headers on every request:
  X-IntelliProxy-Compression:      <level> | disabled
  X-IntelliProxy-Model:            <model_id that responded>
  X-IntelliProxy-Provider:         <provider name>
  X-IntelliProxy-Fallback-Used:    true | false
  X-IntelliProxy-Attempts:         <N>
  X-IntelliProxy-Model-Status:     available | unavailable | unknown

━━━ PHASE 7 — DASHBOARD EXTENSIONS ━━━

Extend the existing web dashboard. Do not rebuild existing views.

─── Model Registry View (extend) ───

Add columns:
  • Status             — color-coded badge: green / red / gray, real-time via WS/SSE
  • Unavailable Since  — datetime if currently down, empty otherwise
  • Est. Recovery      — human-readable ETA string (see formats below)
  • Uptime (7d)        — percentage of last 7 days the model was available
  • Fallback Eligible  — boolean

ETA display formats:
  high confidence:     "~2h 15m  (high confidence)"
  medium confidence:   "~3h 30m  (est.)"
  low confidence:      "~5h      (low confidence)"
  not yet computable:  "Unavailable since <time> — estimating..."
  just recovered:      "Recovered <N> minutes ago"

─── Decision Log (extend) ───

Add columns: fallback_used (boolean badge), attempts (integer).
Clicking a row where fallback_used=true expands an inline detail panel showing
all FallbackAttempt entries for that request including timestamps, error codes,
and latency per attempt.

─── New View — Availability Monitor ───

  Live Status Board:
    • One card per registered model showing: name, provider, status badge,
      unavailable_since, ETA with confidence, last_checked_at, avg response time (24h)
    • Unavailable model cards sort to the top automatically
    • All cards update in real time via WebSocket or SSE

  Downtime Timeline (chart):
    • Gantt-style — one row per model, colored segments: available / unavailable / unknown
    • Hover shows: start time, end time, duration, error code

  Downtime History Table:
    • Columns: model, provider, unavailable_since, recovered_at, duration,
      error_code, avg_downtime_for_model
    • Filterable by model, provider, date range
    • Exportable as CSV / JSON

─── New View — Fallback Analytics ───

  • Table: most frequently failing models (last 7 days, ranked by error count)
  • Table: most frequently used fallback targets (ranked by selection count)
  • Metric cards: total fallback events today, fallback success rate, avg attempts per event
  • Chart: fallback events over time (line, last 7 days)

━━━ PHASE 8 — TESTS ━━━

Add unit tests in the existing test directory. Cover:

  Compression:
    • enabled=false returns input unchanged minus noise
    • each compression level reduces token count appropriately
    • errors and stack traces are never removed at any level
    • output always contains all three required sections when enabled
    • max_tokens truncation appends the [TRUNCATED] marker

  Fallback:
    • successful first attempt → fallback_used=false, attempts=1
    • 429 / 500 / connection_error each trigger retry
    • max_attempts respected — no further retries beyond the limit
    • exhausted fallback returns HTTP 502 with full attempt log
    • failed model is never retried in the same request
    • excluded_models and disabled models are never selected
    • models with current_status=unavailable are skipped by rankFallbackCandidates
    • category_then_size ranks same-category before cross-category
    • mid-stream failure triggers retry without forwarding partial output
    • all attempts persisted to fallback_log with correct ISO 8601 UTC timestamps

  Availability Monitor:
    • mark_unavailable_after threshold respected — single failure does not mark unavailable
    • mark_available_after threshold respected — single success does not mark recovered
    • unavailable_since set to exact probe timestamp (ISO 8601 UTC) on first qualifying failure
    • last_available_at set to exact probe timestamp on recovery
    • unavailable_since cleared (null) on recovery
    • computeRecoveryETA returns null when fewer than min_samples_for_eta events exist
    • ETA confidence assigned correctly based on variance
    • estimated_recovery_at decreases correctly as time elapses
    • reportExternalFailure increments consecutive_failures and transitions state correctly
    • all timestamps stored and returned as ISO 8601 UTC strings
    • image models skipped by probe, assigned status "unknown"

━━━ PHASE 9 — SESSION AFFINITY & CONTEXT-AWARE MODEL SWITCHING ━━━

─── 9.1 The Problem ───

The decision engine selects a model per request. In a stateless single-turn use case
this is correct behavior. However, agentic coding tools (Cline, Continue, Cursor) and
multi-turn chat clients maintain session state — each message builds on prior context.
If the proxy routes message 1 to model A and message 7 to model B, model B has no
awareness of the prior conversation, causing:

  • Lost file context (model B does not know which files were opened or edited)
  • Repeated mistakes the session had already corrected
  • Contradictory suggestions that conflict with prior decisions
  • Broken tool-use chains where model B cannot follow up on model A's actions

The proxy must detect session boundaries and enforce model affinity within them.

─── 9.2 Session Detection ───

A session is defined as a sequence of requests sharing the same logical conversation.
Detect sessions using the following signals, in priority order:

  1. Explicit session ID header (highest priority):
       X-IntelliProxy-Session-ID: <uuid>
     If present, use this value as the session key. The client is responsible for
     generating and maintaining this ID across requests.

  2. OpenAI conversation threading:
     If the request contains a messages array with more than one message and the first
     message role is "system" followed by alternating "user"/"assistant" turns, treat
     this as a continuing conversation. Derive a session key from a hash of:
       SHA-256(first_system_message_content + first_user_message_content)
     This is stable across requests as long as the client sends full history.

  3. Stateless (no session):
     Single-message requests with no prior turns and no session header.
     Apply normal per-request routing with no affinity.

─── 9.3 Session Registry ───

Maintain an in-memory session registry (with optional DB persistence for
cross-restart continuity). Each session entry contains:

  {
    session_id:          string          // derived or client-supplied key
    model_id:            string          // model locked for this session
    provider:            string
    created_at:          string          // ISO 8601 UTC
    last_active_at:      string          // ISO 8601 UTC — updated on every request
    request_count:       number
    mode:                string          // coding | chat | general — detected at session start
    locked:              boolean         // true = affinity enforced, false = re-routing allowed
    context_turn_count:  number          // number of turns exchanged so far
    session_source:      string          // "header" | "conversation_hash" | "stateless"
  }

Session expiry:
  • Sessions expire after a configurable idle timeout (default: 30 minutes)
  • On expiry, remove from registry — the next request starts a fresh session
  • Configurable in config.yaml:

    session:
      enabled: true
      idle_timeout_minutes: 30
      persist_to_db: false          # true = sessions survive proxy restart
      coding_idle_timeout_minutes: 60   # longer timeout for coding sessions
      allow_client_override: true   # honor X-IntelliProxy-Session-ID if present

─── 9.4 Session Mode Detection ───

At session creation, classify the session mode from the first request:

  coding  → detected if any of:
              • User-Agent header contains: cline, cursor, continue, copilot, vscode
              • System prompt contains code-related keywords:
                "you are a coding assistant", "repository", "codebase", "file system"
              • First message contains a file path, code block, or diff
              • Request model was explicitly a coding-category model

  chat    → detected if:
              • Short conversational message with no code markers
              • System prompt is generic assistant framing

  general → default if no other mode is detected

Mode affects session behavior:
  • coding  → longest affinity lock, context forwarding on forced switch (see 9.5)
  • chat    → medium affinity, lighter context forwarding
  • general → shortest affinity, standard fallback behavior

─── 9.5 Affinity Enforcement ───

On every request where a session is active:

  1. Look up the session in the registry by session key
  2. If session found and session.locked = true:
     a. Bypass the decision engine entirely
     b. Check the locked model's current availability_status
     c. If available → forward directly to the locked model, skip to step 5
     d. If unavailable → enter forced switch flow (see 9.6)
  3. If session not found (new session):
     a. Run the decision engine to select the best model
     b. Create a new session entry with the selected model, locked = true
     c. Forward to the selected model
  4. If session found and session.locked = false:
     a. Run the decision engine normally
     b. Update session.last_active_at
  5. Update session.last_active_at and session.request_count on every request

─── 9.6 Forced Model Switch (Session Continuity Protocol) ───

A forced switch occurs when the session-locked model becomes unavailable mid-session.
This must be handled differently from a standard fallback because context continuity
matters.

Trigger conditions:
  • The locked model's availability_status is "unavailable" at request time
  • The locked model returns an error during an active session request

Forced switch procedure:

  STEP 1 — Select replacement model
    Call rankFallbackCandidates() with the locked model as the failed model.
    Prefer candidates in the same category (e.g. coding → coding).
    If no same-category candidate is available, prefer any available model.

  STEP 2 — Assess context forwarding feasibility
    Count the number of turns in the existing conversation (context_turn_count).
    If context_turn_count >= 1 (any prior context exists):
      → Context forwarding is required (see Step 3)
    If context_turn_count = 0:
      → No prior context — proceed as a clean session start on the new model

  STEP 3 — Forward context to the new model
    Prepend a context handoff block to the system prompt of the new model's
    first request. This block summarizes what the session established so far:

      [SESSION CONTEXT HANDOFF]
      You are continuing a session that was previously handled by a different model.
      The prior model became unavailable. Below is a summary of the established context:

      Session mode: {{mode}}
      Turns completed: {{context_turn_count}}
      Session started: {{created_at}}

      Conversation history:
      {{full_messages_array_from_prior_turns}}

      Continue from this point as if you were present for the entire session.
      Do NOT acknowledge the model switch or mention the handoff to the user.
      [END HANDOFF]

    Pass the full prior messages array (not a summary) if it fits within the
    new model's context window. If it exceeds the context window:
      → Apply contextCompression (HIGH level) to the prior history
      → Prepend the compressed summary instead
      → Append: [NOTE: prior context was compressed due to context window constraints]

  STEP 4 — Notify the client (optional, configurable)
    If session.notify_on_switch = true (config default: true for coding, false for chat):
      Inject a structured notification into the response before the model's reply:

        [IntelliProxy: Model switched mid-session]
        Prior model:       {{old_model_id}} ({{provider}}) — unavailable since {{unavailable_since}}
        Replacement model: {{new_model_id}} ({{provider}})
        Context forwarded: yes | no | compressed
        Reason:            {{error_code}} — {{error_message}}

      This notification appears as a visible prefix in the response body so the user
      knows a switch occurred and can decide whether to restart the session cleanly.

  STEP 5 — Update session registry
    Update the session entry:
      model_id    → new model ID
      provider    → new provider
      locked      → true (re-lock to the new model)
    Do NOT reset context_turn_count — it reflects the full session history.

  STEP 6 — Add response headers
    X-IntelliProxy-Session-ID:       <session_id>
    X-IntelliProxy-Session-Switched: true
    X-IntelliProxy-Prior-Model:      <old_model_id>
    X-IntelliProxy-Model:            <new_model_id>

─── 9.7 Session Persistence (optional) ───

If session.persist_to_db = true, add a new table via additive migration:

  session_registry
  ├── session_id            string        PK
  ├── model_id              string
  ├── provider              string
  ├── mode                  string
  ├── created_at            datetime      ISO 8601 UTC
  ├── last_active_at        datetime      ISO 8601 UTC
  ├── expires_at            datetime      ISO 8601 UTC
  ├── request_count         integer
  ├── context_turn_count    integer
  ├── locked                boolean
  ├── session_source        string
  ├── switch_count          integer       number of forced model switches in this session
  ├── notify_on_switch      boolean

─── 9.8 Configuration ───

Add to config.yaml (do NOT remove existing keys):

  session:
    enabled: true
    idle_timeout_minutes: 30
    coding_idle_timeout_minutes: 60
    persist_to_db: false
    allow_client_override: true         # honor X-IntelliProxy-Session-ID header
    notify_on_switch: true              # inject switch notification into response
    notify_modes:                       # only notify for these session modes
      - coding
    context_forwarding_enabled: true    # forward prior turns to replacement model
    context_forwarding_compression: auto
                                        # auto | always | never
                                        # auto = compress only if history exceeds
                                        #        new model's context window

─── 9.9 Dashboard Extensions ───

Extend the existing dashboard (do not rebuild existing views):

  Model Registry View:
    → Add column: "Active Sessions" — count of currently live sessions locked to this model

  New View — Session Monitor:

    Active Sessions Table:
      • Columns: session_id (truncated), mode, locked model, provider, turns,
        started_at, last_active_at, idle_for, switch_count
      • Clicking a row expands: full session timeline showing each request,
        model used, latency, and any switch events
      • Sessions approaching idle timeout shown with a warning indicator

    Session Health Metrics (cards):
      • Active sessions right now
      • Sessions switched mid-session today
      • Context forwarding success rate (forwarded vs compressed vs skipped)
      • Average session length (turns)
      • Average session duration (minutes)

    Session Switch Log:
      • Paginated table of all forced mid-session switches
      • Columns: session_id, mode, prior_model, replacement_model, reason,
        context_forwarded, switched_at
      • Filterable by mode, model, date range
      • Exportable as CSV / JSON

─── 9.10 Tests ───

Add to the existing test file:

  Session Affinity:
    • New session is created and model is locked after first request
    • Second request in same session bypasses decision engine and uses locked model
    • Session expires after idle_timeout_minutes of inactivity
    • Coding sessions use coding_idle_timeout_minutes, not default timeout
    • X-IntelliProxy-Session-ID header is respected when allow_client_override = true
    • Conversation hash produces a stable session key across requests with same history

  Forced Switch:
    • Unavailable locked model triggers forced switch, not a clean fallback
    • Replacement model receives full prior conversation history in handoff block
    • Replacement model receives compressed history when prior context exceeds context window
    • Session registry is updated to lock the new model after switch
    • context_turn_count is preserved across a switch (not reset)
    • X-IntelliProxy-Session-Switched header is true after a forced switch
    • Notification block is injected into response when notify_on_switch = true
    • Notification block is NOT injected when notify_on_switch = false

  Session Persistence:
    • Sessions are recoverable from DB after proxy restart when persist_to_db = true
    • Sessions are not persisted when persist_to_db = false
    • Expired sessions are purged from DB on next access or on a cleanup cycle

━━━ CONSTRAINTS FOR CLINE ━━━

  • Complete Phase 0 fully and receive explicit approval before writing any code
  • Update IMPLEMENTATION_PLAN.md at the start of each phase to mark completed tasks
  • Match the language, style, and patterns already used in this codebase exactly
  • Do NOT introduce new dependencies unless essential — prefer stdlib and existing packages
  • Do NOT modify any file outside:
      IMPLEMENTATION_PLAN.md
      the new compression module
      the new fallback module
      the new availability monitor module
      the config schema
      the request handler (wiring only)
      the dashboard (extensions only)
      the test file
      the DB migration file (additive only)
  • All timestamps must be stored and returned as ISO 8601 UTC strings — no exceptions
  • Implement phases in strict order — do not begin a phase before the previous is complete
  • If anything is ambiguous, add it to the Open Questions section of
    IMPLEMENTATION_PLAN.md and surface it in the Phase 0 review rather than
    making silent assumptions