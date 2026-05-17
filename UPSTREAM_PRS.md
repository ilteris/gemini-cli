# Upstream PR drafts

Four commits in this fork that would benefit from upstreaming to
`google-gemini/gemini-cli`. Each PR is self-contained; ship them in any order.

The fifth commit on this branch
(`5bf874020 fix: restore corrupted useRunEventNotifications.js dist file`) is
**not upstream-worthy** — it restores a dist file that was locally corrupted by
an out-of-band edit; upstream's dist file is fine.

---

## PR 1: `feat(core): honor GEMINI_SESSION_ID env var in createSessionId`

**Branch suggestion:** `feat/respect-gemini-session-id-env` **Files:**
`packages/core/src/utils/session.ts` (+7) **Local commit:** `1c65ea0f6`

### Summary

`createSessionId()` currently always returns `randomUUID()`. This PR reads
`GEMINI_SESSION_ID` from the environment first, validates it as a canonical
UUID, and returns it; falls back to `randomUUID()` on missing or malformed
values.

### Why

External orchestrators (terminal wrappers, IDE integrations, agent frameworks)
need to correlate gemini-cli sessions with their own session identities. Today,
a wrapper that launches `gemini` with `GEMINI_SESSION_ID=<known-uuid>` has no
way to pin gemini-cli's internal sessionId to that UUID — gemini mints an
independent one, leaving the wrapper unable to cross-reference its records
(hooks, transcripts, logs) with gemini-cli's own session storage.

Reading the env var with strict validation is the minimal-surface fix: opt-in
(does nothing if unset), validated (garbage is rejected), and
backwards-compatible (random fallback preserved).

### Behavior

```ts
const env = process.env['GEMINI_SESSION_ID'];
if (
  env &&
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(env)
) {
  return env.toLowerCase();
}
return randomUUID();
```

- Valid canonical UUID → adopted (lowercased)
- Missing or non-UUID → falls back to `randomUUID()`
- No breaking changes for anyone not setting the env var

### Real-world motivation

Soul OS (https://github.com/anthropics/...) uses this in its terminal launcher
to pin gemini-cli's UUID to the kernel-owned session id so hook events written
from gemini-cli line up with the kernel's records without a separate correlation
map.

---

## PR 2: `feat(hooks): include additionalContext from BeforeTool hook responses`

**Branch suggestion:** `feat/before-tool-additional-context` **Files:**
`packages/core/src/core/coreToolHookTriggers.ts` (+18) **Local commit:**
`b05e73a5e`

### Summary

`BeforeTool` is the only hook type that discards
`hookSpecificOutput. additionalContext` from its response. `AfterTool`,
`BeforeAgent`, and `AfterAgent` all inject it into LLM context. This PR aligns
`BeforeTool` with the other three.

### Why

Hooks that run _before_ a tool executes need to communicate context to the model
— most importantly when they intercept or modify execution (safety gates,
workspace scoping, permission checks). Today, a `BeforeTool` hook can deny or
modify a tool call but has no way to explain _why_ to the LLM in the next turn.
The model sees a blank denial and lacks the context to adapt.

`AfterTool` and the agent-lifecycle hooks already wrap their `additionalContext`
in `<hook_context>` tags appended to `llmContent`; this PR uses the identical
pattern for `BeforeTool`.

### Behavior

After tool execution, if the `BeforeTool` hook returned `additionalContext`,
append it to `toolResult.llmContent`:

```ts
if (beforeAdditionalContext) {
  toolResult.llmContent += `\n\n<hook_context>${beforeAdditionalContext}</hook_context>`;
}
```

No type changes — `BeforeToolHookOutput` already inherits
`getAdditionalContext()` from `DefaultHookOutput`.

### Real-world motivation

Soul OS middleware returns crash warnings in `additionalContext` when a
`BeforeTool` safety check fails (workspace scoping, audit gate, persona gate).
Without this PR the LLM had no visibility that its safety layer had fallen over
and would retry the same call indefinitely.

---

## PR 3: `feat(core): clip shell tool stdout before shipping to model`

**Branch suggestion:** `feat/shell-stdout-llm-cap` **Files:**
`packages/core/src/tools/shell.ts` (+25) **Local commit:** `7f4f4f03c`

### Summary

The shell tool passes `result.output` verbatim into `llmContent` (the payload
returned to the model on the next turn). A single recursive grep, find, or git
log from a deep tree can dump >1M tokens of stdout, exceeding the Gemini API
context window and rejecting the next request with a 400. The user loses the
turn even though the agent's intent was correct — the tool just had no output
budget.

This PR caps shell stdout shipped to the model at 100KB (~25K tokens) via
head+tail clip with a `[TRUNCATED N bytes — re-run narrower if needed]` marker.

### Behavior

- User-facing `returnDisplay` is unchanged — full output still renders in the
  chat panel
- Only the LLM-facing copy is clipped
- Head/tail strategy preserves the most-useful diagnostic signal (errors are
  usually at the bottom; command echo at the top)
- Clip marker tells the model what was dropped so it can re-issue a scoped
  command if it needs more

### Why this shape

Other tools already have per-tool caps:

- `web-fetch` truncates HTML response bodies
- `read-file` / `read-many-files` cap output length
- `grep-utils` clips long match counts

The bash tool was the outlier with no cap. This brings it in line.

---

## PR 4: `fix(core): three layered guards in ProjectRegistry against duplicate -N slug creation`

**Branch suggestion:** `fix/project-registry-marker-precedence` **Files:**
`packages/core/src/config/projectRegistry.ts`,
`packages/core/src/config/projectRegistry.test.ts` **Local commits:**
`c4ffe9340` (SOUL-017) + `f54c566b3` (SOUL-024)

### Summary

Two coordinated fixes for projectRegistry inconsistencies that surface as
duplicate `-N` slug directories for the same project path:

1. `normalizePath` now resolves symlinks via `fs.realpathSync` before
   comparison. Without this, the same project accessed via
   `/Users/x/dotfiles/foo` and a `/Users/x/foo` symlink pointing to it was
   treated as two distinct projects and got separate slugs.

2. `claimNewSlug` now pre-checks `findExistingSlugForPath` before minting a new
   slug. If a marker file on disk already claims this `projectPath` (e.g.
   `~/.gemini/tmp/foo/.project_root` resolves to `/Users/x/dotfiles/foo`), adopt
   that slug instead of creating `foo-1`, `foo-2`, etc.

### Why

The current `claimNewSlug` walks `<slug>`, `<slug>-1`, `<slug>-2`, ... and on
disk-collision (marker exists, owner is _different_) skips to the next
candidate. But it doesn't check the _match_ case — if the marker exists AND its
owner matches the requested project, the new-mint path still creates a fresh
`-N` slug. Net effect: same project gets multiple slug dirs whenever the
in-memory registry is desynced from disk.

The symlink-resolved normalization removes the second common driver: two
different access paths for one canonical project ending up as two slugs.

### Tests

`projectRegistry.test.ts` adds coverage for all three paths:

```ts
it('resolves symlinks in normalizePath', async () => {
  // real-project gets slug X; symlink-project resolving to real-project
  // gets the same slug X (not a new one)
});

it('claimNewSlug adopts existing marker for matching path', async () => {
  // pre-existing ~/.gemini/tmp/foo/.project_root → /Users/x/foo
  // claimNewSlug for /Users/x/foo returns "foo", not "foo-1"
});

it('prefers canonical bare slug over -N siblings when both have valid markers', async () => {
  // projects.json → foo-1; both foo and foo-1 have markers for /Users/x/foo
  // getShortId returns "foo" and rewrites mapping
});
```

### Third guard: registry-lookup hierarchy (`ensureProjectIdentifier`)

The first two guards close the _new-mint_ path. There's a residual case where
`ensureProjectIdentifier`'s registry-lookup runs before `claimNewSlug` ever
fires: once `projects.json` contains a stale mapping to `foo-1` whose marker
validly points at the project, `verifySlugOwnership("foo-1")` succeeds and the
function returns `"foo-1"` indefinitely. Guards 1 and 2 never get a chance to
correct it.

After `verifySlugOwnership(mappedSlug)` succeeds, this PR also checks whether
the canonical bare slug (`slugify(basename(path))`) is different AND has a valid
marker. If so, the mapping is rewritten to the canonical slug, markers are
ensured, the file is saved, and the canonical slug is returned directly.
(Returning directly rather than falling through to `findExistingSlugForPath`
matters — the latter's `readdir` order is filesystem-dependent and could
non-deterministically re-elect the `-N` slug.)

### Real-world repro

This fork's downstream consumer (Soul OS) hit this with a
`/Users/x/dotfiles/soul` project repeatedly migrating to `soul-1` across
launches. Manual cleanup (repoint `projects.json`, trash the `-N` dir) worked
but recurred every few weeks. Each of the three guards is necessary; with only
the first two, the lookup-hierarchy path kept resurrecting the bad mapping from
`projects.json` even after the disk side was cleaned.
