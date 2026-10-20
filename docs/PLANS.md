# Project plans

Record substantive decisions and their implementation steps here before execution. Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Latest entries get recoreded first, stale entries come later.

## Entry format

```text
## DD-MM-YY HH:MM:SS IST — Title
- Status: proposed / adopted / superseded
- Decision:
- Rationale and trade-offs:
- Affected components:
- Implementation steps:
- Completion criteria:
```

## 20-10-26 13:18:20 IST — Establish documentation scaffolding

- **Status:** Adopted.
- **Decision:** Keep the README to two lines, explain the architecture's components, and maintain separate state and plan documents. Defer detailed module contracts.
- **Rationale and trade-offs:** Provide a small, usable documentation structure for preprocessing work without prematurely specifying implementation interfaces.
- **Affected components:** `README.md`, `docs/ARCHITECTURE.md`, `docs/STATE.md`, `docs/PLANS.md`, and the README source notes in `docs/GOALS.md`.
- **Implementation steps:** Add the summary and documentation links; explain the existing diagram; initialise the state document with the requested sections and IST timestamps; keep affected references consistent.
- **Completion criteria:** The README has exactly two lines; documentation links resolve; the diagram's components are explained; state tracking includes Latest Decision and Current Progress instead of Latest Verification; timestamps follow the declared format.
