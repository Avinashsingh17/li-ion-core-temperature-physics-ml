# Knowledge base

A plain-markdown notebook for this 4-week battery internal-temperature project.
The user reads; Claude writes and maintains everything here.

## Layout

```
knowledge/
  README.md       this file
  index.md        catalog of every note (sources / concepts / decisions)
  log.md          append-only activity log
  sources/        one note per paper or dataset we ingest
  concepts/       ML + physics explainers, written for someone new-ish to ML
  decisions/      design decisions with rationale (physics choices, splits, etc.)
```

## Note conventions

### `sources/<slug>.md`
- Full citation (authors, title, venue, year, DOI/URL).
- 3–5 plain-language takeaways.
- Equations we will actually use, transcribed in LaTeX. Every symbol defined with its unit.
- A `How this project uses it` section.
- Links to related concept / decision notes.

### `concepts/<slug>.md`
Audience: comfortable with Python + basic stats, new to ML time-series. Each note:
- Defines the term.
- Gives the intuition.
- One concrete example tied to **this** project (battery signals).
- The failure mode it prevents.
- When it matters here.
- Tight — explainer, not textbook chapter.

### `decisions/<slug>.md`
- The question.
- Options considered.
- The call we made.
- Why.
- What would make us revisit.

(Decisions are often made in chat with another Claude; this note records the conclusion so it isn't lost.)

## Workflows

Three commands the user uses to drive the KB:

- **INGEST `<source>`** — read it, discuss key takeaways with the user **first**, then write the source note, update `index.md`, append to `log.md`, and create/update any concept notes for terms it introduces. **One source at a time; check in with the user.**
- **CONCEPT `<term>`** — whenever code or discussion introduces an ML/physics idea the user may not know (e.g. "blocked time-series CV", "entropic heat", "leakage", "HistGBR"), write or update its concept note and link it from wherever it came up. Do this **proactively** while building — that's the explain-the-ML-as-we-go goal.
- **LINT** — on request: scan for contradictions, stale claims, orphan notes, and concepts referenced but not yet explained.

## Log format

`log.md` is append-only, newest at bottom. One entry per action:

```
## [YYYY-MM-DD] ingest|concept|decision|query | <title>
one or two lines of detail
```

## Linking

Use plain relative markdown links: `[surface temp](../concepts/surface-vs-core-temp.md)`. No frontmatter, no plugins, no special syntax.

## Scope guardrail

Markdown files only. No Obsidian plugins, no qmd, no Marp, no vector search, no frontmatter tooling. If the project outgrows that, revisit — but for 4 weeks, a markdown tree + index + log is enough.
