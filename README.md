# AI-Powered Event Alert System for SEAS (Beginner Version)

This is a small prototype that helps find Harvard SEAS events related to a topic.

## What changed in this version

- Removed SMTP/email sending code.
- Kept only alert preview output in the terminal.
- Added code comments that explain each section.

## How it works (simple)

1. Load events from a JSON file.
2. Split your topic into words (example: `"ai ethics"` -> `ai`, `ethics`).
3. Split each event's text into words.
4. Score each event by word overlap.
5. Keep events above a threshold.
6. Print an alert preview message.

This is intentionally simple so a beginner can read and change it quickly.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.seas_alerts --topic "ai ethics"
```

## Example command

```bash
python -m src.seas_alerts \
  --topic "religious life" \
  --events-json examples/seas_events.json \
  --threshold 0.3
```

## Notes

- This is a prototype and not production-ready.
- The scoring is basic word overlap, not advanced AI embeddings.
- It is easy to improve later (synonyms, embeddings, calendar scraping, scheduling).
