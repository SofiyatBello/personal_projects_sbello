from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


# -----------------------------
# Data model
# -----------------------------
# Event holds the key fields we care about when creating alerts.
@dataclass
class Event:
    title: str
    description: str
    start_time: str
    location: str
    organization: str
    link: str


# -----------------------------
# Text preprocessing
# -----------------------------
# Convert raw text into a set of simple lowercase words.
# We remove a few punctuation characters to keep logic easy for beginners.
def tokenize(text: str) -> set[str]:
    clean = text.lower().replace("/", " ").replace("-", " ").replace(",", " ").replace(".", " ")
    return {word for word in clean.split() if word}


# Combine event fields into one searchable text blob.
def event_text(event: Event) -> str:
    return f"{event.title} {event.description} {event.organization}"


# -----------------------------
# Scoring and ranking
# -----------------------------
# Score formula:
#   overlap words / number of topic words
# Example:
#   topic = "ai ethics" => {ai, ethics}
#   event = "ai policy ethics" => overlap is 2
#   score = 2 / 2 = 1.0
def relevance_score(topic: str, event: Event) -> float:
    topic_words = tokenize(topic)
    if not topic_words:
        return 0.0

    words = tokenize(event_text(event))
    overlap = topic_words.intersection(words)
    return len(overlap) / len(topic_words)


# Rank events from highest to lowest score and keep only those above threshold.
def rank_events_by_topic(events: list[Event], topic: str, threshold: float = 0.3) -> list[tuple[Event, float]]:
    matches: list[tuple[Event, float]] = []

    for event in events:
        score = relevance_score(topic, event)
        if score >= threshold:
            matches.append((event, score))

    matches.sort(key=lambda item: item[1], reverse=True)
    return matches


# -----------------------------
# Input / output helpers
# -----------------------------
# Read events from JSON and convert each dict into an Event object.
def load_events_from_json(path: str | Path) -> list[Event]:
    raw = json.loads(Path(path).read_text())
    return [Event(**item) for item in raw]


# Build a human-readable email-style message.
# (This version only PREVIEWS the alert text; it does not send SMTP email.)
def compose_email(topic: str, matches: list[tuple[Event, float]]) -> str:
    if not matches:
        return f"No relevant SEAS events found for topic: {topic}\n"

    lines = [f"SEAS events relevant to '{topic}':", ""]

    for event, score in matches:
        lines.append(f"- {event.title}")
        lines.append(f"  Relevance score: {score:.2f}")
        lines.append(f"  Date/Time: {event.start_time}")
        lines.append(f"  Location: {event.location}")
        lines.append(f"  Organization: {event.organization}")
        lines.append(f"  Link: {event.link}")
        lines.append("")

    return "\n".join(lines)


# -----------------------------
# CLI entry point
# -----------------------------
# This command-line interface:
# 1) loads events
# 2) scores/ranks them by topic
# 3) prints a preview message
# There is intentionally no SMTP sending in this beginner version.
def main() -> None:
    parser = argparse.ArgumentParser(description="Simple SEAS topic-based event alerts")
    parser.add_argument("--topic", required=True, help="Topic to match, e.g. 'ai ethics'")
    parser.add_argument("--events-json", default="examples/seas_events.json", help="Path to events JSON file")
    parser.add_argument("--threshold", type=float, default=0.3, help="Minimum score to include an event")
    args = parser.parse_args()

    events = load_events_from_json(args.events_json)
    matches = rank_events_by_topic(events, args.topic, args.threshold)
    body = compose_email(args.topic, matches)

    print(f"Generated {len(matches)} alert(s) for topic: {args.topic}")
    print("\n--- ALERT PREVIEW ---\n")
    print(body)


if __name__ == "__main__":
    main()
