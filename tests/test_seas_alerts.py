import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.seas_alerts import Event, compose_email, rank_events_by_topic, relevance_score


def test_relevance_score_matches_topic_words():
    event = Event(
        title="AI Ethics and Accountability",
        description="How machine learning affects civil rights.",
        start_time="2026-03-10 12:00",
        location="SEAS",
        organization="SEAS",
        link="https://example.com/ai",
    )

    score = relevance_score("ai ethics", event)
    assert score == 1.0


def test_rank_events_returns_relevant_matches():
    events = [
        Event(
            title="AI Ethics and Accountability",
            description="How machine learning affects civil rights.",
            start_time="2026-03-10 12:00",
            location="SEAS",
            organization="SEAS",
            link="https://example.com/ai",
        ),
        Event(
            title="Classical Music Performance",
            description="Orchestra showcase",
            start_time="2026-03-12 20:00",
            location="Lowell House",
            organization="Arts",
            link="https://example.com/music",
        ),
    ]

    matches = rank_events_by_topic(events, "ai ethics", threshold=0.5)

    assert len(matches) == 1
    assert matches[0][0].title == "AI Ethics and Accountability"


def test_compose_email_no_matches_message():
    body = compose_email("religious life", [])
    assert "No relevant SEAS events found" in body
