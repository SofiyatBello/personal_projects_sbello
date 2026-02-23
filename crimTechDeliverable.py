from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import List, Tuple
from urllib.parse import urljoin

import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer



@dataclass
class Event:
    """Represents all information tied to a single calendar event
    
    """
    title: str
    description: str
    start_time: str
    location: str
    organization: str
    link: str


# webscrapping to extract event data from SEAS calendar 
# this was the most time-consuming part of the project, as the SEAS calendar is built on a Drupal-based event platform
#  No public API exists,so after long debugging with chat, scraping the calendar page and individual event detail pages yieled...something lol
#  
#
# events = .em-card elements 
# titles are in h3.em-card_title 
# dates/locations =  p.em-card_event-text

# extract the full description by following the event link to the detail page, 

seas_URL = "https://events.seas.harvard.edu"


def scrape_seas_events(limit: int = 30) -> List[Event]:
    """scraping pipeline:
    1 review the calendar HTML 
    2. parse with BeautifulSoup to extract event cards
    3. extract info like title, date/time, location, etc...
    5. return  list of events
    """
    # start w/ empty list; append Event objects as we parse
    events: List[Event] = []

    try:
        # Fetch calendar page with 15-second timeout (Drupal sites can be slow)
        resp = requests.get(urljoin(seas_URL, "/calendar"), timeout=15)
        resp.raise_for_status()  # Raise exception for HTTP errors (404, 500, etc.)
    except Exception as e:
        print(f"[SCRAPER] Request failed: {e}")
        return events  # Return empty list if fetch was unsuccessful
 # chat recommended adding error handling here to avoid crashing if the site is down or structure changes
 # also used chat for understnding Localist setup and how to extract location data, which is not in a consistent format across events (sometimes in text, sometimes in links)
    soup = BeautifulSoup(resp.content, "html.parser")
    
   
    cards = soup.select(".em-card, article, .views-row, .event")

    for card in cards[:limit]:  
        title_el = card.select_one("h3.em-card_title a, h3 a, h2 a")
        if title_el:
            title = title_el.get_text(strip=True)
            link = title_el.get("href")  # Event link is typically in the title element
            if link:
                # Convert relative URLs (e.g., "/event/123") to absolute (https://...)
                link = urljoin(seas_URL, link)
        else:
            heading = card.select_one("h3, h2")
            if not heading:
                continue
            title = heading.get_text(strip=True)
            link = None 

     
        time_el = card.select_one("p.em-card_event-text time, p.em-card_event-text, time")
        start_time = time_el.get_text(strip=True) if time_el else "TBD"

     
        location_el = card.select_one("p.em-card_event-text a[href]") or card.select_one(".location")
        location = (
            location_el.get_text(strip=True) if location_el else "TBD"
        )

       
        desc_el = card.select_one(".summary, .field--name-body, .description, p")
        description = desc_el.get_text(" ", strip=True) if desc_el else title

       
        if link:
            try:
                # Fetch detail page with 10-second timeout (shorter than initial page to fail fast)
                dresp = requests.get(link, timeout=10)
                dresp.raise_for_status()
                dsoup = BeautifulSoup(dresp.content, "html.parser")

                # Drupal CMS uses .field--name-body for main content field
                # Try multiple selectors for robustness across theme variations
                detail_desc = dsoup.select_one(
                    ".field--name-body, .node__content, .content, .event-description, .pane-content"
                )
                if detail_desc and detail_desc.get_text(strip=True):
                    # Replace card description with full detail page text
                    description = detail_desc.get_text(" ", strip=True)

                # Attempt to improve date/time from detail page (may be more precise)
                dt_el = dsoup.select_one("time")
                if dt_el and dt_el.get_text(strip=True):
                    start_time = dt_el.get_text(strip=True)

                # Update location with detail page version if available
                loc_el = dsoup.select_one(".location, .event-location, .field--name-field-location")
                if loc_el and loc_el.get_text(strip=True):
                    location = loc_el.get_text(strip=True)
            except Exception:
                pass
# if the shoe fits the keyword, add to list of recommended events :)
        events.append(
            Event(
                title=title,
                description=description,
                start_time=start_time,
                location=location,
                organization="Harvard SEAS",
                link=link or "",
            )
        )

    print(f"[SCRAPER] Extracted {len(events)} events.")
    for e in events[:5]:
        print(" -", e.title)

    return events

# ============================================================================
# ranking structure -  represent text as vectors where similar meanings produce adjacent vectors
# 
# 1. Encode user's topic and each event's text into vectors (embeddings)
# 2. cosine similarity between topic vector and each event vector
# 3. Filter events by threshold (keep relevance >= threshold)
# 4. Sort by score descending (highest relevance first)
#

_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str) -> np.ndarray:
    
    return _model.encode(text, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors.
    Cosine similarity measures the angle between vectors in embedding space:
    - 1.0: identical direction (perfect match)
    - 0.5: moderate alignment
    - 0.25: weak but detectable similarity
    - 0.0: orthogonal (unrelated)
    - -1.0: opposite direction
    
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def event_text(event: Event) -> str:
    return f"{event.title}. {event.description}. Hosted by {event.organization}."



def rank_events_by_topic(
    events: List[Event],
    topic: str,
    threshold: float = 0.25,
) -> List[Tuple[Event, float]]:
    """Score and filter events by relevance to a topic.
     Keep only events with score >= threshold (filter noise)
    Sort by most relevant first
    
   
    """

    topic_embedding = embed(topic)
    matches: List[Tuple[Event, float]] = []

    # Score each event against the topic
    for event in events:
        # Combine event fields, embed, and compute similarity to topic
        score = cosine_similarity(topic_embedding, embed(event_text(event)))
        
        # Keep only events above threshold
        if score >= threshold:
            matches.append((event, score))

    # Sort by score descending: highest-relevance events first
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


# print

def calendar_report(topic: str, matches: List[Tuple[Event, float]]) -> str:
  
    # empty results, no matching events found above threshold
    if not matches:
        return f"No relevant SEAS events found for topic: {topic}\n"

    lines = [f"SEAS events relevant to '{topic}':", ""]

    # Add each matched event with its relevance score and details
    for event, score in matches:
        lines.append(f"- {event.title}")
        lines.append(f"  AI Score: {score:.2f}")  # Format score to 2 decimal places
        lines.append(f"  Date/Time: {event.start_time}")
        lines.append(f"  Location: {event.location}")
        lines.append(f"  Link: {event.link}")
        lines.append("")  # Blank line between events for readability

    return "\n".join(lines)




# full pipeline 

def main():
    """""
    """
    # Setup argument parser for command-line interface
    parser = argparse.ArgumentParser(description="AI SEAS Event Alerts")
    parser.add_argument(
        "--topic",
        required=True,
        help="Topic to search for (e.g., 'engineering', 'AI', 'biology')"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Minimum relevance score [0-1] to include an event (default: 0.25)"
    )

    args = parser.parse_args()

    # 1: gather events from SEAS calendar
    print("Fetching live SEAS events...")
    events = scrape_seas_events()

    # rank events by relevance
    print(f"Scoring events for topic: {args.topic}")
    matches = rank_events_by_topic(events, args.topic, args.threshold)

    # 3: Format results
    body = calendar_report(args.topic, matches)

    # 4: Display report
    print("\n--- EVENT REPORT---\n")
    print(body)

if __name__ == "__main__":
    main()