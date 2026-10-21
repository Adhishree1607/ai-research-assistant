import os
import re
from datetime import datetime
from pathlib import Path

import requests


# Folder where generated reports will be saved
BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"


def search_tool(query: str) -> str:
    """
    Search the web for information about a topic.
    Uses a short timeout so a slow search does not block the program for too long.
    """
    try:
        # Support both newer and older versions of the DDGS package.
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS(timeout=8) as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No web search results were found."

        formatted_results = []

        for index, result in enumerate(results, start=1):
            title = result.get("title", "Untitled")
            body = result.get("body", "")
            url = result.get("href") or result.get("link") or ""

            formatted_results.append(
                f"{index}. {title}\n"
                f"Summary: {body}\n"
                f"URL: {url}"
            )

        return "\n\n".join(formatted_results)

    except Exception as error:
        return (
            "Web search was unavailable. "
            f"Reason: {type(error).__name__}: {error}"
        )


def wiki_tool(topic: str) -> str:
    """
    Retrieve a short topic summary from Wikipedia's MediaWiki API.
    """
    url = "https://en.wikipedia.org/w/api.php"

    headers = {
        "User-Agent": "AIResearchAssistant/1.0 (educational research project)"
    }

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "redirects": 1,
        "titles": topic,
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()

        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        for page in pages.values():
            title = page.get("title", topic)
            extract = page.get("extract", "").strip()

            if extract:
                return f"Wikipedia summary for {title}:\n{extract}"

        return f"No Wikipedia summary was found for '{topic}'."

    except requests.RequestException as error:
        return f"Wikipedia request failed: {error}"

    except ValueError:
        return "Wikipedia returned a response that could not be read."


def clean_text(text: str) -> str:
    """Remove excessive whitespace from generated text."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def save_to_txt(content: str, topic: str) -> str:
    """
    Save a report with a unique filename inside the reports directory.
    Returns the full path of the saved report.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    safe_topic = re.sub(r"[^a-zA-Z0-9_-]+", "_", topic.strip())
    safe_topic = safe_topic.strip("_")[:50] or "research"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"research_report_{safe_topic}_{timestamp}.txt"
    file_path = REPORTS_DIR / filename

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(clean_text(content))
        file.write("\n")

    return str(file_path)