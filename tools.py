from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests
from langchain_core.tools import StructuredTool


# ==================================================
# CONFIGURATION
# ==================================================

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "AIResearchAssistant/1.0 (educational project)"
}

REQUEST_TIMEOUT = 20


# ==================================================
# 1. DUCKDUCKGO WEB SEARCH
# ==================================================

def search_web(query: str) -> str:
    """Search the web and return relevant search results."""

    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        search = DuckDuckGoSearchRun()
        result = search.invoke(query)

        if not result or not str(result).strip():
            return "No useful web search results were returned."

        return str(result)

    except Exception as error:
        return (
            "WEB_SEARCH_FAILED\n"
            f"Error type: {type(error).__name__}\n"
            f"Details: {error}\n"
            "No verified web-search results were obtained."
        )


search_tool = StructuredTool.from_function(
    func=search_web,
    name="search",
    description=(
        "Search the web for information about the requested research topic. "
        "Use focused queries about the topic's architecture, components, "
        "working principles, examples, applications, and limitations."
    ),
)


# ==================================================
# 2. WIKIPEDIA SEARCH
# Uses the official MediaWiki API directly.
# ==================================================

def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return article summaries with URLs."""

    try:
        search_response = requests.get(
            WIKIPEDIA_API_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 5,
                "format": "json",
            },
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        search_response.raise_for_status()
        search_data = search_response.json()

        results = search_data.get("query", {}).get("search", [])

        if not results:
            return f"No Wikipedia articles found for: {query}"

        articles = []

        for result in results:
            title = result.get("title", "").strip()

            if not title:
                continue

            summary_response = requests.get(
                WIKIPEDIA_API_URL,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "titles": title,
                    "format": "json",
                },
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            summary_response.raise_for_status()
            summary_data = summary_response.json()

            pages = (
                summary_data.get("query", {}).get("pages", {})
            )

            for page in pages.values():
                summary = page.get("extract", "").strip()

                if not summary:
                    continue

                article_url = (
                    "https://en.wikipedia.org/wiki/"
                    + quote(title.replace(" ", "_"), safe="()_")
                )

                articles.append(
                    f"ARTICLE TITLE: {title}\n"
                    f"ARTICLE URL: {article_url}\n"
                    f"ARTICLE SUMMARY:\n{summary[:3500]}"
                )

        if not articles:
            return (
                f"Wikipedia returned no usable article summaries "
                f"for: {query}"
            )

        return "\n\n" + ("\n\n" + "=" * 60 + "\n\n").join(articles)

    except requests.exceptions.Timeout:
        return "WIKIPEDIA_FAILED: The request timed out."

    except requests.exceptions.RequestException as error:
        return (
            "WIKIPEDIA_FAILED: "
            f"{type(error).__name__}: {error}"
        )

    except (ValueError, KeyError) as error:
        return (
            "WIKIPEDIA_FAILED: Invalid API response. "
            f"{type(error).__name__}: {error}"
        )

    except Exception as error:
        return (
            "WIKIPEDIA_FAILED: "
            f"{type(error).__name__}: {error}"
        )


wiki_tool = StructuredTool.from_function(
    func=search_wikipedia,
    name="wikipedia",
    description=(
        "Search Wikipedia for relevant background information, "
        "technical definitions, architecture, and applications. "
        "Use a specific research query."
    ),
)


# ==================================================
# 3. SAVE A COMPLETE REPORT
# ==================================================

def save_to_txt(data: str) -> str:
    """Save the complete report to a unique text file."""

    reports_folder = Path(__file__).resolve().parent / "reports"
    reports_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S-%f"
    )

    filename = reports_folder / f"research_report_{timestamp}.txt"

    with filename.open("w", encoding="utf-8") as file:
        file.write(data.rstrip() + "\n")

    return str(filename)