import re
from datetime import datetime
from urllib.parse import unquote

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from tools import search_tool, wiki_tool, save_to_txt


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

MODEL_NAME = "qwen2.5:3b"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,
)

SECTION_WORD_TARGET = 180

SECTION_HEADINGS = [
    "1. Introduction",
    "2. Background and Purpose",
    "3. Architecture and Core Components",
    "4. Tools, Chains, and Agents",
    "5. How It Works",
    "6. Real-World Applications",
    "7. Advantages",
    "8. Limitations and Challenges",
    "9. Practical Example",
    "10. Conclusion",
]


# ==================================================
# 1. CLEAN AND VALIDATE TEXT
# ==================================================

def clean_text(text: str) -> str:
    """Remove repeated paragraphs without deleting headings."""

    paragraphs = re.split(r"\n\s*\n", text.strip())
    seen = set()
    cleaned = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        normalized = " ".join(paragraph.casefold().split())

        # Keep headings even if a heading appears more than once.
        if paragraph.startswith("#"):
            cleaned.append(paragraph)
            continue

        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(paragraph)

    return "\n\n".join(cleaned)


def count_words(text: str) -> int:
    """Count words in a piece of text."""

    return len(re.findall(r"\b[\w'-]+\b", text))


def extract_urls(text: str) -> list[str]:
    """Extract URLs that actually appear in research results."""

    urls = re.findall(r"https?://[^\s<>\"']+", text)
    cleaned_urls = []

    for url in urls:
        url = url.rstrip(".,;:!?)\]")

        if url not in cleaned_urls:
            cleaned_urls.append(url)

    return cleaned_urls


def is_relevant_title(title: str, topic: str) -> bool:
    """Keep article titles that overlap with the research topic."""

    stop_words = {
        "the", "and", "for", "with", "from", "into",
        "about", "what", "how", "why", "using", "real",
        "world", "applications", "research", "explain",
        "explanation", "overview", "detailed", "report",
    }

    topic_words = {
        word.casefold()
        for word in re.findall(r"[a-zA-Z0-9]+", topic)
        if len(word) > 2 and word.casefold() not in stop_words
    }

    title_words = {
        word.casefold()
        for word in re.findall(r"[a-zA-Z0-9]+", unquote(title))
        if len(word) > 2
    }

    if not topic_words:
        return True

    return bool(topic_words & title_words)


# ==================================================
# 2. RESEARCH THE TOPIC
# ==================================================

def collect_research(topic: str):
    """Collect research from both tools and record their outcomes."""

    research_notes = []
    successful_tools = []
    attempted_tools = []
    sources = []

    # Focus the search on the exact topic.
    search_queries = [
        f"{topic} official documentation architecture components agents tools",
        f"{topic} working principles examples real-world applications limitations",
    ]

    print("\n[1/3] Searching the web...")

    for query in search_queries:
        attempted_tools.append("search")

        try:
            result = search_tool.invoke({"query": query})
            result = str(result).strip()

            if (
                result
                and "WEB_SEARCH_FAILED" not in result
                and "No useful web search results" not in result
            ):
                successful_tools.append("search")
                research_notes.append(
                    f"WEB SEARCH QUERY: {query}\n{result}"
                )

                # Only keep URLs whose surrounding result text
                # contains terms relevant to the requested topic.
                for line in result.splitlines():
                    if "http://" not in line and "https://" not in line:
                        continue

                    line_words = set(
                        re.findall(
                            r"[a-zA-Z0-9]+",
                            line.casefold(),
                        )
                    )
                    topic_words = set(
                        re.findall(
                            r"[a-zA-Z0-9]+",
                            topic.casefold(),
                        )
                    )

                    if topic_words & line_words:
                        sources.extend(extract_urls(line))

        except Exception as error:
            research_notes.append(
                f"Web search failed: {type(error).__name__}: {error}"
            )

    print("[2/3] Searching Wikipedia...")

    wiki_queries = [
        topic,
        f"{topic} architecture and components",
        f"{topic} applications and examples",
    ]

    for query in wiki_queries:
        attempted_tools.append("wikipedia")

        try:
            result = str(
                wiki_tool.invoke({"query": query})
            ).strip()

            if (
                result
                and not result.startswith("WIKIPEDIA_FAILED:")
                and not result.startswith("No Wikipedia articles found")
                and not result.startswith("Wikipedia returned no usable")
            ):
                successful_tools.append("wikipedia")
                research_notes.append(
                    f"WIKIPEDIA QUERY: {query}\n{result}"
                )

                # Extract article titles and their actual URLs.
                article_blocks = re.split(
                    r"(?=ARTICLE TITLE:)",
                    result,
                )

                for block in article_blocks:
                    title_match = re.search(
                        r"ARTICLE TITLE:\s*(.+)",
                        block,
                    )
                    url_match = re.search(
                        r"ARTICLE URL:\s*(https?://\S+)",
                        block,
                    )

                    if title_match and url_match:
                        title = title_match.group(1).strip()
                        url = url_match.group(1).strip()

                        if is_relevant_title(title, topic):
                            sources.append(url)

        except Exception as error:
            research_notes.append(
                f"Wikipedia search failed: {type(error).__name__}: {error}"
            )

    # Keep tool names unique, preserving their order.
    successful_tools = list(dict.fromkeys(successful_tools))
    attempted_tools = list(dict.fromkeys(attempted_tools))
    sources = list(dict.fromkeys(sources))

    return (
        "\n\n".join(research_notes),
        successful_tools,
        attempted_tools,
        sources,
    )


# ==================================================
# 3. GENERATE THE REPORT SECTION BY SECTION
# ==================================================

def generate_section(
    topic: str,
    section_heading: str,
    research_notes: str,
) -> str:
    """Generate one detailed, topic-focused report section."""

    prompt = f"""
You are a careful technical research writer.

Write the section "{section_heading}" for a research report about:
{topic}

RESEARCH EVIDENCE:
{research_notes[:18000]}

STRICT RULES:
1. Discuss ONLY information directly relevant to the topic.
2. Do not introduce unrelated subjects, products, companies,
   scientific fields, or technologies.
3. Use the supplied evidence. Do not invent facts, statistics,
   citations, or source URLs.
4. Explain technical concepts clearly, including how they work
   and why they matter.
5. Include concrete examples when relevant to this section.
6. Do not repeat the introduction or content belonging to other sections.
7. If the evidence is insufficient for a claim, qualify it or omit it.
8. Write approximately {SECTION_WORD_TARGET} words when evidence permits.
9. Return only the section heading and its content.
10. Never leave the section empty.

The report must be useful to a student learning this topic.
"""

    response = llm.invoke(prompt)
    content = response.content

    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            f"The model returned an empty section: {section_heading}"
        )

    content = content.strip()

    # If the model returned a heading but no meaningful explanation,
    # retry once with a clearer instruction.
    body = re.sub(
        r"^\s*#+.*$",
        "",
        content,
        count=1,
        flags=re.MULTILINE,
    ).strip()

    if count_words(body) < 35:
        retry_prompt = prompt + """

Your previous attempt was too short or incomplete.
Write a meaningful explanation with relevant details and examples.
Do not return a heading alone.
"""

        retry = llm.invoke(retry_prompt)
        retry_content = retry.content

        if isinstance(retry_content, str) and retry_content.strip():
            if count_words(retry_content) > count_words(content):
                content = retry_content.strip()

    # Ensure the expected heading is present.
    if section_heading.casefold() not in content.casefold():
        content = f"## {section_heading}\n\n{content}"

    return content


def generate_report(topic: str, research_notes: str) -> str:
    """Build a complete report with all required sections."""

    if not research_notes.strip():
        research_notes = (
            "No usable external research results were retrieved. "
            "Clearly distinguish general background knowledge from "
            "claims that could not be verified. Do not invent sources."
        )

    sections = []

    for index, heading in enumerate(SECTION_HEADINGS, start=1):
        print(
            f"[3/3] Writing section {index}/{len(SECTION_HEADINGS)}: "
            f"{heading}"
        )

        section = generate_section(
            topic=topic,
            section_heading=heading,
            research_notes=research_notes,
        )

        sections.append(section)

    report_body = clean_text("\n\n".join(sections))

    # Check that every required heading exists in the final report.
    missing_headings = [
        heading
        for heading in SECTION_HEADINGS
        if heading.casefold() not in report_body.casefold()
    ]

    if missing_headings:
        raise ValueError(
            "The report is incomplete. Missing sections: "
            + ", ".join(missing_headings)
        )

    return report_body


# ==================================================
# 4. FORMAT THE COMPLETE REPORT
# ==================================================

def format_report(
    topic: str,
    report_body: str,
    sources: list[str],
    successful_tools: list[str],
    attempted_tools: list[str],
) -> str:
    """Include the report, sources, and tool status in one text file."""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    source_section = (
        "\n".join(
            f"{index}. {url}"
            for index, url in enumerate(sources, start=1)
        )
        if sources
        else (
            "No relevant source URLs could be extracted from the "
            "available tool results. Verify the report before relying on it."
        )
    )

    successful_section = (
        "\n".join(f"- {name}" for name in successful_tools)
        if successful_tools
        else "- No research tool returned usable results."
    )

    attempted_section = (
        "\n".join(f"- {name}" for name in attempted_tools)
        if attempted_tools
        else "- No tools were attempted."
    )

    return f"""
AI RESEARCH REPORT
============================================================

Generated: {generated_at}
Model: {MODEL_NAME}

TOPIC
{topic}

DETAILED REPORT
============================================================

{report_body}

============================================================
SOURCE URLS EXTRACTED FROM RESEARCH RESULTS
============================================================

{source_section}

============================================================
TOOLS THAT RETURNED USABLE RESULTS
============================================================

{successful_section}

============================================================
TOOLS ATTEMPTED
============================================================

{attempted_section}

============================================================
REPORT STATISTICS
============================================================

Word count: {count_words(report_body)}

Note: Sources are collected from research-tool output. Their
relevance and factual accuracy should be checked before academic use.
""".strip()


# ==================================================
# 5. RUN THE RESEARCH ASSISTANT
# ==================================================

def show_banner():
    print("\n" + "=" * 60)
    print("             AI RESEARCH ASSISTANT")
    print("=" * 60)
    print(f"Model       : {MODEL_NAME}")
    print("Search      : DuckDuckGo")
    print("Reference   : Wikipedia")
    print("Output      : reports/")
    print("Type 'exit' or 'quit' to close.")
    print("-" * 60)


def run_research(topic: str):
    print("\nCollecting research. Please wait...\n")

    (
        research_notes,
        successful_tools,
        attempted_tools,
        sources,
    ) = collect_research(topic)

    if not research_notes.strip():
        print(
            "Warning: Neither research tool returned usable results. "
            "The report may be less reliable."
        )

    report_body = generate_report(topic, research_notes)

    # Keep only sources extracted from tool output.
    full_report = format_report(
        topic=topic,
        report_body=report_body,
        sources=sources,
        successful_tools=successful_tools,
        attempted_tools=attempted_tools,
    )

    # Save the exact same complete report shown to the user.
    saved_path = save_to_txt(full_report)

    print("\n" + "=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    print(full_report)
    print("\n" + "-" * 60)
    print(f"Complete report saved to:\n{saved_path}")


def main():
    show_banner()

    while True:
        try:
            topic = input("\nEnter your research topic: ").strip()

            if topic.casefold() in {"exit", "quit"}:
                print("\nThank you for using AI Research Assistant!")
                break

            if not topic:
                print("Please enter a research topic.")
                continue

            run_research(topic)

        except KeyboardInterrupt:
            print("\n\nResearch cancelled.")
            break

        except Exception as error:
            print(
                f"\nResearch error: {type(error).__name__}: {error}"
            )
            print(
                "Check that Ollama is running, the model is installed, "
                "and your network connection is available."
            )


if __name__ == "__main__":
    main()