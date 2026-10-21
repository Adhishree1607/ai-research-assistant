from langchain_ollama import ChatOllama

from tools import search_tool, wiki_tool, save_to_txt, clean_text


# Local Ollama model
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.2,
    num_predict=1800,
)


SECTION_HEADINGS = [
    "1. Introduction",
    "2. Background and Purpose",
    "3. Architecture and Core Components",
    "4. Tools, Chains, and Agents",
    "5. How It Works",
    "6. Real-World Applications",
    "7. Advantages and Limitations",
    "8. Practical Example",
    "9. Conclusion",
]


def collect_research(topic: str) -> str:
    """
    Collect information using one web search and one Wikipedia request.
    Each tool has its own timeout or error handling.
    """
    print("\n[1/2] Searching the web...")

    web_results = search_tool(topic)

    print("[2/2] Retrieving Wikipedia information...")

    wiki_results = wiki_tool(topic)

    research_notes = f"""
RESEARCH TOPIC:
{topic}

WEB SEARCH RESULTS:
{web_results}

WIKIPEDIA INFORMATION:
{wiki_results}
"""

    return research_notes


def generate_report(topic: str, research_notes: str) -> str:
    """
    Generate the complete report with one LLM call rather than
    making a separate LLM call for every section.
    """
    prompt = f"""
You are a careful technical research assistant.

Write a detailed, well-organized report about this topic:

{topic}

Use the following research notes as supporting information:

---------------- RESEARCH NOTES ----------------
{research_notes[:10000]}
-------------------------------------------------

Use these exact section headings:

{chr(10).join(SECTION_HEADINGS)}

REPORT REQUIREMENTS:

1. Explain the requested topic clearly and in depth.
2. Write approximately 800-1200 words when the available evidence supports it.
3. Give every section meaningful, topic-specific content.
4. Explain architecture, components, tools, chains, agents, or workflow
   only when they are relevant to the topic.
5. Include practical examples and real-world applications where appropriate.
6. Discuss advantages, limitations, and challenges.
7. Keep the explanation focused on the requested topic.
8. Do not repeat the same information across sections.
9. Do not invent statistics, quotations, research findings, or URLs.
10. If the research notes are incomplete or a fact cannot be verified,
    acknowledge the limitation instead of guessing.
11. Distinguish established facts from illustrative examples.
12. Return plain text with clear headings.
13. Do not return JSON, markdown code fences, or instructions to the reader.
14. Include all nine headings in the specified order.

Write the complete report in this single response.
"""

    print("\nGenerating the complete report in one model call...")
    response = llm.invoke(prompt)

    report = response.content

    if not isinstance(report, str) or not report.strip():
        raise ValueError("The model returned an empty report.")

    report = clean_text(report)

    missing_headings = [
        heading
        for heading in SECTION_HEADINGS
        if heading.casefold() not in report.casefold()
    ]

    if missing_headings:
        print(
            "\nWarning: The report may be incomplete. "
            "Missing headings: " + ", ".join(missing_headings)
        )

    return report


def main():
    print("=" * 55)
    print("          AI RESEARCH ASSISTANT")
    print("=" * 55)

    topic = input("\nEnter the research topic: ").strip()

    if not topic:
        print("Error: Please enter a valid research topic.")
        return

    try:
        research_notes = collect_research(topic)

        print("\nResearch collection finished.")

        report = generate_report(topic, research_notes)

        print("\nSaving the report...")

        saved_path = save_to_txt(report, topic)

        print("\n" + "=" * 55)
        print("REPORT GENERATED SUCCESSFULLY")
        print("=" * 55)

        print(f"\nTopic: {topic}")
        print(f"Saved report: {saved_path}")

        print("\nREPORT PREVIEW")
        print("-" * 55)
        print(report[:2500])

        if len(report) > 2500:
            print("\n[Preview shortened. Open the saved file to read the full report.]")

    except KeyboardInterrupt:
        print("\nResearch cancelled by the user.")

    except Exception as error:
        print(f"\nAn error occurred: {type(error).__name__}: {error}")
        print(
            "\nCheck that Ollama is running and the model "
            "'qwen2.5:3b' is installed."
        )


if __name__ == "__main__":
    main()