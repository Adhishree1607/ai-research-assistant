# AI Research Assistant

A terminal-based AI research assistant built with Python, LangChain, and Ollama. It helps users research topics using web search and Wikipedia, then saves research reports to a text file.

## Features

- Research topics using an AI language model.
- Search the web using DuckDuckGo.
- Retrieve information from Wikipedia.
- Save research reports to `research_output.txt`.
- Display structured research results in the terminal.
- Run locally using Ollama.

## Tech Stack

- Python
- LangChain
- Ollama
- Qwen 2.5 (3B)
- DuckDuckGo Search
- Wikipedia

## Project Structure

```text
Ai agent/
├── main.py
├── tools.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
└── research_output.txt
```

The `venv/` and `__pycache__/` folders are excluded from the project listing because they are local environment and cache folders.

## Prerequisites

- Python installed
- Ollama installed and running
- Git (optional, for version control)

## Installation

1. Open the project folder in VS Code.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install the required packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Download the required Ollama model:

   ```powershell
   ollama pull qwen2.5:3b
   ```

5. Start Ollama if it is not already running.

## Run the Project

```powershell
python main.py
```

Enter a research topic when prompted. Type `exit` or `quit` to close the assistant.

## Output

Research reports are saved to `research_output.txt`.

## Security

Do not upload `.env` files, API keys, passwords, or other secrets to GitHub.

## Future Improvements

- Add a graphical user interface.
- Improve source citations.
- Export reports to PDF.
- Add support for more local language models.
