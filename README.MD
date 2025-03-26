# Personalized Job Posting Analyzer (via OpenRouter)

This Python script automates the analysis of online job postings against **your specific professional profile**, using Large Language Models (LLMs) accessed via the OpenRouter.ai API. It takes a job posting URL as a command-line argument, reads your profile from `candidate_profile.json`, performs a multi-step analysis, and **conditionally** proceeds to perform deep-dive company research and generate tailored resume suggestions and a cover letter draft if the initial fit seems promising.

## Features

*   **Personalized Analysis:** Compares job requirements directly against your profile stored in `candidate_profile.json`.
*   **Command-Line Interface:** Accepts the target job posting URL directly as a command-line argument (primary usage method).
*   **Interactive Mode:** An alternative script (`job_analyzer_interactive.py`) prompts for the URL if preferred for manual use.
*   **External Profile Data:** Reads your detailed professional profile from a separate `candidate_profile.json` file for easy updates.
*   **Web Scraping (Best-Effort):** Attempts to scrape text content from the job posting URL.
*   **Core Analysis (Steps 1-5):**
    *   Analyzes the job description (responsibilities, qualifications, keywords).
    *   Maps your profile against job requirements (skills, experience, achievements).
    *   Defines the ideal candidate profile based on the posting.
    *   Crafts personalized positioning points for you.
    *   Performs an initial company vetting.
    *   Synthesizes findings into an overall fit assessment and your unique value proposition (UVP).
*   **Conditional Deep Dive & Generation (Steps 5.5-7):**
    *   **Triggered:** Only if the initial fit assessment (Step 5) indicates a promising match (e.g., "Good Fit", "Strong Fit", "Moderate Fit").
    *   **Step 5.5: Deep Dive Company Research:** Performs comprehensive research (history, leadership, products, competitors, financials, culture, news) for rich context and interview prep.
    *   **Step 6: Resume Tailoring Suggestions:** Suggests specific text changes/rephrasing for your resume based on the job, your profile, and the deep dive research.
    *   **Step 7: Cover Letter Draft Generation:** Creates a tailored cover letter draft incorporating job requirements, your profile highlights, and insights from the deep dive research.
*   **Model Agnostic (via OpenRouter):** Configurable to use various LLMs available through OpenRouter.
*   **Formatted Output:** Generates a structured report printed to the console and saved to a Markdown file, including conditional sections only if they were executed.

## Requirements

*   Python 3.7+
*   An account with [OpenRouter.ai](https://openrouter.ai/) to obtain an API key.

## Installation

1.  **Clone the repository (or download files):**
    Place `job_analyzer_cli.py`, `job_analyzer_interactive.py` (optional), and create `candidate_profile.json` (see step 3) in the same directory.
2.  **Install dependencies:**
    Create `requirements.txt`:
    ```text
    openai
    requests
    beautifulsoup4
    ```
    Run:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Create and Populate `candidate_profile.json`:**
    *   Create `candidate_profile.json` in the same directory.
    *   **Paste your complete professional profile data into this file in valid JSON format.** This file is crucial. Use a structured template (like the one discussed previously).

## Configuration

1.  **API Key:** Set your OpenRouter API key as an environment variable `OPENROUTER_API_KEY`. **Do not hardcode it.**
    *(See previous README versions for platform-specific instructions)*
2.  **Candidate Data (`candidate_profile.json`):** Ensure this file exists and contains your accurate, up-to-date profile in valid JSON format.
3.  **Model Selection (Optional):**
    Inside the `.py` scripts, change `model_name`. Consider models with strong web browsing/reasoning (e.g., `google/gemini-1.5-pro-latest`, `openai/gpt-4o`) for potentially better results on research steps (4 & 5.5).
4.  **OpenRouter Ranking Headers (Optional):**
    Set `YOUR_SITE_URL` and `YOUR_SITE_NAME` environment variables if desired.

## Usage (CLI Recommended)

The primary script is `job_analyzer_cli.py`. Run it from your terminal, providing the job posting URL as an argument. **Enclose the URL in quotes if it contains special characters.**

```bash
python job_analyzer_cli.py "<job_posting_url_here>"
```

**Example:**

```bash
python job_analyzer_cli.py "https://boards.greenhouse.io/openai/jobs/4016174007"
```

**(Optional Interactive Usage):**

If you prefer, you can run the interactive script:
```bash
python job_analyzer_interactive.py
```
It will prompt you to enter the URL.

**Output:**

The script prints progress and saves a comprehensive report (e.g., `job_analysis_report_cli_deepdive.md`) including all executed steps. Steps 5.5, 6, and 7 will only appear in the report if the initial fit assessment was positive.

## Important Notes & Limitations

*   **Maintain `candidate_profile.json`:** Accuracy is key. Ensure valid JSON.
*   **Conditional Steps:** Deep dive research and generation steps only run if the initial fit seems promising, saving time/cost on poor matches.
*   **Web Scraping Fragility:** May fail on dynamic sites. LLM URL access capability becomes important.
*   **Model Capabilities & Cost:** Research quality depends on the chosen LLM. Deep dive/generation steps increase token usage and cost when triggered.
*   **Review Generated Content:** Always treat resume suggestions and cover letter drafts as **AI-assisted first drafts**. Review, edit, and personalize them thoroughly. Formatting requires manual work.
*   **Output Variability:** LLM outputs can differ slightly between runs.

## Contributing

Contributions are welcome via issues or pull requests.

## License

(Specify your license, e.g., MIT License)
