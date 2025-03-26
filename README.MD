# Personalized Job Posting Analyzer (via OpenRouter)

This Python script automates the analysis of online job postings against **your specific professional profile**, using Large Language Models (LLMs) accessed via the OpenRouter.ai API. It takes a job posting URL as a command-line argument, reads your profile from `candidate_profile.json`, performs a multi-step analysis comparing your profile to the job description, vets the company/industry, and generates a comprehensive, personalized report.

## Features

*   **Personalized Analysis:** Compares job requirements directly against your skills, experience, and accomplishments stored in `candidate_profile.json`.
*   **Command-Line Interface:** Accepts the target job posting URL directly as a command-line argument.
*   **External Profile Data:** Reads your detailed professional profile from a separate `candidate_profile.json` file for easy updates and maintenance.
*   **Web Scraping (Best-Effort):** Attempts to scrape text content directly from the provided job posting URL.
*   **Job Description Analysis:** Extracts key responsibilities, required/preferred qualifications, and essential keywords/skills from the posting.
*   **Candidate Data Mapping:** Compares your profile (skills, experience, achievements) against the extracted job requirements, identifying strengths and potential gaps.
*   **Personalized Positioning:** Crafts a sample professional summary tailored for you and the specific job.
*   **Company & Industry Vetting:** Performs external research (via LLM capabilities) on the company's business, reputation, culture, industry position, competitors, and trends.
*   **Synthesis & Fit Assessment:** Provides an overall assessment of *your* fit for the role and your unique value proposition.
*   **Model Agnostic (via OpenRouter):** Configurable to use various LLMs available through OpenRouter (e.g., Gemini, GPT models, Mistral, etc.).
*   **Formatted Output:** Generates a structured report printed to the console and optionally saved to a Markdown file.

## Requirements

*   Python 3.7+
*   An account with [OpenRouter.ai](https://openrouter.ai/) to obtain an API key.

## Installation

1.  **Clone the repository (or download files):**
    Place `job_analyzer_cli.py` and create `candidate_profile.json` (see step 3 below) in the same directory.
2.  **Install dependencies:**
    Create a file named `requirements.txt` with the following content:
    ```text
    openai
    requests
    beautifulsoup4
    ```
    Then, run:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Create and Populate `candidate_profile.json`:**
    *   Create a file named `candidate_profile.json` in the same directory as `job_analyzer_cli.py`.
    *   **Paste your complete professional profile data into this file, ensuring it follows the correct JSON format.** Use the structure provided in the previous interaction (or adapt it carefully) as a template. This file is crucial for the personalized analysis.

## Configuration

1.  **API Key:** Set your OpenRouter API key as an environment variable named `OPENROUTER_API_KEY`. **Do not hardcode it.**
    *   **Linux/macOS:** `export OPENROUTER_API_KEY='your_key_here'` (Add to `.bashrc`/`.zshrc` for persistence)
    *   **Windows (CMD):** `set OPENROUTER_API_KEY=your_key_here` (Temporary)
    *   **Windows (PowerShell):** `$env:OPENROUTER_API_KEY='your_key_here'` (Temporary)
    *(Use System Properties -> Environment Variables for persistence on Windows)*

2.  **Candidate Data (`candidate_profile.json`):** Ensure this file exists in the same directory as the script and contains your accurate, up-to-date professional information in valid JSON format. **This is where you manage your profile data.**

3.  **Model Selection (Optional):**
    Inside `job_analyzer_cli.py`, you can change the `model_name` variable to specify a different model available on OpenRouter.

4.  **OpenRouter Ranking Headers (Optional):**
    Set `YOUR_SITE_URL` and `YOUR_SITE_NAME` environment variables if desired for OpenRouter leaderboard attribution.

## Usage

Run the script from your terminal, providing the full URL of the job posting as a command-line argument. **Remember to enclose the URL in quotes if it contains special characters.**

```bash
python job_analyzer_cli.py "<job_posting_url_here>"
```

**Example:**

```bash
python job_analyzer_cli.py "https://www.somecompany.com/careers/job/12345"
```

**Output:**

The script will print progress and token usage (optional) for each analysis step. The final personalized report comparing your profile to the job will be printed to the console and saved to `job_analysis_report_personalized.md`.

## Important Notes & Limitations

*   **Maintain `candidate_profile.json`:** Keep this file accurate and up-to-date for meaningful analysis results. Ensure it remains valid JSON.
*   **Web Scraping Fragility:** Scraping may fail on dynamic sites. Analysis quality depends on LLM's ability to access the URL directly if scraping fails.
*   **Model Capabilities:** The quality of URL access (Step 1) and external research (Step 4) depends on the chosen LLM.
*   **API Costs:** Monitor your OpenRouter usage and costs.
*   **Output Variability:** LLM outputs can differ slightly between runs.

## Contributing

Contributions are welcome via issues or pull requests.

## License

(Specify your license, e.g., MIT License)
