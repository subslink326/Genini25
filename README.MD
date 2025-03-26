# Job Posting Analyzer (via OpenRouter)

This Python script automates the analysis of online job postings using Large Language Models (LLMs) accessed via the OpenRouter.ai API. It takes a job posting URL as a command-line argument, performs a multi-step analysis based on the job description, vets the company and industry context, and generates a comprehensive report summarizing its findings.

## Features

*   **Command-Line Interface:** Accepts the target job posting URL directly as a command-line argument.
*   **Web Scraping (Best-Effort):** Attempts to scrape text content directly from the provided job posting URL.
*   **Job Description Analysis:** Extracts key responsibilities, required/preferred qualifications, and essential keywords/skills.
*   **Hypothetical Candidate Mapping:** Simulates how a strong, hypothetical candidate might align with the role's requirements, experience, achievements, and culture.
*   **Ideal Profile Definition:** Describes the employer's likely ideal candidate profile based on the posting.
*   **Candidate Positioning:** Crafts a sample professional summary for the hypothetical candidate.
*   **Company & Industry Vetting:** Performs external research (via LLM capabilities) on the company's business, reputation, culture, industry position, competitors, and trends.
*   **Synthesis & Fit Assessment:** Provides an overall assessment of the hypothetical candidate's fit and unique value proposition.
*   **Model Agnostic (via OpenRouter):** Configurable to use various LLMs available through OpenRouter (e.g., Gemini, GPT models, Mistral, etc.).
*   **Formatted Output:** Generates a structured report printed to the console and optionally saved to a Markdown file.

## Requirements

*   Python 3.7+
*   An account with [OpenRouter.ai](https://openrouter.ai/) to obtain an API key.

## Installation

1.  **Clone the repository (or download the script):**
    ```bash
    # git clone <repository_url> # If applicable
    # cd <repository_directory>
    ```
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

## Configuration

1.  **API Key:** This script requires your OpenRouter API key. **Do not hardcode your API key directly into the script.** Set it as an environment variable named `OPENROUTER_API_KEY`.

    *   **Linux/macOS:**
        ```bash
        export OPENROUTER_API_KEY='your_openrouter_api_key_here'
        ```
        *(Add this line to your `.bashrc`, `.zshrc`, or shell profile file for persistence)*

    *   **Windows (Command Prompt):**
        ```bash
        set OPENROUTER_API_KEY=your_openrouter_api_key_here
        ```
        *(This is usually temporary for the current session)*

    *   **Windows (PowerShell):**
        ```bash
        $env:OPENROUTER_API_KEY='your_openrouter_api_key_here'
        ```
        *(This is usually temporary for the current session)*

        *(For persistent environment variables on Windows, use the System Properties dialog -> Environment Variables)*

2.  **Model Selection (Optional):**
    Inside the script (`job_analyzer_cli.py`), you can change the `model_name` variable to specify which model available on OpenRouter you wish to use (e.g., `"google/gemini-pro"`, `"openai/gpt-4o"`, `"mistralai/mistral-large-latest"`). Check OpenRouter documentation for available model identifiers.

3.  **OpenRouter Ranking Headers (Optional):**
    If you want your API calls attributed to your application on the OpenRouter leaderboard, set the following environment variables:
    *   `YOUR_SITE_URL`: e.g., `https://myjobanalyzer.com`
    *   `YOUR_SITE_NAME`: e.g., `My Job Analyzer`
    The script will automatically include these as `HTTP-Referer` and `X-Title` headers if the environment variables are set.

## Usage

Run the script from your terminal, providing the full URL of the job posting as a command-line argument. **Remember to enclose the URL in quotes if it contains special characters like `&` or `?`.**

```bash
python job_analyzer_cli.py "<job_posting_url_here>"
```

**Example:**

```bash
python job_analyzer_cli.py "https://www.dialpad.com/careers/jobs/apply/executive-assistant,-marketing/austin-us"
```

**Output:**

The script will print progress for each analysis step to the console. The final comprehensive report will be printed at the end. By default, the report is also saved to a file named `job_analysis_report_cli.md` in the same directory where the script is run.

## Important Notes & Limitations

*   **Web Scraping Fragility:** The initial web scraping step is best-effort and may fail on websites that load content dynamically with JavaScript or have complex/non-standard HTML structures. The script attempts common selectors but cannot guarantee success. If scraping fails, the analysis relies heavily on the LLM's ability to access the URL directly.
*   **Model Capabilities:** The quality and success of the analysis, especially accessing live URLs (Step 1) and external research (Step 4), depend heavily on the capabilities of the specific LLM selected via `model_name`. Choose models known for tool use or web browsing (like `google/gemini-1.5-pro-latest` or `openai/gpt-4o`) for best results in these areas.
*   **API Costs:** Using the OpenRouter API incurs costs based on the chosen model and the number of tokens processed (input + output). Monitor your usage and costs on the OpenRouter platform. The script includes optional token usage printing for each step.
*   **Output Variability:** LLM outputs can vary even with the same prompt. Rerunning the script might yield slightly different results.
*   **Hypothetical Candidate:** The analysis related to the "hypothetical candidate" is purely based on inferring a strong match from the job description. It does not reflect any real individual's profile.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is likely under the MIT License (or specify your chosen license).