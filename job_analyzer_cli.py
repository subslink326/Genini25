# Filename: job_analyzer_cli.py
# Method: Command-Line Argument (Pass URL via terminal)

import os
import re
import requests
import argparse # Import argparse for command-line arguments
from bs4 import BeautifulSoup
from openai import OpenAI # Use the OpenAI library structure for OpenRouter

# --- Configuration ---

# Attempt to get API key from environment variable for OpenRouter
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable not set. Please set it before running.")

# Initialize the OpenAI client to point to OpenRouter
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

# Optional: Set headers for OpenRouter Ranking from environment variables
http_referer = os.getenv("YOUR_SITE_URL", "") # Optional. Your site URL
x_title = os.getenv("YOUR_SITE_NAME", "")      # Optional. Your site name

extra_headers = {}
if http_referer:
    extra_headers["HTTP-Referer"] = http_referer
if x_title:
    extra_headers["X-Title"] = x_title

# --- Choose the model ---
# Use OpenRouter's naming. Examples:
# model_name = "google/gemini-1.5-pro-latest" # Good for URL fetching, might be slower/costlier
model_name = "google/gemini-pro"             # Reliable text model
# model_name = "openai/gpt-4o"                 # Alternative powerful model
# model_name = "mistralai/mistral-large-latest" # Another alternative
# Choose the one that best suits your needs and budget on OpenRouter.

# --- Helper Function (Web Scraping) ---
def scrape_job_posting_text(url):
    """
    Attempts to scrape the main text content from a job posting URL.
    Returns the scraped text or None if scraping fails.
    """
    print(f"   Attempting to scrape: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        selectors = [
            'article.job-description', 'div.job-description', 'section.job-description',
            'div[class*="jobDescription"]', 'div[id*="jobDescription"]',
            'div.job-details-content', 'div.job-details', 'div.description',
            'article', 'main'
        ]
        text_content = ""
        for selector in selectors:
            container = soup.select_one(selector)
            if container:
                print(f"   Scraping using selector: '{selector}'")
                text_content = container.get_text(separator='\n', strip=True)
                break
        else:
             print("   Specific selectors failed, falling back to body.")
             body = soup.find('body')
             if body:
                 text_content = body.get_text(separator='\n', strip=True)

        if text_content:
            text_content = re.sub(r'\n\s*\n', '\n\n', text_content).strip()
            print(f"   Scraped content length: {len(text_content)} characters.")
            if len(text_content) < 200:
                 print("   Warning: Scraped text seems short. Content might be missing or loaded dynamically.")
        else:
            print("   Warning: No text content scraped.")
            return None

        max_length = 15000
        if len(text_content) > max_length:
            print(f"   Warning: Truncating scraped text from {len(text_content)} to {max_length} characters.")
            return text_content[:max_length]
        else:
            return text_content

    except requests.exceptions.Timeout:
        print(f"   Error: Request timed out while fetching URL {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"   Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"   Error parsing or processing URL {url}: {e}")
        return None

# --- Main Workflow Function ---
def analyze_job_application(job_url: str):
    """
    Executes the full job analysis workflow using the specified model via OpenRouter.
    """
    print(f"--- Starting Analysis for: {job_url} ---")
    print(f"--- Using Model: {model_name} via OpenRouter ---")

    scraped_text = scrape_job_posting_text(job_url)
    job_context = f"Job Posting URL: {job_url}"
    if scraped_text:
        job_context += f"\n\nRelevant Scraped Text (may be incomplete or inaccurate, use URL as primary source if accessible):\n```\n{scraped_text}\n```"
    else:
        job_context += "\n\n(Could not scrape significant text from URL. Please analyze based on the URL directly if possible. If you cannot access the URL, please state that clearly.)"

    # --- Define Prompts for Each Step ---
    prompt_step1 = f"""
    **Task 1: Analyze Job Posting**

    Based on the provided context below (URL and potentially scraped text), please:
    1.  Access the live Job Posting URL ({job_url}) if your capabilities allow. Prioritize information directly from the live URL.
    2.  If URL access fails or is not possible, rely on the provided scraped text.
    3.  If both URL access fails and no usable text is provided, state clearly that you cannot perform the analysis.
    4.  If successful, extract and summarize:
        *   Key responsibilities.
        *   Required qualifications ("must-haves").
        *   Preferred qualifications ("nice-to-haves").
        *   Essential keywords and skills mentioned.

    **Context:**
    {job_context}

    Present the output clearly under the headings: Responsibilities, Required Qualifications, Preferred Qualifications, and Keywords/Skills.
    """

    prompt_step2 = f"""
    **Task 2: Simulate Candidate Mapping (Hypothetical Candidate)**

    Based *only* on the job description details analyzed in Task 1 (assume that analysis was successful using the URL or provided text), imagine a **strong, well-suited hypothetical candidate** for this role. Do NOT invent information beyond what's plausible for someone fitting the description derived from the job posting. Perform the following analyses:
    *   **Qualification/Skill Mapping:** How do their likely qualifications/skills align strongly with the job requirements?
    *   **Experience Mapping:** How would their relevant work history plausibly demonstrate the needed skills/responsibilities?
    *   **Achievement Mapping:** What *types* of quantifiable achievements would be most relevant?
    *   **Soft Skill/Cultural Fit Mapping:** How might they demonstrate necessary soft skills and align with the likely culture?

    Present the output clearly under headings for each mapping type.
    """

    prompt_step3 = f"""
    **Task 3: Define Ideal Profile & Positioning**

    Based on the analysis of the job description (from Task 1, assuming success):
    1.  **Ideal Candidate Profile:** Describe the profile of the *employer's ideal candidate* concisely.
    2.  **Hypothetical Candidate Positioning:** Craft a compelling candidate narrative/summary (3-5 sentences) for the *hypothetical strong candidate*.

    Present the output under the headings 'Ideal Candidate Profile' and 'Hypothetical Candidate Positioning'.
    """

    prompt_step4 = f"""
    **Task 4: Vet Employer and Industry**

    Perform external research based on the company associated with the job posting URL ({job_url}). Use browsing capabilities if available. If the company cannot be determined, state that. Provide a concise summary covering:
    *   Company Overview (Business, Products/Services)
    *   Reputation/Recent News (Funding, Launches, Awards, M&A)
    *   Culture Hints (From reliable sources)
    *   Industry & Competitors
    *   Market Position/Trends
    *   Career Prospects/Red Flags

    Present findings clearly, using bullet points or brief paragraphs.
    """

    prompt_step5 = f"""
    **Task 5: Synthesize Findings**

    Based on all previous analysis steps (assuming success):
    1.  Provide a final **Overall Fit Assessment** of the hypothetical candidate for this role/company (Strong/Moderate/Weak? Why?).
    2.  Summarize the hypothetical candidate's **Unique Value Proposition (UVP)** in 1-2 sentences.

    Present the output under the headings 'Overall Fit Assessment' and 'Unique Value Proposition'.
    """

    prompts_with_steps = [
        ("Step 1: Analyze Job Posting", prompt_step1),
        ("Step 2: Simulate Candidate Mapping", prompt_step2),
        ("Step 3: Define Ideal Profile & Positioning", prompt_step3),
        ("Step 4: Vet Employer and Industry", prompt_step4),
        ("Step 5: Synthesize Findings", prompt_step5),
    ]

    full_report = f"--- Analysis Report for Job Posting: {job_url} ---\n\n"
    full_report += f"--- Model Used: {model_name} via OpenRouter ---\n\n"

    # --- Execute loop ---
    for step_name, prompt_text in prompts_with_steps:
        print(f"\n--- Executing {step_name} ---")
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text}],
                extra_headers=extra_headers if extra_headers else None,
            )

            if completion.choices:
                message_content = completion.choices[0].message.content.strip()
                finish_reason = completion.choices[0].finish_reason
                report_section = f"**{step_name}**\n\n{message_content}\n\n"
                print(f"   *Completed successfully (Finish Reason: {finish_reason}).*")
                if completion.usage:
                   print(f"   *Tokens used: Prompt={completion.usage.prompt_tokens}, Completion={completion.usage.completion_tokens}, Total={completion.usage.total_tokens}*")
            else:
                report_section = f"**{step_name}**\n*No valid response choice received from the model.*\n\n"
                print(f"   *Warning: No valid response choice received.*")

            full_report += report_section

        except Exception as e:
            print(f"   *Error during {step_name}: {e}*")
            full_report += f"**{step_name}**\n*An error occurred during generation: {e}*\n\n"

    print("\n--- Analysis Complete ---")
    return full_report

# --- Execute the Workflow ---
if __name__ == "__main__":
    # Setup argument parser for command-line input
    parser = argparse.ArgumentParser(description="Analyze a job posting URL using an LLM via OpenRouter.")
    parser.add_argument("job_url", help="The full URL of the job posting to analyze.")

    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Use the URL provided by the user via command-line argument
    target_job_url = args.job_url

    # Run the analysis using the provided URL
    final_report = analyze_job_application(target_job_url)

    # --- Output the Final Report ---
    print("\n\n======== FINAL COMPREHENSIVE REPORT ========")
    print(final_report)

    # Optional: Save the report to a file
    report_filename = "job_analysis_report_cli.md"
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"\nReport saved to {report_filename}")
    except Exception as e:
        print(f"\nError saving report to file: {e}")