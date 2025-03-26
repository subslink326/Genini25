# Filename: job_analyzer_cli.py
# Method: Command-Line Argument (Pass URL via terminal)
# Modification: Reads candidate data from 'candidate_profile.json'

import os
import re
import requests
import argparse
import json # To load the candidate profile JSON file
from bs4 import BeautifulSoup
from openai import OpenAI

# --- Configuration (API Key, Client, Headers, Model) ---

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable not set. Please set it before running.")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

http_referer = os.getenv("YOUR_SITE_URL", "")
x_title = os.getenv("YOUR_SITE_NAME", "")
extra_headers = {}
if http_referer: extra_headers["HTTP-Referer"] = http_referer
if x_title: extra_headers["X-Title"] = x_title

model_name = "google/gemini-pro" # Or your preferred model

# --- Function to Load Candidate Data ---
def load_candidate_data(filepath="candidate_profile.json"):
    """Loads candidate data from the specified JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            candidate_data = json.load(f)
            print(f"--- Successfully loaded candidate data from {filepath} ---")
            return candidate_data
    except FileNotFoundError:
        print(f"--- ERROR: Candidate data file not found at {filepath} ---")
        print("Please ensure 'candidate_profile.json' exists in the same directory.")
        exit(1) # Exit script if profile is essential and missing
    except json.JSONDecodeError as e:
        print(f"--- ERROR: Failed to parse JSON from {filepath} ---")
        print(f"Error details: {e}")
        print("Please ensure 'candidate_profile.json' contains valid JSON.")
        exit(1)
    except Exception as e:
        print(f"--- ERROR: An unexpected error occurred loading {filepath}: {e} ---")
        exit(1)

# --- Helper Function (Web Scraping - Remains the same) ---
def scrape_job_posting_text(url):
    """ Attempts to scrape text content. Returns text or None. """
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
            'article', 'main']
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
             if body: text_content = body.get_text(separator='\n', strip=True)

        if text_content:
            text_content = re.sub(r'\n\s*\n', '\n\n', text_content).strip()
            print(f"   Scraped content length: {len(text_content)} characters.")
            if len(text_content) < 200: print("   Warning: Scraped text seems short.")
        else:
            print("   Warning: No text content scraped.")
            return None

        max_length = 15000 # Keep truncation for safety
        return text_content[:max_length] if len(text_content) > max_length else text_content
    except requests.exceptions.Timeout:
        print(f"   Error: Request timed out for {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"   Error fetching {url}: {e}")
        return None
    except Exception as e:
        print(f"   Error parsing/processing {url}: {e}")
        return None


# --- Main Workflow Function (Modified Prompts) ---
def analyze_job_application(job_url: str, candidate_profile: dict):
    """
    Executes the full job analysis workflow using the loaded candidate profile.
    """
    # This function remains largely the same as the previous "hardcoded" version,
    # as it already accepted candidate_profile as an argument.
    # The prompts will now use the data loaded from the JSON file.

    print(f"--- Starting Analysis for: {job_url} ---")
    print(f"--- Using Model: {model_name} via OpenRouter ---")
    print(f"--- Analyzing against candidate: {candidate_profile.get('personalInfo', {}).get('name', 'N/A')} ---") # Added candidate name

    scraped_text = scrape_job_posting_text(job_url)
    job_context = f"Job Posting URL: {job_url}"
    if scraped_text:
        job_context += f"\n\nRelevant Scraped Text (may be incomplete or inaccurate, use URL as primary source if accessible):\n```\n{scraped_text}\n```"
    else:
        job_context += "\n\n(Could not scrape significant text from URL. Please analyze based on the URL directly if possible. If you cannot access the URL, please state that clearly.)"

    # --- Define Prompts ---
    # (Prompts are identical to the previous version where candidate data was passed in)

    prompt_step1 = f"""
    **Task 1: Analyze Job Posting**
    Based on the provided context below (URL and potentially scraped text), please:
    1. Access the live Job Posting URL ({job_url}) if possible. Prioritize live URL info.
    2. If URL access fails, rely on scraped text.
    3. If both fail, state clearly you cannot perform the analysis.
    4. If successful, extract and summarize: Key responsibilities, Required qualifications, Preferred qualifications, Essential keywords/skills.
    **Context:**
    {job_context}
    Present output clearly under headings.
    """

    candidate_data_string = json.dumps(candidate_profile, indent=2)
    prompt_step2 = f"""
    **Task 2: Candidate Data Mapping (Against Job Requirements)**
    Compare the candidate's profile below against the job requirements analysis from Task 1 (assume success). Base analysis strictly on provided data.
    **Candidate Data:**
    ```json
    {candidate_data_string}
    ```
    **Perform comparisons:**
    *   **Qualification & Skill Mapping:** Compare candidate skills/education vs required/preferred qualifications. Identify strengths & gaps.
    *   **Experience Mapping:** Evaluate how candidate experience demonstrates job responsibilities/qualifications. Give examples.
    *   **Achievement Mapping:** Select candidate accomplishments most relevant to THIS job.
    *   **Soft Skill & Cultural Fit Assessment (Preliminary):** Assess alignment based on candidate's profile vs likely role needs/culture hints.
    Present output clearly under headings. Be objective.
    """

    prompt_step3 = f"""
    **Task 3: Ideal Profile Definition & Personalized Positioning**
    Based on Task 1 (job analysis) and Task 2 (candidate mapping):
    1.  **Ideal Candidate Profile (Employer View):** Briefly summarize the ideal profile based *only* on Task 1.
    2.  **Personalized Candidate Positioning:** Craft a compelling summary (3-5 sentences) for *this specific job*, highlighting the *provided candidate's* relevant strengths/experiences/achievements from Task 2.
    Present output under headings 'Ideal Candidate Profile (Employer View)' and 'Personalized Candidate Positioning (For This Job)'.
    """

    prompt_step4 = f"""
    **Task 4: Vet Employer and Industry**
    Perform external research on the company from URL ({job_url}). Use browsing if available. If company unknown, state that. Summarize:
    *   Company Overview (Business, Products)
    *   Reputation/Recent News (Funding, Launches, M&A)
    *   Culture Hints (Reliable sources)
    *   Industry & Competitors
    *   Market Position/Trends
    *   Career Prospects/Red Flags
    Present findings clearly (bullets/paragraphs).
    """

    prompt_step5 = f"""
    **Task 5: Synthesize Findings (Personalized)**
    Based on all previous steps:
    1.  Provide a final **Overall Fit Assessment** for the *provided candidate* against this role/company (Strong/Good/Moderate/Weak? Why?). Consider skill/experience alignment (Task 2), comparison to ideal (Task 3), company context (Task 4). Note strengths/gaps.
    2.  Summarize the *provided candidate's* **Unique Value Proposition (UVP)** for this role (1-2 sentences). What makes *this candidate* stand out for *this opportunity*?
    Present output under headings 'Overall Fit Assessment (Personalized)' and 'Unique Value Proposition (Personalized)'.
    """

    prompts_with_steps = [
        ("Step 1: Analyze Job Posting", prompt_step1),
        ("Step 2: Candidate Data Mapping", prompt_step2),
        ("Step 3: Ideal Profile & Personalized Positioning", prompt_step3),
        ("Step 4: Vet Employer and Industry", prompt_step4),
        ("Step 5: Synthesize Findings (Personalized)", prompt_step5),
    ]

    full_report = f"--- Analysis Report for Job Posting: {job_url} ---\n\n"
    full_report += f"--- Model Used: {model_name} via OpenRouter ---\n"
    full_report += f"--- Analyzing Against Candidate: {candidate_profile.get('personalInfo', {}).get('name', 'N/A')} ---\n\n" # Added name

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
    # Load candidate data from the JSON file
    candidate_data = load_candidate_data() # Default filename is "candidate_profile.json"

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Analyze a job posting URL against candidate data from candidate_profile.json using an LLM via OpenRouter.")
    parser.add_argument("job_url", help="The full URL of the job posting to analyze.")

    # Parse arguments
    args = parser.parse_args()
    target_job_url = args.job_url

    # Run the analysis, passing the loaded candidate data
    final_report = analyze_job_application(target_job_url, candidate_data)

    # --- Output the Final Report ---
    print("\n\n======== FINAL COMPREHENSIVE REPORT ========")
    print(final_report)

    # Optional: Save the report
    report_filename = "job_analysis_report_personalized.md" # Updated filename
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"\nReport saved to {report_filename}")
    except Exception as e:
        print(f"\nError saving report to file: {e}")
