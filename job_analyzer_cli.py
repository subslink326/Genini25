# Filename: job_analyzer_cli.py
# Method: Command-Line Argument (Pass URL via terminal)
# Modification: Reads candidate data from JSON, includes CONDITIONAL Steps 6 & 7.

import os
import re
import requests
import argparse
import json
from bs4 import BeautifulSoup
from openai import OpenAI

# --- Configuration (API Key, Client, Headers, Model) ---

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

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
        exit(1)
    except json.JSONDecodeError as e:
        print(f"--- ERROR: Failed to parse JSON from {filepath}: {e} ---")
        exit(1)
    except Exception as e:
        print(f"--- ERROR: An unexpected error occurred loading {filepath}: {e} ---")
        exit(1)

# --- Helper Function (Web Scraping - Remains the same) ---
def scrape_job_posting_text(url):
    """ Attempts to scrape text content. Returns text or None. """
    # (Code is identical to previous version - omitted for brevity)
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

# --- Helper Function to Call LLM ---
def call_llm(step_name, prompt_text):
    """Handles the API call and basic response processing."""
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
            return report_section, message_content # Return both formatted section and raw text
        else:
            report_section = f"**{step_name}**\n*No valid response choice received from the model.*\n\n"
            print(f"   *Warning: No valid response choice received.*")
            return report_section, "" # Return empty text on failure

    except Exception as e:
        print(f"   *Error during {step_name}: {e}*")
        report_section = f"**{step_name}**\n*An error occurred during generation: {e}*\n\n"
        return report_section, "" # Return empty text on failure


# --- Main Workflow Function (With Conditional Steps) ---
def analyze_job_application(job_url: str, candidate_profile: dict):
    """
    Executes the analysis workflow, including conditional generation steps.
    """
    print(f"--- Starting Analysis for: {job_url} ---")
    print(f"--- Using Model: {model_name} via OpenRouter ---")
    print(f"--- Analyzing against candidate: {candidate_profile.get('personalInfo', {}).get('name', 'N/A')} ---")

    scraped_text = scrape_job_posting_text(job_url)
    job_context = f"Job Posting URL: {job_url}"
    if scraped_text:
        job_context += f"\n\nRelevant Scraped Text (may be incomplete or inaccurate, use URL as primary source if accessible):\n```\n{scraped_text}\n```"
    else:
        job_context += "\n\n(Could not scrape significant text from URL. Please analyze based on the URL directly if possible. If you cannot access the URL, please state that clearly.)"

    # Store results from each step for later use
    step_results_text = {}
    full_report = f"--- Analysis Report for Job Posting: {job_url} ---\n\n"
    full_report += f"--- Model Used: {model_name} via OpenRouter ---\n"
    full_report += f"--- Analyzing Against Candidate: {candidate_profile.get('personalInfo', {}).get('name', 'N/A')} ---\n\n"

    # --- Define Core Analysis Prompts (Steps 1-5) ---
    # (Prompts 1-5 are identical to the previous version - reusing definitions)
    candidate_data_string = json.dumps(candidate_profile, indent=2)

    prompts_core = [
        ("Step 1: Analyze Job Posting", f"""**Task 1: Analyze Job Posting** ... (Full prompt text from previous version) ... **Context:**\n{job_context}\n..."""),
        ("Step 2: Candidate Data Mapping", f"""**Task 2: Candidate Data Mapping (Against Job Requirements)** ... (Full prompt text from previous version) ... **Candidate Data:**\n```json\n{candidate_data_string}\n``` ..."""),
        ("Step 3: Ideal Profile & Personalized Positioning", f"""**Task 3: Ideal Profile Definition & Personalized Positioning** ... (Full prompt text from previous version) ..."""),
        ("Step 4: Vet Employer and Industry", f"""**Task 4: Vet Employer and Industry** ... (Full prompt text referencing {job_url}) ..."""),
        ("Step 5: Synthesize Findings (Personalized)", f"""**Task 5: Synthesize Findings (Personalized)** ... (Full prompt text from previous version) ..."""),
    ] # NOTE: You need to paste the FULL prompt text from the previous version here where indicated by "..."

    # --- Execute Core Analysis Steps (1-5) ---
    for step_name, prompt_text in prompts_core:
        report_section, text_content = call_llm(step_name, prompt_text)
        full_report += report_section
        step_results_text[step_name] = text_content # Store raw text output

    # --- Conditional Generation Logic ---
    proceed_with_generation = False
    step5_output = step_results_text.get("Step 5: Synthesize Findings (Personalized)", "").lower()

    # Define keywords indicating a positive fit (case-insensitive)
    positive_fit_keywords = ["strong fit", "good fit", "excellent fit", "very good fit", "high potential", "well-suited", "strong match", "good match"]
    # Optional: Include "moderate" if desired
    # positive_fit_keywords.append("moderate fit")

    if any(keyword in step5_output for keyword in positive_fit_keywords):
        # More specific check to ensure it's part of the assessment, not just mentioned randomly
        if "overall fit assessment" in step5_output:
             # Simple check near the classification keywords
             proceed_with_generation = True
             print("\n--- Assessment indicates good fit. Proceeding with generation steps (6 & 7). ---")
        else:
             # Keyword found but maybe not in the assessment part, be cautious
             print("\n--- Positive keyword found, but context unclear. Skipping generation steps. ---")

    else:
        print("\n--- Assessment does not indicate a strong/good fit. Skipping generation steps (6 & 7). ---")


    # --- Execute Generation Steps (6 & 7) if Condition Met ---
    if proceed_with_generation:
        # Ensure we have necessary inputs from previous steps
        job_analysis_text = step_results_text.get("Step 1: Analyze Job Posting", "Job analysis data unavailable.")
        company_research_text = step_results_text.get("Step 4: Vet Employer and Industry", "Company research data unavailable.")
        positioning_text = step_results_text.get("Step 3: Ideal Profile & Personalized Positioning", "Positioning statement unavailable.")

        # --- Define Generation Prompts (Steps 6 & 7) ---
        prompt_step6 = f"""
        **Task 6: Resume Tailoring Suggestions**

        Act as a resume optimization expert. You are given the analysis of a job posting, research about the company, and the candidate's profile. Your goal is to suggest specific, actionable changes to the candidate's resume *text* to better align it with THIS specific job opportunity. Do NOT rewrite the entire resume. Focus on targeted suggestions.

        **Job Posting Analysis (from Step 1):**
        ```
        {job_analysis_text}
        ```

        **Company Research (from Step 4):**
        ```
        {company_research_text}
        ```

        **Candidate Data:**
        ```json
        {candidate_data_string}
        ```

        **Provide suggestions for:**
        1.  **Summary Enhancement:** Suggest 1-2 minor tweaks to the candidate's summary to better reflect keywords or values relevant to this specific job/company.
        2.  **Keyword Integration:** Identify 3-5 high-priority keywords from the job description that are weakly represented or missing in the candidate's profile (skills/experience) and suggest where/how they could be naturally integrated (e.g., in experience bullet points, skills list).
        3.  **Experience Bullet Point Rephrasing:** Select 2-3 existing bullet points from the candidate's work experience that are relevant to the job. Suggest alternative phrasing using strong action verbs and keywords from the job description to maximize impact for *this* role. Show the original and the suggested rephrased version.
        4.  **Skills Highlighting:** Recommend which 3-4 skills (from the candidate's list) should be most prominently highlighted or mentioned early in the application materials for this specific job.

        Present suggestions clearly under headings. Be specific and provide concrete examples of rephrasing.
        """

        prompt_step7 = f"""
        **Task 7: Cover Letter Draft Generation**

        Act as a professional writer crafting a cover letter draft for the candidate applying to the specific job identified in the inputs. The draft should be tailored, professional, and persuasive.

        **Job Posting Analysis (from Step 1 - especially responsibilities & qualifications):**
        ```
        {job_analysis_text}
        ```

        **Company Research (from Step 4 - especially company mission, values, recent news):**
        ```
        {company_research_text}
        ```

        **Candidate Data (use relevant skills, experience, accomplishments):**
        ```json
        {candidate_data_string}
        ```

        **Personalized Positioning Statement (from Step 3):**
        ```
        {positioning_text}
        ```

        **Instructions:**
        1.  **Address:** Include placeholders for recipient name/title/company address if possible, otherwise use generic greetings. Mention the specific job title being applied for.
        2.  **Introduction:** Start with a strong opening referencing the job posting and briefly state the candidate's core value proposition (drawing from the personalized positioning statement).
        3.  **Body Paragraphs (2-3):**
            *   Connect the candidate's key skills and experiences (provide 2-3 specific examples from their profile) directly to the most critical requirements mentioned in the job posting analysis. Use keywords naturally.
            *   Subtly weave in 1-2 relevant insights from the company research to demonstrate genuine interest and alignment (e.g., connect skills to a company value, mention excitement about a recent product launch). Avoid just listing facts.
        4.  **Conclusion:** Reiterate enthusiasm for the role and the company. Include a clear call to action (e.g., requesting an interview).
        5.  **Tone:** Professional, confident, and tailored.

        Generate the complete cover letter text draft. Include placeholders like [Recipient Name], [Company Address], [Your Name] where appropriate.
        """

        # Execute Step 6
        report_section_6, _ = call_llm("Step 6: Resume Tailoring Suggestions", prompt_step6)
        full_report += report_section_6

        # Execute Step 7
        report_section_7, _ = call_llm("Step 7: Cover Letter Draft Generation", prompt_step7)
        full_report += report_section_7

    else:
        # Add a note to the report if generation was skipped
        full_report += "**Steps 6 & 7 (Resume Suggestions & Cover Letter Draft):**\n\nSkipped as initial analysis did not indicate a strong/good fit.\n\n"


    print("\n--- Analysis Complete ---")
    return full_report

# --- Execute the Workflow ---
if __name__ == "__main__":
    # Load candidate data
    candidate_data = load_candidate_data()

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Analyze job posting against candidate profile (candidate_profile.json), with conditional resume/cover letter generation.")
    parser.add_argument("job_url", help="The full URL of the job posting to analyze.")
    args = parser.parse_args()
    target_job_url = args.job_url

    # Run the analysis
    final_report = analyze_job_application(target_job_url, candidate_data)

    # --- Output the Final Report ---
    print("\n\n======== FINAL COMPREHENSIVE REPORT ========")
    print(final_report)

    # Save the report
    report_filename = "job_analysis_report_conditional.md"
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"\nReport saved to {report_filename}")
    except Exception as e:
        print(f"\nError saving report to file: {e}")
