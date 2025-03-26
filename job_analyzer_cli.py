# Filename: job_analyzer_cli.py
# Method: Command-Line Argument (Pass URL via terminal)
# Modification: Reads candidate data from JSON, includes CONDITIONAL Deep Dive (5.5) + Generation (6 & 7).

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

model_name = "google/gemini-pro" # Or your preferred model (gemini-1.5-pro or gpt-4o better for research)

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

        max_length = 15000
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
        # Consider using a model with stronger web browsing for research steps
        current_model = model_name
        if "Research" in step_name or "Vet" in step_name:
             # Optionally switch model for research, e.g.:
             # current_model = "google/gemini-1.5-pro-latest"
             # print(f"   (Using potentially enhanced model for research: {current_model})")
             pass # Keep same model for now, but this is where you could switch

        completion = client.chat.completions.create(
            model=current_model,
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
            return report_section, message_content # Return formatted section and raw text
        else:
            report_section = f"**{step_name}**\n*No valid response choice received from the model.*\n\n"
            print(f"   *Warning: No valid response choice received.*")
            return report_section, ""

    except Exception as e:
        print(f"   *Error during {step_name}: {e}*")
        report_section = f"**{step_name}**\n*An error occurred during generation: {e}*\n\n"
        return report_section, ""


# --- Main Workflow Function (With Conditional Steps) ---
def analyze_job_application(job_url: str, candidate_profile: dict):
    """
    Executes the analysis workflow, including conditional deep dive and generation steps.
    """
    print(f"--- Starting Analysis for: {job_url} ---")
    print(f"--- Using Primary Model: {model_name} via OpenRouter ---")
    print(f"--- Analyzing against candidate: {candidate_profile.get('personalInfo', {}).get('name', 'N/A')} ---")

    scraped_text = scrape_job_posting_text(job_url)
    job_context = f"Job Posting URL: {job_url}"
    if scraped_text:
        job_context += f"\n\nRelevant Scraped Text:\n```\n{scraped_text}\n```"
    else:
        job_context += "\n\n(Could not scrape text. Rely on URL access.)"

    step_results_text = {}
    full_report = f"--- Analysis Report for Job Posting: {job_url} ---\n\n"
    full_report += f"--- Model Used: {model_name} (May vary for research) via OpenRouter ---\n"
    full_report += f"--- Analyzing Against Candidate: {candidate_profile.get('personalInfo', {}).get('name', 'N/A')} ---\n\n"

    # --- Define Core Analysis Prompts (Steps 1-5) ---
    candidate_data_string = json.dumps(candidate_profile, indent=2)

    prompts_core = [
        ("Step 1: Analyze Job Posting", f"""**Task 1: Analyze Job Posting** ... (Full prompt text from previous version) ... **Context:**\n{job_context}\n..."""),
        ("Step 2: Candidate Data Mapping", f"""**Task 2: Candidate Data Mapping (Against Job Requirements)** ... (Full prompt text) ... **Candidate Data:**\n```json\n{candidate_data_string}\n``` ..."""),
        ("Step 3: Ideal Profile & Personalized Positioning", f"""**Task 3: Ideal Profile Definition & Personalized Positioning** ... (Full prompt text) ..."""),
        ("Step 4: Initial Company Vetting", f"""**Task 4: Initial Company Vetting** Perform **brief** research on the company from URL ({job_url}). Use browsing. If unknown, state that. Summarize concisely: Company Overview (Business), General Reputation/Recent Highlight, Industry & Main Competitors, Potential Red Flags (briefly). This is a preliminary check."""), # Made Step 4 briefer
        ("Step 5: Synthesize Findings (Personalized)", f"""**Task 5: Synthesize Findings (Personalized)** ... (Full prompt text) ...""")
    ] # NOTE: Paste FULL prompt text where indicated by "..."

    # --- Execute Core Analysis Steps (1-5) ---
    for step_name, prompt_text in prompts_core:
        report_section, text_content = call_llm(step_name, prompt_text)
        full_report += report_section
        step_results_text[step_name] = text_content

    # --- Conditional Generation Logic ---
    proceed_with_generation = False
    step5_output = step_results_text.get("Step 5: Synthesize Findings (Personalized)", "").lower()
    positive_fit_keywords = ["strong fit", "good fit", "excellent fit", "very good fit", "high potential", "well-suited", "strong match", "good match", "moderate fit"] # Added moderate

    if any(keyword in step5_output for keyword in positive_fit_keywords):
        if "overall fit assessment" in step5_output:
             proceed_with_generation = True
             print("\n--- Assessment indicates potential fit. Proceeding with deep dive and generation steps (5.5, 6, 7). ---")
        else:
             print("\n--- Positive keyword found, but context unclear in Step 5. Skipping deep dive/generation. ---")
    else:
        print("\n--- Assessment does not indicate a promising fit. Skipping deep dive and generation steps (5.5, 6, 7). ---")


    # --- Execute Deep Dive and Generation Steps if Condition Met ---
    if proceed_with_generation:
        # --- Define and Execute NEW Step 5.5 ---
        prompt_step5_5 = f"""
        **Task 5.5: Deep Dive Company Research**

        Perform a **comprehensive deep dive research analysis** on the company associated with the job posting URL ({job_url}). Utilize your web browsing capabilities extensively. If the company cannot be reliably determined, state that. Structure the report clearly with the following sections:

        1.  **Company Overview:** Full name, headquarters, year founded, brief mission statement.
        2.  **History & Founding Story:** Key milestones, founders (if notable), evolution over time.
        3.  **Key Leadership:** CEO, relevant C-suite executives (e.g., Head of Marketing if relevant to job), board members if significant and public. Include names and titles; brief public bio summary if available.
        4.  **Products & Services:** Detailed breakdown of main offerings, flagship products/services, key features, and target customer segments.
        5.  **Business Model:** How does the company primarily make money? (e.g., SaaS subscriptions, advertising, direct sales, etc.)
        6.  **Market & Competitors:** In-depth analysis of the industry landscape. Identify primary and secondary competitors. What are the company's key differentiators or competitive advantages? What are its weaknesses?
        7.  **Recent News & Developments (Last 6-12 months):** Significant funding rounds, acquisitions, major product launches, strategic partnerships, significant press releases, notable awards, or any major controversies.
        8.  **Financial Health (Public Info):** If public (or recent reliable reports), mention revenue trends, valuation, funding status, profitability status. If private, note that details may be limited.
        9.  **Mission, Vision & Values:** Explicitly stated mission, vision, and core values (usually found on 'About Us' or 'Careers' pages).
        10. **Culture Insights:** Synthesize information about the work environment, employee reviews (cite source type like 'Glassdoor mentions...', 'Company career page emphasizes...'), DE&I initiatives. Provide a balanced view if possible.
        11. **Potential Interview Talking Points:** Based on the research, suggest 2-3 specific topics the candidate could discuss to show interest and knowledge (e.g., "Ask about the strategy behind the recent [Product X] launch," "Discuss how they are addressing [Industry Challenge Y]," "Mention alignment with their stated value of [Value Z]").

        Present the findings clearly under numbered headings corresponding to the sections above. Be thorough and cite information implicitly through the details provided.
        """
        report_section_5_5, deep_dive_research_text = call_llm("Step 5.5: Deep Dive Company Research", prompt_step5_5)
        full_report += report_section_5_5
        step_results_text["Step 5.5: Deep Dive Company Research"] = deep_dive_research_text # Store result

        # --- Define and Execute Step 6 (Resume Suggestions) ---
        # Using the deep_dive_research_text now
        prompt_step6 = f"""
        **Task 6: Resume Tailoring Suggestions**
        Act as a resume optimization expert. Use the job analysis (Step 1), the **detailed company research (Step 5.5)**, and the candidate's profile to suggest specific, actionable changes to the candidate's resume *text* for THIS job opportunity. Focus on targeted content suggestions.

        **Job Posting Analysis (from Step 1):**
        ```
        {step_results_text.get("Step 1: Analyze Job Posting", "Job analysis unavailable.")}
        ```

        **Deep Dive Company Research (from Step 5.5):**
        ```
        {deep_dive_research_text} 
        ```

        **Candidate Data:**
        ```json
        {candidate_data_string}
        ```

        **Provide suggestions for:**
        1.  **Summary Enhancement:** Minor tweaks to align summary with keywords or company values/mission from deep dive research.
        2.  **Keyword Integration:** Identify 3-5 high-priority job keywords missing/weak in candidate profile; suggest integration points.
        3.  **Experience Bullet Point Rephrasing:** Select 2-3 relevant candidate bullet points; suggest rephrasing using job keywords and strong action verbs. Show original vs. suggestion.
        4.  **Skills Highlighting:** Recommend 3-4 candidate skills most crucial for this job/company.
        Present suggestions clearly under headings with specific examples.
        """
        report_section_6, _ = call_llm("Step 6: Resume Tailoring Suggestions", prompt_step6)
        full_report += report_section_6

        # --- Define and Execute Step 7 (Cover Letter Draft) ---
        # Using the deep_dive_research_text now
        prompt_step7 = f"""
        **Task 7: Cover Letter Draft Generation**
        Act as a professional writer crafting a tailored cover letter draft for the candidate applying to the specific job.

        **Job Posting Analysis (from Step 1):**
        ```
        {step_results_text.get("Step 1: Analyze Job Posting", "Job analysis unavailable.")}
        ```

        **Deep Dive Company Research (from Step 5.5 - use mission, values, news, products):** 
        ```
        {deep_dive_research_text}
        ```

        **Candidate Data (use relevant skills, experience, accomplishments):**
        ```json
        {candidate_data_string}
        ```

        **Personalized Positioning Statement (from Step 3):**
        ```
        {step_results_text.get("Step 3: Ideal Profile & Personalized Positioning", "Positioning statement unavailable.")}
        ```

        **Instructions:**
        1.  **Address:** Placeholders for recipient/company. Mention specific job title.
        2.  **Introduction:** Strong opening referencing job, state core value prop (use positioning statement).
        3.  **Body Paragraphs (2-3):** Connect candidate's skills/experience (2-3 specific examples) to job requirements. Weave in 1-2 relevant insights from the **deep dive company research** (e.g., align experience with a company value, mention excitement about a relevant product/initiative) to show genuine interest.
        4.  **Conclusion:** Reiterate enthusiasm. Clear call to action.
        5.  **Tone:** Professional, confident, tailored.

        Generate the complete cover letter text draft with placeholders like [Recipient Name], [Company Address], [Your Name].
        """
        report_section_7, _ = call_llm("Step 7: Cover Letter Draft Generation", prompt_step7)
        full_report += report_section_7

    else:
        # Update skipped message
        full_report += "**Steps 5.5, 6, & 7 (Deep Dive Research, Resume Suggestions & Cover Letter Draft):**\n\nSkipped as initial analysis did not indicate a promising fit.\n\n"

    print("\n--- Analysis Complete ---")
    return full_report

# --- Execute the Workflow ---
if __name__ == "__main__":
    candidate_data = load_candidate_data()
    parser = argparse.ArgumentParser(description="Analyze job posting against candidate profile (candidate_profile.json), with conditional deep dive research and resume/cover letter generation.")
    parser.add_argument("job_url", help="The full URL of the job posting to analyze.")
    args = parser.parse_args()
    target_job_url = args.job_url

    final_report = analyze_job_application(target_job_url, candidate_data)

    print("\n\n======== FINAL COMPREHENSIVE REPORT ========")
    print(final_report)

    report_filename = "job_analysis_report_conditional_deepdive.md" # Updated filename
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"\nReport saved to {report_filename}")
    except Exception as e:
        print(f"\nError saving report to file: {e}")
