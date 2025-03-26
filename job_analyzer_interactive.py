# Filename: job_analyzer_interactive.py
# Method: Interactive Input (Prompts user for URL)
# Feature: Reads candidate data from JSON, includes CONDITIONAL Deep Dive (5.5) + Generation (6 & 7).
# Version: Full prompts included.

import os
import re
import requests
# import argparse # Not needed for interactive version
import json # To load the candidate profile JSON file
from bs4 import BeautifulSoup
from openai import OpenAI # Use the OpenAI library structure for OpenRouter

# --- Configuration (API Key, Client, Headers, Model) ---

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
if http_referer: extra_headers["HTTP-Referer"] = http_referer
if x_title: extra_headers["X-Title"] = x_title

# --- Choose the model ---
# Use OpenRouter's naming. Examples:
# model_name = "google/gemini-1.5-pro-latest" # Good for URL fetching, might be slower/costlier
# model_name = "google/gemini-pro"             # Reliable text model
model_name = "google/gemini-2.5-pro-exp-03-25:free" # <<< UPDATED MODEL
# model_name = "openai/gpt-4o"                 # Alternative powerful model
# model_name = "mistralai/mistral-large-latest" # Another alternative
# Choose the one that best suits your needs and budget on OpenRouter.

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
        selectors = [ # List of CSS selectors to try
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
                break # Use the first successful selector
        else:
             print("   Specific selectors failed, falling back to body.")
             body = soup.find('body')
             if body: text_content = body.get_text(separator='\n', strip=True)

        if text_content:
            text_content = re.sub(r'\n\s*\n', '\n\n', text_content).strip() # Clean up extra blank lines
            print(f"   Scraped content length: {len(text_content)} characters.")
            if len(text_content) < 200: print("   Warning: Scraped text seems short.")
        else:
            print("   Warning: No text content scraped.")
            return None

        max_length = 15000 # Limit length to avoid excessive token usage
        if len(text_content) > max_length:
             print(f"   Warning: Truncating scraped text to {max_length} characters.")
             return text_content[:max_length]
        else:
             return text_content
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
    """Handles the API call to OpenRouter and basic response processing."""
    print(f"\n--- Executing {step_name} ---")
    try:
        # Use the globally defined model name directly
        current_model = model_name
        # <<< REMOVED CONDITIONAL MODEL SWITCHING BLOCK >>>

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
    print(f"--- Using Model: {model_name} via OpenRouter ---") # Updated print statement
    print(f"--- Analyzing against candidate: {candidate_profile.get('personalInfo', {}).get('name', 'N/A')} ---")

    scraped_text = scrape_job_posting_text(job_url)
    job_context = f"Job Posting URL: {job_url}"
    if scraped_text:
        job_context += f"\n\nRelevant Scraped Text:\n```\n{scraped_text}\n```"
    else:
        job_context += "\n\n(Could not scrape text. Rely on URL access.)"

    step_results_text = {} # Store raw text output of each step
    full_report = f"--- Analysis Report for Job Posting: {job_url} ---\n\n"
    full_report += f"--- Model Used: {model_name} via OpenRouter ---\n" # Updated print statement
    full_report += f"--- Analyzing Against Candidate: {candidate_profile.get('personalInfo', {}).get('name', 'N/A')} ---\n\n"

    # --- Define Core Analysis Prompts (Steps 1-5) ---
    candidate_data_string = json.dumps(candidate_profile, indent=2)

    # ** FULL PROMPT DEFINITIONS START HERE **

    prompt_step1 = f"""
    **Task 1: Analyze Job Posting**

    Based on the provided context below (URL and potentially scraped text), please:
    1.  Access the live Job Posting URL ({job_url}) if your capabilities allow. Prioritize information directly from the live URL.
    2.  If URL access fails or is not possible, rely on the provided scraped text.
    3.  If both URL access fails and no usable text is provided, state clearly that you cannot perform the analysis.
    4.  If successful, extract and summarize:
        * Key responsibilities.
        * Required qualifications ("must-haves").
        * Preferred qualifications ("nice-to-haves").
        * Essential keywords and skills mentioned in the posting.

    **Context:**
    {job_context}

    Present the output clearly under the headings: Responsibilities, Required Qualifications, Preferred Qualifications, and Keywords/Skills.
    """

    prompt_step2 = f"""
    **Task 2: Candidate Data Mapping (Against Job Requirements)**

    You are given the job requirements analysis from Task 1 (assume it was successful) and the profile data of a specific candidate below.
    **Your task is to compare the candidate's profile against the job requirements.** Do NOT invent information about the candidate; base your analysis strictly on the provided data.

    **Candidate Data:**
    ```json
    {candidate_data_string}
    ```

    **Perform the following comparisons:**
    * **Qualification & Skill Mapping:** Compare the candidate's skills (technical, core, soft) and education against the required and preferred qualifications from the job posting. Identify key strengths (strong matches) and potential gaps.
    * **Experience Mapping:** Evaluate how the candidate's listed work experience (roles, responsibilities) demonstrates the key responsibilities and required qualifications mentioned in the job posting. Provide specific examples from the candidate's experience where possible.
    * **Achievement Mapping:** Select the candidate's accomplishments (from their profile) that are MOST relevant and impactful for THIS specific job posting's responsibilities and requirements.
    * **Soft Skill & Cultural Fit Assessment (Preliminary):** Based on the candidate's listed soft skills and general professional summary, assess potential alignment with the soft skills likely needed for the role (based on Task 1) and any cultural hints inferable from the job posting/company type. Note alignment points.

    Present the output clearly under headings for each mapping/assessment type (e.g., Qualification/Skill Analysis, Experience Relevance, Relevant Achievements, Soft Skill/Culture Alignment). Be objective in identifying both strengths and potential gaps.
    """

    prompt_step3 = f"""
    **Task 3: Ideal Profile Definition & Personalized Positioning**

    Based on the analysis of the job description (from Task 1, assuming success) and the candidate mapping (Task 2):
    1.  **Ideal Candidate Profile (Employer View):** Briefly restate or summarize the profile of the *employer's ideal candidate* based *only* on the job posting analysis from Task 1.
    2.  **Personalized Candidate Positioning:** Craft a compelling candidate narrative or professional summary (3-5 sentences) *specifically tailored for this job*. This summary should highlight the *provided candidate's* most relevant strengths, experiences, and achievements (identified in Task 2) that directly address the requirements of *this specific job posting*.

    Present the output under the headings 'Ideal Candidate Profile (Employer View)' and 'Personalized Candidate Positioning (For This Job)'.
    """

    prompt_step4 = f"""
    **Task 4: Initial Company Vetting**

    Perform **brief** research on the company associated with the job posting URL ({job_url}). Use your web Browse capabilities if available. If the company cannot be reliably determined, state that. Summarize concisely covering:
    * Company Overview (Business, Main Products/Services)
    * General Reputation / Recent Highlight (e.g., major funding, award, notable product launch)
    * Industry & Main Competitors
    * Potential Red Flags (briefly, if any obvious ones surface)

    This is intended as a preliminary check only. Present findings clearly.
    """

    prompt_step5 = f"""
    **Task 5: Synthesize Findings (Personalized)**

    Based on all previous analysis steps (Job Req Analysis, Candidate Mapping, Ideal Profile, Positioning, Initial Company Vetting - assuming success):
    1.  Provide a final **Overall Fit Assessment** for the *provided candidate* against this specific role and company. Classify the fit (e.g., Strong, Good, Moderate, Weak) and briefly explain why, considering the alignment of skills/experience (Task 2), how they compare to the ideal profile (Task 3), and the company context (Task 4). Mention key strengths and any significant gaps identified.
    2.  Summarize the *provided candidate's* **Unique Value Proposition (UVP)** for this role in 1-2 sentences. What makes *this specific candidate* stand out for *this specific opportunity*, based on your analysis?

    Present the output under the headings 'Overall Fit Assessment (Personalized)' and 'Unique Value Proposition (Personalized)'.
    """

    prompts_core = [
        ("Step 1: Analyze Job Posting", prompt_step1),
        ("Step 2: Candidate Data Mapping", prompt_step2),
        ("Step 3: Ideal Profile & Personalized Positioning", prompt_step3),
        ("Step 4: Initial Company Vetting", prompt_step4),
        ("Step 5: Synthesize Findings (Personalized)", prompt_step5),
    ]

    # --- Execute Core Analysis Steps (1-5) ---
    for step_name, prompt_text in prompts_core:
        report_section, text_content = call_llm(step_name, prompt_text)
        full_report += report_section
        step_results_text[step_name] = text_content # Store raw text

    # --- Conditional Generation Logic ---
    proceed_with_generation = False
    step5_output = step_results_text.get("Step 5: Synthesize Findings (Personalized)", "").lower()
    # Keywords indicating a worthwhile fit for proceeding
    positive_fit_keywords = ["strong fit", "good fit", "excellent fit", "very good fit", "high potential", "well-suited", "strong match", "good match", "moderate fit", "promising fit"]

    # Check if Step 5 output contains positive keywords within the assessment context
    if "overall fit assessment" in step5_output:
        assessment_section = step5_output.split("overall fit assessment")[1] # Rough check in the relevant section
        if any(keyword in assessment_section for keyword in positive_fit_keywords):
             proceed_with_generation = True
             print("\n--- Assessment indicates potential fit. Proceeding with deep dive and generation steps (5.5, 6, 7). ---")
        else:
             print("\n--- Assessment does not indicate promising fit. Skipping deep dive/generation. ---")
    else:
         # Fallback check if section title isn't perfect
         if any(keyword in step5_output for keyword in positive_fit_keywords):
             print("\n--- Positive keyword found, but 'Overall Fit Assessment' context unclear. Proceeding cautiously with deep dive/generation. ---")
             proceed_with_generation = True
         else:
             print("\n--- Assessment does not indicate promising fit. Skipping deep dive/generation. ---")


    # --- Execute Deep Dive and Generation Steps if Condition Met ---
    if proceed_with_generation:
        # --- Define and Execute NEW Step 5.5 ---
        prompt_step5_5 = f"""
        **Task 5.5: Deep Dive Company Research**

        Perform a **comprehensive deep dive research analysis** on the company associated with the job posting URL ({job_url}). Utilize your web Browse capabilities extensively. If the company cannot be reliably determined, state that. Structure the report clearly with the following sections:

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
        prompt_step6 = f"""
        **Task 6: Resume Tailoring Suggestions**

        Act as a resume optimization expert. You are given the analysis of a job posting (Step 1), the **detailed company research (Step 5.5)**, and the candidate's profile. Your goal is to suggest specific, actionable changes to the candidate's resume *text* to better align it with THIS specific job opportunity. Do NOT rewrite the entire resume. Focus on targeted content suggestions.

        **Job Posting Analysis (from Step 1):**
        ```
        {step_results_text.get("Step 1: Analyze Job Posting", "Job analysis data unavailable.")}
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
        1.  **Summary Enhancement:** Suggest 1-2 minor tweaks to the candidate's summary to better reflect keywords or values relevant to this specific job/company based on the deep dive research.
        2.  **Keyword Integration:** Identify 3-5 high-priority keywords from the job description that are weakly represented or missing in the candidate's profile (skills/experience) and suggest where/how they could be naturally integrated (e.g., in experience bullet points, skills list).
        3.  **Experience Bullet Point Rephrasing:** Select 2-3 existing bullet points from the candidate's work experience that are relevant to the job. Suggest alternative phrasing using strong action verbs and keywords from the job description to maximize impact for *this* role. Show the original and the suggested rephrased version.
        4.  **Skills Highlighting:** Recommend which 3-4 skills (from the candidate's list) should be most prominently highlighted or mentioned early in the application materials for this specific job.

        Present suggestions clearly under headings. Be specific and provide concrete examples of rephrasing.
        """
        report_section_6, _ = call_llm("Step 6: Resume Tailoring Suggestions", prompt_step6)
        full_report += report_section_6

        # --- Define and Execute Step 7 (Cover Letter Draft) ---
        prompt_step7 = f"""
        **Task 7: Cover Letter Draft Generation**

        Act as a professional writer crafting a tailored cover letter draft for the candidate applying to the specific job identified in the inputs. The draft should be tailored, professional, and persuasive.

        **Job Posting Analysis (from Step 1 - especially responsibilities & qualifications):**
        ```
        {step_results_text.get("Step 1: Analyze Job Posting", "Job analysis data unavailable.")}
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
        1.  **Address:** Include placeholders for recipient name/title/company address if possible, otherwise use generic greetings. Mention the specific job title being applied for.
        2.  **Introduction:** Start with a strong opening referencing the job posting and briefly state the candidate's core value proposition (drawing from the personalized positioning statement).
        3.  **Body Paragraphs (2-3):**
            * Connect the candidate's key skills and experiences (provide 2-3 specific examples from their profile) directly to the most critical requirements mentioned in the job posting analysis. Use keywords naturally.
            * Subtly weave in 1-2 relevant insights from the **deep dive company research** to demonstrate genuine interest and alignment (e.g., connect skills to a company value, mention excitement about a recent product launch or company direction). Avoid just listing facts.
        4.  **Conclusion:** Reiterate enthusiasm for the role and the company. Include a clear call to action (e.g., requesting an interview).
        5.  **Tone:** Professional, confident, and tailored.

        Generate the complete cover letter text draft. Include placeholders like [Recipient Name], [Company Address], [Your Name] where appropriate.
        """
        report_section_7, _ = call_llm("Step 7: Cover Letter Draft Generation", prompt_step7)
        full_report += report_section_7

    else:
        # Add a note to the report if generation was skipped
        full_report += "**Steps 5.5, 6, & 7 (Deep Dive Research, Resume Suggestions & Cover Letter Draft):**\n\nSkipped as initial analysis did not indicate a promising fit.\n\n"


    print("\n--- Analysis Complete ---")
    return full_report

# --- Execute the Workflow ---
if __name__ == "__main__":
    # Load candidate data from the JSON file
    candidate_data = load_candidate_data()

    # Prompt user for the URL interactively
    while True:
        target_job_url = input("Please enter the full URL of the job posting: ").strip()
        if target_job_url:
            break
        else:
            print("URL cannot be empty. Please try again.")

    # Run the analysis, passing the loaded candidate data
    final_report = analyze_job_application(target_job_url, candidate_data)

    # --- Output the Final Report ---
    print("\n\n======== FINAL COMPREHENSIVE REPORT ========")
    print(final_report)

    # Optional: Save the report
    report_filename = "job_analysis_report_interactive_deepdive.md" # Updated filename
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"\nReport saved to {report_filename}")
    except Exception as e:
        print(f"\nError saving report to file: {e}")

# ** END OF FILE **
