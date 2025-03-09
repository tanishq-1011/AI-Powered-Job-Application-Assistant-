import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Set up OpenAI API
openai.api_key = "your-api-key"

# 1. Perception Agent: Scrapes job listings
class PerceptionAgent:
    def __init__(self, job_title, location):
        self.job_title = job_title
        self.location = location

    def scrape_jobs(self):
        url = f"https://www.indeed.com/jobs?q={self.job_title.replace(' ', '+')}&l={self.location.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        jobs = []
        for job in soup.find_all("div", class_="job_seen_beacon"):
            title = job.find("h2").text.strip() if job.find("h2") else "N/A"
            company = job.find("span", class_="companyName").text.strip() if job.find("span", class_="companyName") else "N/A"
            desc = job.find("div", class_="job-snippet").text.strip() if job.find("div", class_="job-snippet") else "N/A"
            jobs.append({"title": title, "company": company, "description": desc})

        return pd.DataFrame(jobs)


# 2. Reasoning Agent: Matches job descriptions to resume
class ReasoningAgent:
    def __init__(self, resume_text):
        self.resume_text = resume_text

    def rank_jobs(self, jobs_df):
        vectorizer = TfidfVectorizer(stop_words="english")
        job_descriptions = jobs_df["description"].tolist()
        all_text = [self.resume_text] + job_descriptions

        tfidf_matrix = vectorizer.fit_transform(all_text)
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        jobs_df["match_score"] = similarity_scores

        return jobs_df.sort_values("match_score", ascending=False)


# 3. Action Agent: Tailors resume and writes cover letters
class ActionAgent:
    def __init__(self, resume_text):
        self.resume_text = resume_text

    def tailor_resume(self, job_description):
        prompt = f"""
        You are an expert resume writer. Here is my current resume:

        {self.resume_text}

        Here is the job description I am applying for:

        {job_description}

        Please tailor my resume to align with this job description while keeping it ATS-friendly.
        """
        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=500)
        return response["choices"][0]["text"].strip()

    def generate_cover_letter(self, job_title, company, job_description):
        prompt = f"""
        You are a professional cover letter writer. Write a personalized cover letter for the job title "{job_title}" at "{company}".

        Job description:
        {job_description}

        Make sure the cover letter highlights my qualifications and enthusiasm for this role.
        """
        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=500)
        return response["choices"][0]["text"].strip()


# 4. Orchestrator: Ties everything together
class Orchestrator:
    def __init__(self, job_title, location, resume_text):
        self.perception_agent = PerceptionAgent(job_title, location)
        self.reasoning_agent = ReasoningAgent(resume_text)
        self.action_agent = ActionAgent(resume_text)

    def run(self):
        print("Scraping job listings...")
        jobs_df = self.perception_agent.scrape_jobs()

        print("Ranking job matches...")
        ranked_jobs = self.reasoning_agent.rank_jobs(jobs_df)

        top_job = ranked_jobs.iloc[0]
        print(f"\nTop Job Match: {top_job['title']} at {top_job['company']}\n")

        tailored_resume = self.action_agent.tailor_resume(top_job["description"])
        print(f"\nTailored Resume:\n{tailored_resume}\n")

        cover_letter = self.action_agent.generate_cover_letter(top_job["title"], top_job["company"], top_job["description"])
        print(f"\nGenerated Cover Letter:\n{cover_letter}\n")


# 5. Run the agentic system
if __name__ == "__main__":
    JOB_TITLE = "Data Scientist"
    LOCATION = "Remote"
    RESUME_TEXT = """
    Experienced Data Scientist with expertise in machine learning, NLP, and data analysis.
    Skilled in Python, TensorFlow, PyTorch, and deploying models into production.
    """
    
    orchestrator = Orchestrator(JOB_TITLE, LOCATION, RESUME_TEXT)
    orchestrator.run()
