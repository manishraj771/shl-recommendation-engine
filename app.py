import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests
import logging
import re
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MOCK_API_URL = "https://my-json-server.typicode.com/<your-username>/shl-mock-api/assessments"

# def fetch_assessments() -> List[Dict]:
#     try:
#         response = requests.get(MOCK_API_URL, timeout=5)
#         response.raise_for_status()
#         return response.json()
#     except requests.RequestException as e:
#         logger.warning(f"Failed to fetch from API: {e}. Using local mock data.")
#         with open('data/assessments.json', 'r') as f:
#             return json.load(f)
def fetch_assessments() -> List[Dict]:
    try:
        with open('data/assessments.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Local mock data not found: {e}")
        return []

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_tfidf_matrix(texts: List[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    return vectorizer.fit_transform(texts).toarray(), vectorizer

def recommend_assessments(job_description: str, assessments: List[Dict], top_k: int = 3) -> List[Dict]:
    job_description = preprocess_text(job_description)
    assessment_texts = [
        preprocess_text(f"{a['title']} {a['description']} {' '.join(a['competencies'])} {a['target_role']}")
        for a in assessments
    ]
    all_texts = [job_description] + assessment_texts
    tfidf_matrix, vectorizer = compute_tfidf_matrix(all_texts)
    job_vector = tfidf_matrix[0:1]
    assessment_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(job_vector, assessment_vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [
        {
            "id": assessments[i]["id"],
            "title": assessments[i]["title"],
            "description": assessments[i]["description"],
            "similarity_score": float(similarities[i])
        }
        for i in top_indices
    ]

def evaluate_recommendations(test_cases: List[Dict], assessments: List[Dict], k: int = 3) -> Dict[str, float]:
    precisions, recalls, maps = [], [], []
    for test in test_cases:
        job_desc = test["job_description"]
        expected_ids = test["expected_assessment_ids"]
        recommendations = recommend_assessments(job_desc, assessments, top_k=k)
        recommended_ids = [rec["id"] for rec in recommendations]
        relevant = set(expected_ids)
        retrieved = set(recommended_ids)
        precision = len(relevant & retrieved) / len(retrieved) if retrieved else 0
        precisions.append(precision)
        recall = len(relevant & retrieved) / len(relevant) if relevant else 0
        recalls.append(recall)
        ap = 0
        relevant_count = 0
        for i, rec_id in enumerate(recommended_ids, 1):
            if rec_id in relevant:
                relevant_count += 1
                ap += relevant_count / i
        ap = ap / len(relevant) if relevant else 0
        maps.append(ap)
    return {
        "precision@k": np.mean(precisions),
        "recall@k": np.mean(recalls),
        "MAP@k": np.mean(maps)
    }

def main():
    st.title("SHL Assessment Recommendation Engine")
    st.write("Enter a job description to get recommended SHL assessments.")
    assessments = fetch_assessments()
    job_description = st.text_area("Job Description", height=150)
    top_k = st.slider("Number of Recommendations", min_value=1, max_value=5, value=3)
    if st.button("Recommend Assessments"):
        if not job_description.strip():
            st.error("Please enter a job description.")
        else:
            recommendations = recommend_assessments(job_description, assessments, top_k)
            st.subheader("Recommended Assessments")
            for rec in recommendations:
                st.write(f"**{rec['title']}** (Similarity: {rec['similarity_score']:.2f})")
                st.write(rec['description'])
                st.write("---")
    st.subheader("Evaluation Metrics")
    test_cases = [
        {
            "job_description": "Hiring a software engineer proficient in Java and teamwork.",
            "expected_assessment_ids": ["4"]  # Coding Skills Assessment
        },
        {
            "job_description": "Seeking a manager with strong leadership skills.",
            "expected_assessment_ids": ["1"]  # OPQ
        },
        {
            "job_description": "Need a data analyst for financial modeling.",
            "expected_assessment_ids": ["3"]  # Numerical Reasoning Test
        }
    ]
    if st.button("Run Evaluation"):
        metrics = evaluate_recommendations(test_cases, assessments, k=3)
        st.write(f"**Precision@3**: {metrics['precision@k']:.3f}")
        st.write(f"**Recall@3**: {metrics['recall@k']:.3f}")
        st.write(f"**MAP@3**: {metrics['MAP@k']:.3f}")

if __name__ == "__main__":
    main()