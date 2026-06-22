# AI Resume Rewriter & ATS Optimization System

## Overview

AI Resume Rewriter is an NLP-powered application that automatically tailors resumes to specific job descriptions. The system analyzes resume-job compatibility, identifies missing skills, calculates ATS scores, and generates an optimized ATS-friendly resume using Groq-hosted Llama 3.3 70B.

The objective of this project is to help job seekers improve resume relevance, keyword alignment, and overall ATS performance while preserving genuine experience and qualifications.

---

## Problem Statement

Many qualified candidates are rejected by Applicant Tracking Systems (ATS) because their resumes do not contain the required skills, keywords, or terminology present in job descriptions.

This project addresses that challenge by:

* Analyzing resumes against job descriptions
* Detecting missing technical skills and keywords
* Measuring resume-job alignment using NLP techniques
* Automatically rewriting resumes to improve ATS compatibility

---

## Key Features

### Resume Parsing

* Supports PDF, DOCX, and TXT resumes
* Extracts and cleans resume text automatically

### Job Description Analysis

* Extracts technical skills and job-specific requirements
* Identifies missing skills and keywords

### ATS Score Evaluation

Uses a custom 4-factor scoring framework:

* TF-IDF Similarity (15%)
* Keyword Overlap Analysis (35%)
* Technical Skill Match (35%)
* Semantic Similarity (15%)

### AI Resume Rewriting

* Powered by Groq Llama 3.3 70B
* Rewrites professional summary, experience, and project descriptions
* Preserves genuine candidate information
* Improves alignment with target job roles

### Resume Coach Chatbot

* Provides personalized resume improvement suggestions
* Explains missing skills and optimization opportunities

### Export Functionality

* Download optimized resumes in DOCX format
* Download optimized resumes in TXT format

---

## NLP Pipeline

Resume Upload
↓
Text Extraction
↓
Text Cleaning & Normalization
↓
Tokenization
↓
Technical Skill Extraction
↓
Keyword Gap Analysis
↓
ATS Score Calculation
↓
Semantic Similarity Analysis
↓
Groq AI Resume Rewriting
↓
Optimized Resume Generation

---

## Technologies Used

### Programming Language

* Python

### NLP & Machine Learning

* Scikit-Learn
* TF-IDF Vectorization
* Cosine Similarity
* Sentence Transformers
* Semantic Embeddings

### Generative AI

* Groq API
* Llama 3.3 70B Versatile

### Deployment

* Streamlit

### File Processing

* PyPDF
* Python-Docx

---

## Project Architecture

1. Resume and Job Description are uploaded by the user.
2. Text is extracted and preprocessed.
3. Technical skills are identified from the Job Description.
4. Missing skills are detected from the resume.
5. ATS compatibility is calculated using four scoring techniques.
6. Groq Llama 3.3 70B generates an optimized resume.
7. The optimized resume is displayed and made available for download.

---

## Business Impact

* Reduces manual resume customization effort.
* Helps candidates understand skill gaps.
* Improves job-specific resume targeting.
* Demonstrates practical integration of NLP and Large Language Models.
* Provides a scalable AI-assisted resume optimization solution.

---

## Future Enhancements

* Multi-language resume optimization.
* Support for multiple ATS scoring standards.
* Resume ranking against multiple job descriptions.
* Automatic cover letter generation.
* Advanced recruiter feedback analytics.

---

## Outcome

Successfully developed an end-to-end AI-powered Resume Rewriter that combines NLP, Semantic Similarity Analysis, ATS Scoring, and Large Language Models to generate job-specific resumes and improve resume-job alignment.

