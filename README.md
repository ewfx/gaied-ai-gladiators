# 🚀 Project Name

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
Welcome to our GenAI-based Email Classification and ticketing for loan servicing platforms. This project was developed as part of the Technology Hackathon focused on applying generative AI to streamline loan servicing workflows. Our solution uses advanced LLMs to automatically classify incoming emails, extract relevant information, and generate structured data for seamless integration with existing loan servicing systems.


## 🎥 Demo
🔗 [Live Demo](#) (if applicable)  
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)

## 💡 Inspiration
Loan servicing operations are often bogged down by manual email processing, leading to:
- Delayed response times to customer requests
- Inconsistent classification of service requests
- Human error in data extraction
- Inefficient allocation of human resources
Our team was inspired to solve these challenges by leveraging the latest advancements in generative AI to create an intelligent system that can understand, classify, and extract information from unstructured emails and documents with high accuracy and consistency.

## ⚙️ What It Does
**Our solution provides an end-to-end AI pipeline that:**
- Reads and processes emails - Ingests emails and attachments from various sources
- Classifies requests - Identifies the primary request type and sub-request type
- Extracts critical information - Pulls out relevant data points like amounts, dates, and account details
- Assigns confidence scores - Provides reliability metrics for classifications
- Detects duplicates - Identifies potential duplicate requests to prevent redundant processing
- Generates structured output - Creates standardized JSON for system integration which can be integration with UI
- Provides explanations - Offers reasoning for classifications to assist human reviewers
- The system handles a comprehensive range of loan servicing request types, including:
Adjustments
Administrative Unit Transfers
Closing Notices
Commitment Changes
Fee Payments
Money Movement (Inbound and Outbound)
And more...

## 🛠️ How We Built It
Our solution consists of three main components:
1. **Email Processing Service**
Connects to email servers and document repositories
Parses emails, extracts attachments
Preprocesses text for optimal AI analysis

2. **AI Classification Service**
Utilizes LLMs (OpenAI's GPT,Gemini Pro,NLTK ,spacy) with custom prompt engineering
Implements a classification pipeline with context-aware prompting
Features confidence scoring algorithms
Includes duplicate detection logic
Maintains explainability for all classifications

3. **Service Ticket Response System**
Generates structured JSON outputs
Provides RESTful API endpoints for integration
Includes a dashboard for human review of classifications
Implements feedback loops for continuous improvement
The system architecture follows modern microservices design principles, ensuring scalability and maintainability.

## 🚧 Challenges We Faced

Throughout development, we encountered several challenges:
Prompt Engineering Complexity - Fine-tuning prompts to handle the nuanced differences between request types required extensive iteration and testing.
Context Preservation - Ensuring the AI model maintained awareness of the full context when processing lengthy emails with multiple topics.
Confidence Calibration - Developing reliable confidence metrics that accurately reflected classification certainty.
System Integration - Designing clean interfaces between services while maintaining end-to-end performance.
Handling Edge Cases - Building resilience for unusual or ambiguous requests that don't clearly fall into predefined categories.

## 🏃 How to Run
Prerequisites

Python 3.8+
Docker and Docker Compose
OpenAI API key or other LLM API credentials
Node.js 16+ (for the frontend dashboard)

Installation
Clone the repository:

bashCopygit clone https://github.com/yourusername/genai-loan-classification.git
cd genai-loan-classification

Set up environment variables:
bashCopycp .env.example .env
Edit .env with your API keys and configuration
Start the services using Docker Compose:
bashCopydocker-compose up -d

Access the dashboard:
Copyhttp://localhost:3000
Configuration
The system can be configured through the .env file and config/ directory:

config/request_types.json - Defines the request types and sub-request types
config/prompts.json - Contains the prompt templates for the AI service
config/integration.json - Specifies integration endpoints and formats


## 🏗️ Tech Stack
**Email Processing:**
Python with email, mime, and pytesseract libraries
Tesseract OCR
PDF processing with PyPDF2/pdfminer

**AI Service:**
Python with OpenAI API / HuggingFace / LangChain
Scikit-learn for confidence scoring and analytics
Redis for caching and rate limiting

**Service Ticket Response:**
FastAPI for backend API
PostgreSQL for data persistence
Node.js and React for dashboard UI
Socket.IO for real-time updates



## 👥 Team
Our diverse team brings together expertise in AI, software engineering, and financial services:

K V Subramanyeswar Sarma (Lead Software Engineer): A seasoned leader with deep technical expertise in software development, driving the architectural and strategic direction of our project.

Siddhant Antil (Software Engineer): A dynamic software engineer contributing fresh perspectives and coding expertise to accelerate development.

Santosh Kumar Potnuru (Lead Software Engineer): An experienced engineer providing robust design and implementation strategies to enhance system efficiency.

Sri Durga Sravanthi Bikkina (Senior Software Engineer):A mainframe modernization expert now exploring the AI/ML space, bringing a unique blend of legacy transformation knowledge and innovative AI-driven solutions.

Dharanidhar L Lokanandi (Senior Software Engineer):  A skilled Scrum Master facilitating agile execution while leading documentation efforts to ensure clear and structured project artifacts.
