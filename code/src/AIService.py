import os
import json
import csv
import re
import nltk
from typing import List, Dict, Optional
from datetime import datetime
import spacy
import numpy as np
import EmailService

# Download all necessary NLTK resources
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    nltk.download('tokenizers/punkt/PY3/english.pickle')
    nltk.download('averaged_perceptron_tagger_eng')
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

class NLPTicketProcessor:
    def __init__(self, 
                 gemini_api_key: str, 
                 prompts_keywords_file: str,
                 output_ticket_file: str = 'service_tickets.csv'):
        """
        Initialize the Advanced NLP Service Ticket Processor
        
        :param gemini_api_key: API key for Google Gemini AI
        :param prompts_keywords_file: Path to JSON file with prompts and keywords
        :param output_ticket_file: Path to output CSV for service tickets
        """
        # Configure Gemini API
        genai.configure(api_key=gemini_api_key)
        
        # Load prompts and keywords
        with open(prompts_keywords_file, 'r') as f:
            self.prompts_config = json.load(f)
        
        # Output ticket file
        self.output_ticket_file = output_ticket_file
        
        # Initialize Gemini model
        # models = genai.list_models()
        # for model in models:
        #     print(model.name)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Load spaCy model for advanced NLP
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
    
    def preprocess_text(self, text: str) -> str:
        """
        Advanced text preprocessing
        
        :param text: Input text
        :return: Cleaned and preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities using spaCy
        
        :param text: Input text
        :return: Dictionary of named entities
        """
        doc = self.nlp(text)
        
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geopolitical entities
            'PRODUCT': []
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        return entities
    
    def semantic_similarity(self, text: str, keywords: List[str]) -> float:
        """
        Calculate semantic similarity between text and keywords
        
        :param text: Input text
        :param keywords: List of keywords
        :return: Similarity score
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Convert text and keywords to spaCy docs
        text_doc = self.nlp(processed_text)
        keyword_docs = [self.nlp(kw) for kw in keywords]
        # print(keyword_docs)

        
        # Calculate maximum similarity
        max_similarity = 0
        for keyword_doc in keyword_docs:
            similarity = text_doc.similarity(keyword_doc)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def extract_intent_features(self, text: str) -> Dict[str, float]:
        """
        Extract intent features using advanced NLP techniques
        
        :param text: Input text
        :return: Dictionary of intent features
        """
        # Initialize features
        features = {
            'urgency_indicators': 0.0,
            'technical_complexity': 0.0,
            'problem_specificity': 0.0
        }
        
        # Handle empty or None text
        if not text or not text.strip():
            return features
        
        # Tokenize and tag parts of speech
        tokens = nltk.word_tokenize(text)
        if not tokens:  # If no tokens, return default features
            return features
            
        pos_tags = nltk.pos_tag(tokens)
        
        # Urgency indicators
        urgency_words = ['urgent', 'critical', 'immediately', 'asap', 'now']
        urgency_count = sum(1 for word, _ in pos_tags if word.lower() in urgency_words)
        features['urgency_indicators'] = urgency_count / len(tokens) if tokens else 0.0
        
        # Technical complexity
        technical_pos_tags = ['NN', 'NNS', 'NNP']  # Nouns
        technical_tokens = [
            word for word, pos in pos_tags 
            if pos in technical_pos_tags
        ]
        features['technical_complexity'] = len(set(technical_tokens)) / len(tokens) if tokens else 0.0
        
        # Problem specificity (based on noun phrases and descriptive words)
        features['problem_specificity'] = len(set(technical_tokens)) / len(tokens) if tokens else 0.0
        
        return features
    
    def analyze_email_content(self, email_content: Dict[str, str]) -> Dict[str, str]:
        """
        Advanced analysis of email content
        
        :param email_content: Raw email content
        :return: Dictionary with detailed ticket details
        """
        # Extract the body, subject, and attachments of the email
        email_body = email_content.get('body', '')
        email_subject = email_content.get('subject', '')
        email_attachments = email_content.get('attachments', '')

        # Combine body, subject, and attachments into a single text
        combined_content = f"{email_subject} {email_body} {email_attachments}"
        
        # Preprocess combined email content
        preprocessed_content = self.preprocess_text(combined_content)
        
        # Extract named entities
        entities = self.extract_named_entities(combined_content)
        
        # Extract intent features
        intent_features = self.extract_intent_features(combined_content)
        
        # Prepare comprehensive prompt for Gemini
        analysis_prompt = f"""Detailed analysis of the following email content:
        
        Content: {combined_content}
        
        Provide a comprehensive analysis considering:
        1. Primary issue category
        2. Specific technical details
        3. Urgency and impact
        4. Recommended support approach
        5. Potential root cause
        """
        
        try:
            # Gemini AI analysis
            response: GenerateContentResponse = self.model.generate_content(analysis_prompt)
            analysis = response.text
            
            # Advanced ticket details parsing
            ticket_details = self._advanced_ticket_parsing(
                analysis, 
                preprocessed_content, 
                entities, 
                intent_features
            )
            
            return ticket_details
        
        except Exception as e:
            print(f"Error analyzing email content: {e}")
            return {
                'category': 'Unclassified',
                'urgency': 'Medium',
                'support_group': 'General Support',
                'summary': 'Unable to automatically classify',
                'confidence': 0.0
            }
    
    def _advanced_ticket_parsing(self, 
                                  analysis: str, 
                                  preprocessed_content: str,
                                  entities: Dict[str, List[str]],
                                  intent_features: Dict[str, float]) -> Dict[str, str]:
        """
        Advanced ticket parsing with multiple signals
        
        :param analysis: Gemini AI analysis
        :param preprocessed_content: Preprocessed email content
        :param entities: Extracted named entities
        :param intent_features: Extracted intent features
        :return: Structured ticket details
        """
        # Default ticket details
        ticket_details = {
            'category': 'Unclassified',
            'urgency': 'Low',
            'support_group': 'General Support',
            'summary': analysis[:300],
            'confidence': 0.0
        }
        
        # Keyword matching with semantic similarity
        best_match = {
            'similarity': 0.0,
            'config': None
        }
        
        # Iterate through prompt configurations
        for config in self.prompts_config:
            keywords = config.get('keywords', [])
            
            # Calculate semantic similarity
            similarity = self.semantic_similarity(preprocessed_content, keywords)
            
            # Update best match
            if similarity > best_match['similarity']:
                best_match = {
                    'similarity': similarity,
                    'config': config
                }
        
        # Update ticket details if a good match is found
        if best_match['similarity'] > 0.5:
            match_config = best_match['config']
            ticket_details.update({
                'requestType': match_config.get('requestType', ticket_details['category']),
                'subRequestType': match_config.get('subRequestType', ticket_details['category']),
                'support_group': match_config.get('support_group', ticket_details['support_group']),
                'confidence': best_match['similarity']
            })
        
        # Adjust urgency based on intent features
        if intent_features['urgency_indicators'] > 0.3:
            ticket_details['urgency'] = 'High'
        elif intent_features['urgency_indicators'] > 0.1:
            ticket_details['urgency'] = 'Medium'
        
        # Incorporate named entities for additional context
        if entities['PRODUCT']:
            ticket_details['summary'] += f" Product(s) mentioned: {', '.join(entities['PRODUCT'])}"
        
        # Add technical complexity to summary
        if intent_features['technical_complexity'] > 0.5:
            ticket_details['summary'] += " High technical complexity detected."
        
        return ticket_details
    
    def create_service_ticket(self, email_content: Dict[str, str]) -> None:
        """
        Create a service ticket from email content
        
        :param email_content: Raw email content as a dictionary
        """
        try:
            # Analyze content
            ticket_details = self.analyze_email_content(email_content)
            
            # Prepare ticket for CSV with default values for missing fields
            ticket_row = {
                'Timestamp': datetime.now().isoformat(),
                'Request Type': ticket_details.get('requestType', 'Unclassified'),
                'Sub Request Type': ticket_details.get('subRequestType', 'General'),
                'Support Group': ticket_details.get('support_group', 'General Support'),
                'Urgency': ticket_details.get('urgency', 'Medium'),
                'Confidence': ticket_details.get('confidence', 0.0),
                'Summary': ticket_details.get('summary', 'No summary available'),
                'Original Content': email_content.get('body', '')
            }
            
            # Write to CSV
            file_exists = os.path.exists(self.output_ticket_file)
            with open(self.output_ticket_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=ticket_row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(ticket_row)
            
            print(f"Service ticket created: {ticket_row['Support Group']} - {ticket_row['Request Type']} (Confidence: {ticket_row['Confidence']:.2f})")
        
        except Exception as e:
            print(f"Error creating service ticket: {e}")
            # Create a fallback ticket
            ticket_row = {
                'Timestamp': datetime.now().isoformat(),
                'Request Type': 'Error',
                'Sub Request Type': 'System Error',
                'Support Group': 'IT Support',
                'Urgency': 'Medium',
                'Confidence': 0.0,
                'Summary': f'Error processing ticket: {str(e)}',
                'Original Content': email_content.get('body', '')
            }
            # Write fallback ticket to CSV
            with open(self.output_ticket_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=ticket_row.keys())
                if not os.path.exists(self.output_ticket_file):
                    writer.writeheader()
                writer.writerow(ticket_row)

# # Usage
# def main():
#     # You would replace these with your actual values
#     # Load configuration from config.json
#     with open('config.json', 'r') as config_file:
#         config = json.load(config_file)
    
#     GEMINI_API_KEY = config['providers']['google']['api_key']
#     PROMPTS_FILE = "prompts_keywords.json"
    
#     # Initialize processor
#     ticket_processor = NLPTicketProcessor(
#         gemini_api_key=GEMINI_API_KEY,
#         prompts_keywords_file=PROMPTS_FILE
#     )
#     keywords = ["Problem", "Issue", "request", "Amount", "Ëxpiration Date", "Name", "Deal Name", "Resolution", "Grievence"]
#     #Email content
#     with open(EmailService.process_email_folder("/",keywords), 'r') as f:
#             sample_emails = json.load(f)   
#     # Create service tickets
#     for email in sample_emails:
#         ticket_processor.create_service_ticket(email)

# if __name__ == "__main__":
#     main()
