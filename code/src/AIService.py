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

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

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
        self.model = genai.GenerativeModel('gemini-pro')
        
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
        # Tokenize and tag parts of speech
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        # Feature extraction
        features = {
            'urgency_indicators': 0.0,
            'technical_complexity': 0.0,
            'problem_specificity': 0.0
        }
        
        # Urgency indicators
        urgency_words = ['urgent', 'critical', 'immediately', 'asap', 'now']
        features['urgency_indicators'] = sum(
            1 for word, _ in pos_tags if word.lower() in urgency_words
        ) / len(tokens)
        
        # Technical complexity
        technical_pos_tags = ['NN', 'NNS', 'NNP']  # Nouns
        technical_tokens = [
            word for word, pos in pos_tags 
            if pos in technical_pos_tags
        ]
        features['technical_complexity'] = len(set(technical_tokens)) / len(tokens)
        
        # Problem specificity (based on noun phrases and descriptive words)
        features['problem_specificity'] = len(set(technical_tokens)) / max(len(tokens), 1)
        
        return features
    
    def analyze_email_content(self, email_content: str) -> Dict[str, str]:
        """
        Advanced analysis of email content
        
        :param email_content: Raw email content
        :return: Dictionary with detailed ticket details
        """
        # Preprocess email content
        preprocessed_content = self.preprocess_text(email_content)
        
        # Extract named entities
        entities = self.extract_named_entities(email_content)
        
        # Extract intent features
        intent_features = self.extract_intent_features(email_content)
        
        # Prepare comprehensive prompt for Gemini
        analysis_prompt = f"""Detailed analysis of the following email content:
        
        Content: {email_content}
        
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
        for config in self.prompts_config.get('prompts_keywords', []):
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
                'category': match_config.get('category', ticket_details['category']),
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
    
    def create_service_ticket(self, email_content: str) -> None:
        """
        Create a service ticket from email content
        
        :param email_content: Raw email content
        """
        # Analyze content
        ticket_details = self.analyze_email_content(email_content)
        
        # Prepare ticket for CSV
        ticket_row = {
            'Timestamp': datetime.now().isoformat(),
            'Category': ticket_details['category'],
            'Support Group': ticket_details['support_group'],
            'Urgency': ticket_details['urgency'],
            'Confidence': ticket_details['confidence'],
            'Summary': ticket_details['summary'],
            'Original Content': email_content
        }
        
        # Write to CSV
        file_exists = os.path.exists(self.output_ticket_file)
        with open(self.output_ticket_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ticket_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(ticket_row)
        
        print(f"Service ticket created: {ticket_details['support_group']} - {ticket_details['category']} (Confidence: {ticket_details['confidence']:.2f})")

# Usage
def main():
    # You would replace these with your actual values
    GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY'
    PROMPTS_FILE = 'prompts_keywords.json'
    
    # Initialize processor
    ticket_processor = NLPTicketProcessor(
        gemini_api_key=GEMINI_API_KEY,
        prompts_keywords_file=PROMPTS_FILE
    )
    keywords = ["Problem", "Issue", "request", "Amount", "Ëxpiration Date", "Name", "Deal Name", "Resolution"]
    #Email content
    with open(EmailService.process_email_folder("/",keywords), 'r') as f:
            sample_emails = json.load(f)   
    # Create service tickets
    for email in sample_emails.get('email_data', [])::
        ticket_processor.create_service_ticket(email)

if __name__ == "__main__":
    main()
