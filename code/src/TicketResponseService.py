from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import List, Dict
import os
import uvicorn
from AIService import NLPTicketProcessor
import json
import EmailService
import logging
app = FastAPI(title="Ticket Response Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TicketResponseService:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.base_dir, "service_tickets.csv")
        self.config_path = os.path.join(self.base_dir, "config.json")
        self.prompts_file = os.path.join(self.base_dir, "prompts_keywords.json")
        self.email_folder = os.path.join(self.base_dir, "../../emails")
        
        self.keywords = ["Problem", "Issue", "request", "Amount", "Expiration Date", 
                        "Name", "Deal Name", "Resolution", "Grievance", "Support"]
        
        # Load configuration
        try:
            with open(self.config_path, 'r') as config_file:
                config = json.load(config_file)
            
            self.gemini_api_key = config['providers']['google']['api_key']
            
            # Initialize AI service
            self.ai_service = NLPTicketProcessor(
                gemini_api_key=self.gemini_api_key,
                prompts_keywords_file=self.prompts_file
            )
            
            # Process emails and generate CSV
            self.process_emails()
            
        except FileNotFoundError as e:
            logging.error(f"Configuration file not found: {e}")
            raise HTTPException(status_code=500, 
                              detail=f"Configuration file not found: {e}")
        except Exception as e:
            logging.error(f"Error initializing service: {e}")
            raise HTTPException(status_code=500, 
                              detail=f"Error initializing service: {e}")

    def process_emails(self):
        EmailService.process_email_folder(self.email_folder, self.keywords)
        email_file = "extracted_data.json"
        if not email_file or not os.path.exists(email_file):
            logging.error(f"Email file not found or invalid: {email_file}")
            raise HTTPException(status_code=500, detail="Email file not found or invalid.")
        
        with open(email_file, 'r') as f:
            sample_emails = json.load(f)
        
        processed_emails = set()
        for email in sample_emails:
            email_data = email.get("email_data", "")
            email_data_str = json.dumps(email_data, sort_keys=True)  # Convert to a hashable type
            if email_data_str not in processed_emails:
                self.ai_service.create_service_ticket(email_data)
                processed_emails.add(email_data_str)

    def load_tickets(self) -> List[Dict]:
        try:
            if not os.path.exists(self.csv_path):
                return []
            df = pd.read_csv(self.csv_path)
            # Convert DataFrame to structured dictionaries
            tickets = []
            for _, row in df.iterrows():
                ticket = {
                    'request_id': row.get('Request ID', ''),
                    'timestamp': row.get('Timestamp', ''),
                    'request_type': row.get('Request Type', 'Unclassified'),
                    'sub_request_type': row.get('Sub Request Type', 'General'),
                    'support_group': row.get('Support Group', 'General Support'),
                    'urgency': row.get('Urgency', 'Medium'),
                    'confidence': float(row.get('Confidence', 0.0)),
                    'summary': row.get('Summary', 'No summary available'),
                }
                tickets.append(ticket)
            return tickets
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading tickets: {str(e)}")

service = TicketResponseService()

@app.get("/tickets/", response_model=List[Dict])
async def get_all_tickets():
    """
    Get all service tickets with their AI-generated responses
    """
    tickets = service.load_tickets()
    return tickets

@app.get("/tickets/{ticket_id}")
async def get_ticket_by_id(ticket_id: int):
    """
    Get a specific ticket by its ID
    """
    tickets = service.load_tickets()
    ticket = next((t for t in tickets if t.get('ticket_id') == ticket_id), None)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)