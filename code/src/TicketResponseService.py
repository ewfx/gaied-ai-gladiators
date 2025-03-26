from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import List, Dict
import os
import uvicorn
from AIService import NLPTicketProcessor
import json
import EmailService
import logging
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application"""
    try:
        # Start email processing in background without awaiting
        asyncio.create_task(service.process_emails_background())
        yield
    finally:
        if service.is_processing:
            service.is_processing = False

app = FastAPI(
    title="Ticket Response Service",
    lifespan=lifespan
)

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
        self.is_processing = False
        self.processed_count = 0
        self.processed_emails = set()
        
        self.keywords = ["Problem", "Issue", "request", "Amount", "Expiration Date", 
                        "Name", "Deal Name", "Resolution", "Grievance", "Support"]
        
        try:
            with open(self.config_path, 'r') as config_file:
                config = json.load(config_file)
            
            self.gemini_api_key = config['providers']['google']['api_key']
            self.ai_service = NLPTicketProcessor(
                gemini_api_key=self.gemini_api_key,
                prompts_keywords_file=self.prompts_file
            )
            
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, 
                              detail=f"Configuration file not found: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, 
                              detail=f"Error initializing service: {e}")

    async def process_emails_background(self):
        """Process emails asynchronously without blocking server startup"""
        if self.is_processing:
            return
        
        self.is_processing = True
        try:
            # Create background task for email processing
            async def process_emails():
                EmailService.process_email_folder(self.email_folder, self.keywords)
                email_file = "extracted_data.json"
                if not os.path.exists(email_file):
                    return

                with open(email_file, 'r') as f:
                    sample_emails = json.load(f)

                for email in sample_emails:
                    email_data = email.get("email_data", "")
                    email_data_str = json.dumps(email_data, sort_keys=True)
                    if email_data_str not in self.processed_emails:
                        await asyncio.sleep(0.1)
                        self.ai_service.create_service_ticket(email_data)
                        self.processed_emails.add(email_data_str)
                        self.processed_count += 1

            # Start processing without blocking
            asyncio.create_task(process_emails())
                    
        except Exception as e:
            print(f"Error processing emails: {e}")
        finally:
            self.is_processing = False

    def load_tickets(self) -> List[Dict]:
        try:
            if not os.path.exists(self.csv_path):
                return []
            df = pd.read_csv(self.csv_path)
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

@app.get("/process-status")
async def get_processing_status():
    """Get the current status of email processing"""
    return {
        "is_processing": service.is_processing,
        "processed_count": service.processed_count
    }

@app.post("/process-emails")
async def trigger_email_processing(background_tasks: BackgroundTasks):
    """Manually trigger email processing"""
    if service.is_processing:
        raise HTTPException(status_code=400, detail="Email processing already in progress")
    # Add and execute background task
    background_tasks.add_task(service.process_emails_background)
    await background_tasks()
    return {"message": "Email processing started"}

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
    try:
        # Initialize service before running
        service = TicketResponseService()
        
        # Run the application
        uvicorn.run(
            "TicketResponseService:app",
            host="127.0.0.1",  # Local connections only
            port=8000,
            reload=True,     # Enable auto-reload
            log_level="info"
        )
    except Exception as e:
        print(f"Failed to start server: {e}")