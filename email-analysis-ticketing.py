import os
import re
import email
import imaplib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sqlite3
import openai

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Configuration
class Config:
    # Email settings
    EMAIL_USERNAME = os.environ.get('EMAIL_USERNAME', 'your_email@example.com')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'your_password')
    IMAP_SERVER = os.environ.get('IMAP_SERVER', 'imap.example.com')
    SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.example.com')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
    
    # OpenAI API settings
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your_openai_api_key')
    
    # Database settings
    DB_PATH = os.environ.get('DB_PATH', 'tickets.db')
    
    # Ticket settings
    SLA_HOURS = int(os.environ.get('SLA_HOURS', 24))
    PRIORITY_KEYWORDS = {
        'high': ['urgent', 'critical', 'emergency', 'asap', 'immediately'],
        'medium': ['important', 'significant', 'attention', 'soon'],
        'low': ['whenever', 'low priority', 'not urgent']
    }
    
    # Department routing
    DEPARTMENTS = {
        'technical': ['error', 'bug', 'crash', 'technical', 'not working', 'broken'],
        'billing': ['payment', 'invoice', 'charge', 'refund', 'subscription', 'billing'],
        'account': ['password', 'login', 'account', 'profile', 'access'],
        'general': ['question', 'inquiry', 'information', 'help', 'support']
    }

# Initialize OpenAI
openai.api_key = Config.OPENAI_API_KEY

# Database setup
def setup_database():
    conn = sqlite3.connect(Config.DB_PATH)
    cursor = conn.cursor()
    
    # Create tickets table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT,
        sender TEXT,
        content TEXT,
        category TEXT,
        department TEXT,
        priority TEXT,
        status TEXT,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        due_by TIMESTAMP,
        message_id TEXT UNIQUE,
        summary TEXT,
        sentiment REAL
    )
    ''')
    
    # Create responses table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id INTEGER,
        content TEXT,
        sent_at TIMESTAMP,
        FOREIGN KEY (ticket_id) REFERENCES tickets (id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Email processing
class EmailProcessor:
    def __init__(self):
        self.connect_to_email()
    
    def connect_to_email(self):
        self.mail = imaplib.IMAP4_SSL(Config.IMAP_SERVER)
        self.mail.login(Config.EMAIL_USERNAME, Config.EMAIL_PASSWORD)
        self.mail.select('inbox')
    
    def fetch_emails(self, days=1):
        """Fetch emails from the last X days"""
        date = (datetime.now() - timedelta(days=days)).strftime("%d-%b-%Y")
        result, data = self.mail.search(None, f'(SINCE {date})')
        
        email_ids = data[0].split()
        emails = []
        
        for email_id in email_ids:
            result, data = self.mail.fetch(email_id, '(RFC822)')
            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)
            
            # Extract email data
            subject = msg['subject']
            sender = msg['from']
            message_id = msg['message-id']
            
            # Get email body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        break
            else:
                body = msg.get_payload(decode=True).decode()
            
            emails.append({
                'subject': subject,
                'sender': sender,
                'body': body,
                'message_id': message_id,
                'date': msg['date']
            })
        
        return emails
    
    def send_email(self, to_address, subject, body):
        """Send an email response"""
        msg = MIMEMultipart()
        msg['From'] = Config.EMAIL_USERNAME
        msg['To'] = to_address
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT)
        server.starttls()
        server.login(Config.EMAIL_USERNAME, Config.EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        return True

# Ticket processing
class TicketProcessor:
    def __init__(self):
        self.setup_nlp()
    
    def setup_nlp(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def preprocess_text(self, text):
        """Clean and preprocess text for analysis"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        filtered_tokens = [w for w in tokens if w not in self.stop_words]
        return " ".join(filtered_tokens)
    
    def categorize_email(self, subject, body):
        """Categorize email using AI"""
        combined_text = f"{subject} {body}"
        
        # Use OpenAI to categorize
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that categorizes customer emails. Categorize this email into one of these categories: Question, Problem, Request, Feedback, Complaint, Other. Respond with just the category name."},
                {"role": "user", "content": combined_text[:1000]}  # Limit to first 1000 chars
            ]
        )
        
        category = response.choices[0].message.content.strip()
        return category
    
    def determine_priority(self, subject, body):
        """Determine ticket priority based on keywords and sentiment"""
        combined_text = f"{subject} {body}".lower()
        
        # Check for priority keywords
        for priority, keywords in Config.PRIORITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in combined_text:
                    return priority
        
        # Default to medium priority
        return "medium"
    
    def route_to_department(self, subject, body):
        """Route ticket to appropriate department"""
        combined_text = f"{subject} {body}".lower()
        
        # Check department keywords
        for dept, keywords in Config.DEPARTMENTS.items():
            for keyword in keywords:
                if keyword in combined_text:
                    return dept
        
        # Use AI to determine department if no keywords match
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that routes customer emails to departments. Route this email to one of these departments: technical, billing, account, general. Respond with just the department name."},
                {"role": "user", "content": combined_text[:1000]}
            ]
        )
        
        department = response.choices[0].message.content.strip().lower()
        if department in Config.DEPARTMENTS.keys():
            return department
        
        # Default to general support
        return "general"
    
    def generate_summary(self, subject, body):
        """Generate a concise summary of the email"""
        combined_text = f"{subject} {body}"
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that summarizes customer emails. Summarize this email in 1-2 sentences."},
                {"role": "user", "content": combined_text[:1500]}
            ]
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of email"""
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that analyzes sentiment. Rate the sentiment of this text on a scale from -1.0 (very negative) to 1.0 (very positive). Respond with just the number."},
                {"role": "user", "content": text[:1000]}
            ]
        )
        
        sentiment_score = float(response.choices[0].message.content.strip())
        return sentiment_score
    
    def create_ticket(self, email_data):
        """Create a ticket from email data"""
        subject = email_data['subject']
        sender = email_data['sender']
        body = email_data['body']
        message_id = email_data['message_id']
        
        # Process email
        category = self.categorize_email(subject, body)
        department = self.route_to_department(subject, body)
        priority = self.determine_priority(subject, body)
        summary = self.generate_summary(subject, body)
        sentiment = self.analyze_sentiment(body)
        
        # Calculate SLA
        created_at = datetime.now()
        due_by = created_at + timedelta(hours=Config.SLA_HOURS)
        
        # Create ticket in database
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO tickets (
            subject, sender, content, category, department, priority,
            status, created_at, updated_at, due_by, message_id, summary, sentiment
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            subject, sender, body, category, department, priority,
            'new', created_at, created_at, due_by, message_id, summary, sentiment
        ))
        
        ticket_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return ticket_id
    
    def generate_response(self, ticket_id):
        """Generate an automated response for a ticket"""
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT subject, content, category, department, priority FROM tickets WHERE id = ?', (ticket_id,))
        ticket_data = cursor.fetchone()
        conn.close()
        
        if not ticket_data:
            return None
        
        subject, content, category, department, priority = ticket_data
        
        # Generate response using AI
        prompt = f"""
        Generate a professional customer service response for a {priority} priority ticket in the {department} department.
        The customer's issue is related to: {subject}
        
        Ticket category: {category}
        
        Original message:
        {content[:500]}...
        
        The response should:
        1. Acknowledge the customer's issue
        2. Provide next steps or information
        3. Set expectations for resolution
        4. Thank the customer
        
        Write as a helpful customer service representative.
        """
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that generates customer service responses. Write a professional and helpful response."},
                {"role": "user", "content": prompt}
            ]
        )
        
        generated_response = response.choices[0].message.content.strip()
        
        # Save response to database
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO responses (ticket_id, content, sent_at)
        VALUES (?, ?, ?)
        ''', (ticket_id, generated_response, datetime.now()))
        
        conn.commit()
        conn.close()
        
        return generated_response

# Dashboard and analytics
class TicketAnalytics:
    def __init__(self):
        self.conn = sqlite3.connect(Config.DB_PATH)
    
    def get_ticket_stats(self):
        """Get basic ticket statistics"""
        cursor = self.conn.cursor()
        
        # Total tickets
        cursor.execute('SELECT COUNT(*) FROM tickets')
        total_tickets = cursor.fetchone()[0]
        
        # Tickets by status
        cursor.execute('SELECT status, COUNT(*) FROM tickets GROUP BY status')
        status_counts = dict(cursor.fetchall())
        
        # Tickets by department
        cursor.execute('SELECT department, COUNT(*) FROM tickets GROUP BY department')
        department_counts = dict(cursor.fetchall())
        
        # Tickets by priority
        cursor.execute('SELECT priority, COUNT(*) FROM tickets GROUP BY priority')
        priority_counts = dict(cursor.fetchall())
        
        # Average response time
        cursor.execute('''
        SELECT AVG(julianday(r.sent_at) - julianday(t.created_at)) * 24
        FROM tickets t
        JOIN responses r ON t.id = r.ticket_id
        ''')
        avg_response_time = cursor.fetchone()[0] or 0
        
        return {
            'total_tickets': total_tickets,
            'by_status': status_counts,
            'by_department': department_counts,
            'by_priority': priority_counts,
            'avg_response_time': avg_response_time
        }
    
    def get_sla_compliance(self):
        """Calculate SLA compliance"""
        cursor = self.conn.cursor()
        
        # Get all tickets with responses
        cursor.execute('''
        SELECT t.id, t.due_by, MIN(r.sent_at) as first_response
        FROM tickets t
        LEFT JOIN responses r ON t.id = r.ticket_id
        GROUP BY t.id
        ''')
        
        tickets_with_responses = cursor.fetchall()
        
        total = len(tickets_with_responses)
        met_sla = 0
        
        for ticket_id, due_by, first_response in tickets_with_responses:
            if first_response and first_response <= due_by:
                met_sla += 1
        
        compliance_rate = (met_sla / total) * 100 if total > 0 else 0
        
        return {
            'total_tickets': total,
            'met_sla': met_sla,
            'compliance_rate': compliance_rate
        }
    
    def get_sentiment_analysis(self):
        """Get sentiment analysis of tickets"""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT sentiment FROM tickets')
        sentiments = [row[0] for row in cursor.fetchall()]
        
        if not sentiments:
            return {'average': 0, 'positive': 0, 'neutral': 0, 'negative': 0}
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        positive = sum(1 for s in sentiments if s > 0.3)
        neutral = sum(1 for s in sentiments if -0.3 <= s <= 0.3)
        negative = sum(1 for s in sentiments if s < -0.3)
        
        return {
            'average': avg_sentiment,
            'positive': positive,
            'neutral': neutral,
            'negative': negative
        }
    
    def get_trending_topics(self):
        """Identify trending topics in tickets"""
        cursor = self.conn.cursor()
        
        # Get recent tickets
        cursor.execute('SELECT content FROM tickets WHERE created_at > datetime("now", "-7 days")')
        recent_contents = [row[0] for row in cursor.fetchall()]
        
        if not recent_contents:
            return []
        
        # Process and vectorize content
        processor = TicketProcessor()
        processed_contents = [processor.preprocess_text(content) for content in recent_contents]
        
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(processed_contents)
        
        # Get top terms
        feature_names = vectorizer.get_feature_names_out()
        tfidf_sums = tfidf_matrix.sum(axis=0).A1
        
        # Get top 10 terms with their weights
        top_indices = tfidf_sums.argsort()[-10:][::-1]
        top_terms = [(feature_names[i], tfidf_sums[i]) for i in top_indices]
        
        return top_terms
    
    def close(self):
        self.conn.close()

# Main application
class EmailTicketingSystem:
    def __init__(self):
        setup_database()
        self.email_processor = EmailProcessor()
        self.ticket_processor = TicketProcessor()
        self.analytics = TicketAnalytics()
    
    def process_new_emails(self):
        """Process new emails and create tickets"""
        print("Fetching new emails...")
        emails = self.email_processor.fetch_emails(days=1)
        
        for email_data in emails:
            # Check if email already processed
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM tickets WHERE message_id = ?', (email_data['message_id'],))
            existing = cursor.fetchone()
            conn.close()
            
            if existing:
                continue
            
            # Create ticket
            print(f"Creating ticket for email: {email_data['subject']}")
            ticket_id = self.ticket_processor.create_ticket(email_data)
            
            # Generate automated response
            response = self.ticket_processor.generate_response(ticket_id)
            
            # Send response
            if response:
                subject = f"Re: {email_data['subject']}"
                self.email_processor.send_email(
                    to_address=email_data['sender'],
                    subject=subject,
                    body=response
                )
                print(f"Sent automated response for ticket #{ticket_id}")
    
    def show_dashboard(self):
        """Display dashboard statistics"""
        stats = self.analytics.get_ticket_stats()
        sla_stats = self.analytics.get_sla_compliance()
        sentiment_stats = self.analytics.get_sentiment_analysis()
        trending_topics = self.analytics.get_trending_topics()
        
        print("\n===== EMAIL TICKETING SYSTEM DASHBOARD =====")
        print(f"Total Tickets: {stats['total_tickets']}")
        print("\nTicket Status:")
        for status, count in stats.get('by_status', {}).items():
            print(f"  {status}: {count}")
        
        print("\nDepartment Distribution:")
        for dept, count in stats.get('by_department', {}).items():
            print(f"  {dept}: {count}")
        
        print("\nPriority Distribution:")
        for priority, count in stats.get('by_priority', {}).items():
            print(f"  {priority}: {count}")
        
        print(f"\nAverage Response Time: {stats['avg_response_time']:.2f} hours")
        print(f"SLA Compliance: {sla_stats['compliance_rate']:.2f}% ({sla_stats['met_sla']}/{sla_stats['total_tickets']})")
        
        print("\nCustomer Sentiment:")
        print(f"  Average: {sentiment_stats['average']:.2f}")
        print(f"  Positive: {sentiment_stats['positive']}")
        print(f"  Neutral: {sentiment_stats['neutral']}")
        print(f"  Negative: {sentiment_stats['negative']}")
        
        print("\nTrending Topics:")
        for term, weight in trending_topics:
            print(f"  {term}: {weight:.2f}")
        
        print("==========================================")
    
    def run(self):
        """Run the email ticketing system"""
        print("Starting Email Ticketing System...")
        self.process_new_emails()
        self.show_dashboard()
        print("Process completed.")

# Entry point
if __name__ == "__main__":
    system = EmailTicketingSystem()
    system.run()
