import os
import re
import email
import imaplib
import time
import sqlite3
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("service_requests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ServiceRequestSystem")

class ServiceRequestSystem:
    def __init__(self, config):
        self.config = config
        self.db_conn = self.init_database()
        
        # Download NLTK resources if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        # Initialize keyword dictionaries for request classification
        self.group_keywords = {
            'IT_SUPPORT': ['password', 'login', 'computer', 'laptop', 'printer', 'network', 'wifi', 'software', 'hardware', 'system', 'access'],
            'HR': ['payroll', 'benefits', 'vacation', 'leave', 'hiring', 'employee', 'training', 'compensation', 'onboarding', 'offboarding'],
            'FACILITIES': ['building', 'office', 'maintenance', 'repair', 'cleaning', 'furniture', 'temperature', 'lights', 'restroom', 'parking'],
            'FINANCE': ['invoice', 'payment', 'reimbursement', 'expense', 'budget', 'purchase', 'accounting', 'tax', 'financial', 'refund'],
            'LEGAL': ['contract', 'agreement', 'legal', 'compliance', 'terms', 'copyright', 'patent', 'trademark', 'regulation', 'policy']
        }
        
        # Initialize priority keywords
        self.priority_keywords = {
            'HIGH': ['urgent', 'critical', 'immediate', 'emergency', 'asap', 'crucial', 'severe', 'important'],
            'MEDIUM': ['soon', 'attention', 'timely', 'moderate', 'priority'],
            'LOW': ['when possible', 'eventually', 'minor', 'not urgent', 'low priority']
        }
    
    def init_database(self):
        """Initialize SQLite database for service request storage"""
        conn = sqlite3.connect(self.config['database_file'])
        cursor = conn.cursor()
        
        # Create service_requests table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS service_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE,
            subject TEXT,
            sender TEXT,
            recipient TEXT,
            status TEXT,
            priority TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            assigned_group TEXT,
            assigned_to TEXT,
            category TEXT,
            sla_due_date TIMESTAMP
        )
        ''')
        
        # Create messages table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT,
            message_id TEXT UNIQUE,
            sender TEXT,
            content TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (request_id) REFERENCES service_requests (request_id)
        )
        ''')
        
        # Create groups table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_name TEXT UNIQUE,
            description TEXT,
            email TEXT,
            members TEXT,
            default_sla_hours INTEGER
        )
        ''')
        
        # Insert default groups if they don't exist
        default_groups = [
            ('IT_SUPPORT', 'IT Support Team', 'it@example.com', 'John,Sarah,Mike', 24),
            ('HR', 'Human Resources', 'hr@example.com', 'Anna,David', 48),
            ('FACILITIES', 'Facilities Management', 'facilities@example.com', 'Robert,Lisa', 72),
            ('FINANCE', 'Finance Department', 'finance@example.com', 'Karen,Tom', 48),
            ('LEGAL', 'Legal Department', 'legal@example.com', 'James,Emily', 72),
            ('GENERAL', 'General Inquiries', 'support@example.com', 'Alex,Maria', 48)
        ]
        
        for group in default_groups:
            cursor.execute('''
            INSERT OR IGNORE INTO groups
            (group_name, description, email, members, default_sla_hours)
            VALUES (?, ?, ?, ?, ?)
            ''', group)
        
        conn.commit()
        return conn
    
    def generate_request_id(self):
        """Generate a unique service request ID"""
        timestamp = int(time.time())
        return f"SR-{timestamp}"
    
    def fetch_emails(self):
        """Connect to email server and fetch unread emails"""
        try:
            mail = imaplib.IMAP4_SSL(self.config['imap_server'])
            mail.login(self.config['email'], self.config['password'])
            mail.select('inbox')
            
            # Search for all unread emails
            status, data = mail.search(None, 'UNSEEN')
            
            if status != 'OK':
                logger.info("No new messages found")
                return []
            
            email_ids = data[0].split()
            emails = []
            
            for e_id in email_ids:
                status, data = mail.fetch(e_id, '(RFC822)')
                raw_email = data[0][1]
                emails.append(email.message_from_bytes(raw_email))
            
            mail.close()
            mail.logout()
            logger.info(f"Fetched {len(emails)} new emails")
            return emails
            
        except Exception as e:
            logger.error(f"Error fetching emails: {str(e)}")
            return []
    
    def classify_request(self, subject, body):
        """Classify the service request by analyzing subject and body"""
        # Combine subject and body for analysis
        text = f"{subject} {body}".lower()
        words = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.isalnum() and word not in stop_words]
        
        # Determine group assignment based on keywords
        group_scores = {group: 0 for group in self.group_keywords}
        
        for group, keywords in self.group_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    group_scores[group] += 1
        
        # Determine priority based on keywords
        priority_scores = {priority: 0 for priority in self.priority_keywords}
        
        for priority, keywords in self.priority_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    priority_scores[priority] += 1
        
        # Assign default values if no matches found
        assigned_group = max(group_scores.items(), key=lambda x: x[1])[0] if any(group_scores.values()) else "GENERAL"
        priority = max(priority_scores.items(), key=lambda x: x[1])[0] if any(priority_scores.values()) else "MEDIUM"
        
        # Try to determine a more specific category
        category = "General Request"
        if assigned_group == "IT_SUPPORT":
            if any(word in text for word in ["password", "login", "access"]):
                category = "Access Issue"
            elif any(word in text for word in ["software", "install", "update", "application"]):
                category = "Software Request"
            elif any(word in text for word in ["hardware", "computer", "laptop", "printer"]):
                category = "Hardware Issue"
            elif any(word in text for word in ["network", "wifi", "internet", "connection"]):
                category = "Network Issue"
        
        logger.info(f"Classified request as: Group={assigned_group}, Priority={priority}, Category={category}")
        return assigned_group, priority, category
    
    def calculate_sla_due_date(self, assigned_group, priority):
        """Calculate SLA due date based on group and priority"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT default_sla_hours FROM groups WHERE group_name = ?", (assigned_group,))
        result = cursor.fetchone()
        
        base_hours = result[0] if result else 48  # Default to 48 hours if group not found
        
        # Adjust hours based on priority
        priority_multiplier = {
            "HIGH": 0.5,   # Half the standard time
            "MEDIUM": 1.0, # Standard time
            "LOW": 1.5     # 1.5x the standard time
        }
        
        sla_hours = base_hours * priority_multiplier.get(priority, 1.0)
        
        # Calculate due date
        now = datetime.now()
        sla_due_date = now.timestamp() + (sla_hours * 3600)  # Convert hours to seconds
        
        return datetime.fromtimestamp(sla_due_date)
    
    def process_emails(self):
        """Process incoming emails and create service requests"""
        emails = self.fetch_emails()
        
        for msg in emails:
            subject = msg['subject'] or "No Subject"
            sender = msg['from']
            recipient = msg['to']
            message_id = msg['message-id']
            
            # Get email body
            body = self.get_email_body(msg)
            
            # Check if this is a reply to an existing request
            cursor = self.db_conn.cursor()
            if "Re:" in subject:
                # Extract request ID from subject if it exists
                request_match = re.search(r'\[([A-Za-z0-9-]+)\]', subject)
                
                if request_match:
                    request_id = request_match.group(1)
                    cursor.execute("SELECT * FROM service_requests WHERE request_id = ?", (request_id,))
                    existing_request = cursor.fetchone()
                    
                    if existing_request:
                        # Update existing request
                        self.add_message_to_request(request_id, message_id, sender, body, datetime.now())
                        self.update_request_status(request_id, "UPDATED")
                        logger.info(f"Updated existing request: {request_id}")
                        continue
            
            # Classify the new request
            assigned_group, priority, category = self.classify_request(subject, body)
            
            # Calculate SLA due date
            sla_due_date = self.calculate_sla_due_date(assigned_group, priority)
            
            # Create new service request
            request_id = self.generate_request_id()
            now = datetime.now()
            
            # Insert new request into database
            cursor.execute('''
            INSERT INTO service_requests 
            (request_id, subject, sender, recipient, status, priority, created_at, updated_at, 
             assigned_group, assigned_to, category, sla_due_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (request_id, subject, sender, recipient, "NEW", priority, now, now,
                 assigned_group, None, category, sla_due_date))
            
            # Add first message
            self.add_message_to_request(request_id, message_id, sender, body, now)
            
            # Send confirmation email
            self.send_confirmation_email(sender, request_id, subject, assigned_group)
            
            # Notify the assigned group
            self.notify_group(assigned_group, request_id, subject, sender, priority, category)
            
            self.db_conn.commit()
            logger.info(f"Created new service request: {request_id}, assigned to {assigned_group}")
    
    def get_email_body(self, msg):
        """Extract email body content from email message"""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    return part.get_payload(decode=True).decode()
        else:
            return msg.get_payload(decode=True).decode()
        
        return "No content found"
    
    def add_message_to_request(self, request_id, message_id, sender, content, timestamp):
        """Add a message to an existing service request"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
        INSERT INTO messages 
        (request_id, message_id, sender, content, timestamp)
        VALUES (?, ?, ?, ?, ?)
        ''', (request_id, message_id, sender, content, timestamp))
        self.db_conn.commit()
    
    def update_request_status(self, request_id, status):
        """Update the status of a service request"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
        UPDATE service_requests 
        SET status = ?, updated_at = ?
        WHERE request_id = ?
        ''', (status, datetime.now(), request_id))
        self.db_conn.commit()
    
    def reassign_request(self, request_id, new_group):
        """Reassign a service request to a different group"""
        cursor = self.db_conn.cursor()
        
        # Get current request details
        cursor.execute("SELECT priority, status FROM service_requests WHERE request_id = ?", (request_id,))
        result = cursor.fetchone()
        
        if not result:
            logger.error(f"Request {request_id} not found for reassignment")
            return False
        
        priority, status = result
        
        # Calculate new SLA due date
        new_sla_due_date = self.calculate_sla_due_date(new_group, priority)
        
        # Update the request
        cursor.execute('''
        UPDATE service_requests 
        SET assigned_group = ?, assigned_to = NULL, updated_at = ?, sla_due_date = ?, status = ?
        WHERE request_id = ?
        ''', (new_group, datetime.now(), new_sla_due_date, "REASSIGNED", request_id))
        
        # Add system message about reassignment
        message_id = f"<system-{int(time.time())}@{self.config['email'].split('@')[1]}>"
        self.add_message_to_request(
            request_id, 
            message_id, 
            "SYSTEM", 
            f"Request reassigned to {new_group} group.", 
            datetime.now()
        )
        
        self.db_conn.commit()
        
        # Notify the new group
        cursor.execute("SELECT subject, sender, priority, category FROM service_requests WHERE request_id = ?", (request_id,))
        result = cursor.fetchone()
        if result:
            subject, sender, priority, category = result
            self.notify_group(new_group, request_id, subject, sender, priority, category)
        
        logger.info(f"Reassigned request {request_id} to {new_group}")
        return True
    
    def assign_to_agent(self, request_id, agent_name):
        """Assign a service request to a specific agent"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
        UPDATE service_requests 
        SET assigned_to = ?, status = ?, updated_at = ?
        WHERE request_id = ?
        ''', (agent_name, "ASSIGNED", datetime.now(), request_id))
        
        # Add system message about assignment
        message_id = f"<system-{int(time.time())}@{self.config['email'].split('@')[1]}>"
        self.add_message_to_request(
            request_id, 
            message_id, 
            "SYSTEM", 
            f"Request assigned to {agent_name}.", 
            datetime.now()
        )
        
        self.db_conn.commit()
        logger.info(f"Assigned request {request_id} to agent {agent_name}")
    
    def get_request(self, request_id):
        """Get service request details by request ID"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT * FROM service_requests WHERE request_id = ?", (request_id,))
        request = cursor.fetchone()
        
        if not request:
            return None
        
        cursor.execute("SELECT * FROM messages WHERE request_id = ? ORDER BY timestamp", (request_id,))
        messages = cursor.fetchall()
        
        return {
            "request": request,
            "messages": messages
        }
    
    def get_all_requests(self, status=None, group=None):
        """Get all service requests, optionally filtered by status and/or group"""
        cursor = self.db_conn.cursor()
        
        query = "SELECT * FROM service_requests"
        params = []
        conditions = []
        
        if status:
            conditions.append("status = ?")
            params.append(status)
        
        if group:
            conditions.append("assigned_group = ?")
            params.append(group)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY updated_at DESC"
        
        cursor.execute(query, params)
        return cursor.fetchall()
    
    def get_overdue_requests(self):
        """Get all service requests that have passed their SLA due date"""
        cursor = self.db_conn.cursor()
        now = datetime.now()
        
        cursor.execute('''
        SELECT * FROM service_requests 
        WHERE sla_due_date < ? 
        AND status NOT IN ('CLOSED', 'RESOLVED')
        ORDER BY sla_due_date
        ''', (now,))
        
        return cursor.fetchall()
    
    def send_confirmation_email(self, recipient, request_id, subject, assigned_group):
        """Send confirmation email for a new service request"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email']
            msg['To'] = recipient
            msg['Subject'] = f"[{request_id}] Service Request Created: {subject}"
            
            body = f"""
            Thank you for contacting our service desk.
            
            Your request has been received and a service request has been created with ID: {request_id}
            
            Your request has been assigned to our {assigned_group} team, and they will assist you as soon as possible.
            
            Please keep this request ID in your responses.
            
            This is an automated message. Please reply to this email if you need to provide additional information.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['email'], self.config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Sent confirmation email for request {request_id}")
            
        except Exception as e:
            logger.error(f"Error sending confirmation email: {str(e)}")
    
    def notify_group(self, group_name, request_id, subject, sender, priority, category):
        """Notify the assigned group about a new or reassigned service request"""
        try:
            # Get group email
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT email, members FROM groups WHERE group_name = ?", (group_name,))
            result = cursor.fetchone()
            
            if not result:
                logger.error(f"Group {group_name} not found for notification")
                return
            
            group_email, members = result
            
            msg = MIMEMultipart()
            msg['From'] = self.config['email']
            msg['To'] = group_email
            msg['Subject'] = f"[{request_id}] New {priority} Service Request: {subject}"
            
            body = f"""
            A new service request has been assigned to your group.
            
            Request ID: {request_id}
            From: {sender}
            Priority: {priority}
            Category: {category}
            Subject: {subject}
            
            Please respond to this request according to SLA guidelines.
            
            This is an automated message.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['email'], self.config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Sent group notification to {group_name} for request {request_id}")
            
        except Exception as e:
            logger.error(f"Error sending group notification: {str(e)}")
    
    def send_response(self, request_id, response_text, agent_name=None):
        """Send a response to a service request"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT * FROM service_requests WHERE request_id = ?", (request_id,))
        request = cursor.fetchone()
        
        if not request:
            logger.error(f"Request {request_id} not found")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email']
            msg['To'] = request[3]  # sender from request
            msg['Subject'] = f"[{request_id}] Re: {request[2]}"  # Re: subject
            
            signature = f"\n\nRegards,\n{agent_name if agent_name else 'Service Desk'}"
            
            msg.attach(MIMEText(response_text + signature, 'plain'))
            
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['email'], self.config['password'])
            server.send_message(msg)
            server.quit()
            
            # Add response to request messages
            message_id = f"<response-{int(time.time())}@{self.config['email'].split('@')[1]}>"
            sender = f"{agent_name} <{self.config['email']}>" if agent_name else self.config['email']
            self.add_message_to_request(request_id, message_id, sender, response_text, datetime.now())
            
            # Update request status
            self.update_request_status(request_id, "RESPONDED")
            
            logger.info(f"Sent response to request {request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending response: {str(e)}")
            return False
    
    def resolve_request(self, request_id, resolution_note=None, agent_name=None):
        """Mark a service request as resolved with optional resolution note"""
        # Add resolution note if provided
        if resolution_note:
            message_id = f"<resolution-{int(time.time())}@{self.config['email'].split('@')[1]}>"
            sender = f"{agent_name} <{self.config['email']}>" if agent_name else "SYSTEM"
            self.add_message_to_request(request_id, message_id, sender, resolution_note, datetime.now())
        
        # Update request status
        self.update_request_status(request_id, "RESOLVED")
        
        # Send resolution email to requester
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT sender, subject FROM service_requests WHERE request_id = ?", (request_id,))
        result = cursor.fetchone()
        
        if result:
            recipient, subject = result
            
            try:
                msg = MIMEMultipart()
                msg['From'] = self.config['email']
                msg['To'] = recipient
                msg['Subject'] = f"[{request_id}] Resolved: {subject}"
                
                body = f"""
                Your service request has been resolved.
                
                Request ID: {request_id}
                Subject: {subject}
                
                {resolution_note if resolution_note else ""}
                
                If you are satisfied with the resolution, no further action is required.
                If you need further assistance, please reply to this email and your request will be reopened.
                
                Thank you for using our service desk.
                
                Regards,
                {agent_name if agent_name else 'Service Desk'}
                """
                
                msg.attach(MIMEText(body, 'plain'))
                
                server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
                server.starttls()
                server.login(self.config['email'], self.config['password'])
                server.send_message(msg)
                server.quit()
                
            except Exception as e:
                logger.error(f"Error sending resolution email: {str(e)}")
        
        logger.info(f"Resolved request {request_id}")
    
    def close_request(self, request_id, closure_note=None):
        """Close a service request permanently"""
        if closure_note:
            message_id = f"<closure-{int(time.time())}@{self.config['email'].split('@')[1]}>"
            self.add_message_to_request(request_id, message_id, "SYSTEM", closure_note, datetime.now())
        
        self.update_request_status(request_id, "CLOSED")
        logger.info(f"Closed request {request_id}")
    
    def generate_sla_report(self):
        """Generate a report on SLA compliance"""
        cursor = self.db_conn.cursor()
        
        # Get all non-closed requests
        cursor.execute('''
        SELECT request_id, created_at, sla_due_date, status, priority, assigned_group 
        FROM service_requests 
        WHERE status NOT IN ('CLOSED', 'RESOLVED')
        ORDER BY sla_due_date
        ''')
        
        open_requests = cursor.fetchall()
        now = datetime.now()
        
        # Categorize requests
        overdue = []
        at_risk = []
        on_track = []
        
        for request in open_requests:
            request_id, created_at, sla_due_date, status, priority, assigned_group = request
            
            if isinstance(sla_due_date, str):
                sla_due_date = datetime.fromisoformat(sla_due_date)
            
            # Calculate time remaining in hours
            time_diff = (sla_due_date - now).total_seconds() / 3600
            
            if time_diff < 0:
                overdue.append((request_id, abs(time_diff), priority, assigned_group))
            elif time_diff < 4:  # Less than 4 hours remaining
                at_risk.append((request_id, time_diff, priority, assigned_group))
            else:
                on_track.append((request_id, time_diff, priority, assigned_group))
        
        return {
            "overdue": overdue,
            "at_risk": at_risk,
            "on_track": on_track,
            "total_open": len(open_requests),
            "report_time": now
        }
    
    def run_service(self, check_interval=300):
        """Run the service request system as a continuous service"""
        logger.info("Starting Service Request System")
        
        try:
            while True:
                # Process new emails
                self.process_emails()
                
                # Check for SLA violations
                sla_report = self.generate_sla_report()
                if sla_report["overdue"]:
                    logger.warning(f"SLA Alert: {len(sla_report['overdue'])} requests are overdue")
                    for request in sla_report["overdue"]:
                        logger.warning(f"Overdue: {request[0]} ({request[2]} priority) assigned to {request[3]}, {request[1]:.1f} hours past due")
                
                if sla_report["at_risk"]:
                    logger.info(f"SLA Warning: {len(sla_report['at_risk'])} requests at risk")
                    for request in sla_report["at_risk"]:
                        logger.info(f"At Risk: {request[0]} ({request[2]} priority) assigned to {request[3]}, {request[1]:.1f} hours remaining")
                
                # Sleep for the specified interval
                logger.info(f"Service sleeping for {check_interval} seconds")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("Service stopped by user")
        except Exception as e:
            logger.error(f"Service error: {str(e)}")
            # Try to restart the service
            logger.info("Attempting to restart service in 60 seconds")
            time.sleep(60)
            self.run_service(check_interval)


# Example usage
if __name__ == "__main__":
    config = {
        'email': 'servicedesk@example.com',
        'password': 'your_email_password',
        'imap_server': 'imap.example.com',
        'smtp_server': 'smtp.example.com',
        'smtp_port': 587,
        'database_file': 'service_requests.db'
    }
    
    service_system = ServiceRequestSystem(config)
    
    # Run as a continuous service
    service_system.run_service()

    # Alternatively, you can just process emails once
    # service_system.process_emails()
    
    # Or manually create a service request
    '''
    request_id = service_system.generate_request_id()
    now = datetime.now()
    
    cursor = service_system.db_conn.cursor()
    cursor.execute("""
    INSERT INTO service_requests 
    (request_id, subject, sender, recipient, status, priority, created_at, updated_at, 
     assigned_group, assigned_to, category, sla_due_date)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        request_id, 
        "Manual Test Request", 
        "user@example.com", 
        "servicedesk@example.com", 
        "NEW", 
        "MEDIUM", 
        now, 
        now,
        "IT_SUPPORT", 
        None, 
        "Test", 
        service_system.calculate_sla_due_date("IT_SUPPORT", "MEDIUM")
    ))
    
    service_system.add_message_to_request(
        request_id, 
        f"<manual-{int(time.time())}@example.com>", 
        "user@example.com", 
        "This is a test request created manually.", 
        now
    )
    
    service_system.db_conn.commit()
    print(f"Created manual test request: {request_id}")
