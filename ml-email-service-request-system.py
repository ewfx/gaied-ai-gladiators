import email
import imaplib
import re
import json
import smtplib
import numpy as np
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class MLEmailServiceRequest:
    def __init__(self, config_file='config.json'):
        """Initialize the system with configuration"""
        self.config = self._load_config(config_file)
        self.email_client = None
        self.smtp_client = None
        self.group_model = self._load_or_create_group_model()
        self.priority_model = self._load_or_create_priority_model()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.group_encoder = LabelEncoder()
        self.priority_encoder = LabelEncoder()
        
        # Load or fit label encoders
        self._prepare_encoders()
    
    def _load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration if file not found
            return {
                "email": {
                    "imap_server": "imap.example.com",
                    "smtp_server": "smtp.example.com",
                    "port": 993,
                    "smtp_port": 587,
                    "username": "service@example.com",
                    "password": "password"
                },
                "ml": {
                    "group_model_path": "models/group_classifier.pkl",
                    "priority_model_path": "models/priority_classifier.pkl",
                    "retrain_threshold": 100,  # Retrain after this many new samples
                    "confidence_threshold": 0.7  # Minimum confidence for auto-assignment
                },
                "groups": [
                    "IT", "HR", "Facilities", "Finance", "Legal", "Marketing", "Sales"
                ],
                "priorities": [
                    "LOW", "NORMAL", "HIGH", "CRITICAL"
                ],
                "database": "service_requests.db",
                "training_data": "training_data.json"
            }
    
    def _preprocess_text(self, text):
        """Preprocess text for ML model input"""
        if not text:
            return ""
            
        # Convert to lowercase and remove non-alphanumeric
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return " ".join(processed_tokens)
    
    def _load_training_data(self):
        """Load training data for ML models"""
        try:
            with open(self.config['training_data'], 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Preprocess text
            df['processed_text'] = df.apply(
                lambda row: self._preprocess_text(row['subject'] + " " + row['body']), 
                axis=1
            )
            
            return df
        except (FileNotFoundError, json.JSONDecodeError):
            # Return empty DataFrame with required columns
            return pd.DataFrame(columns=[
                'id', 'subject', 'body', 'assigned_group', 'priority', 'processed_text'
            ])
    
    def _prepare_encoders(self):
        """Prepare label encoders for groups and priorities"""
        # Fit group encoder
        self.group_encoder.fit(self.config['groups'])
        
        # Fit priority encoder
        self.priority_encoder.fit(self.config['priorities'])
    
    def _load_or_create_group_model(self):
        """Load existing group classification model or create a new one"""
        try:
            with open(self.config['ml']['group_model_path'], 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError):
            # Create a new model
            return Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
    
    def _load_or_create_priority_model(self):
        """Load existing priority classification model or create a new one"""
        try:
            with open(self.config['ml']['priority_model_path'], 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError):
            # Create a new model
            return Pipeline([
                ('tfidf', TfidfVectorizer(max_features=3000)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
    
    def _save_models(self):
        """Save trained models to disk"""
        # Create directories if they don't exist
        import os
        os.makedirs(os.path.dirname(self.config['ml']['group_model_path']), exist_ok=True)
        
        # Save group model
        with open(self.config['ml']['group_model_path'], 'wb') as f:
            pickle.dump(self.group_model, f)
        
        # Save priority model
        with open(self.config['ml']['priority_model_path'], 'wb') as f:
            pickle.dump(self.priority_model, f)
    
    def train_models(self, force=False):
        """Train or retrain ML models with available data"""
        df = self._load_training_data()
        
        # Check if we have enough data
        if len(df) < 20 and not force:
            print("Insufficient training data. Need at least 20 examples.")
            return False
        
        print(f"Training models with {len(df)} examples")
        
        # Train group classification model
        if 'assigned_group' in df.columns and len(df) > 0:
            y_group = self.group_encoder.transform(df['assigned_group'])
            self.group_model.fit(df['processed_text'], y_group)
        
        # Train priority classification model
        if 'priority' in df.columns and len(df) > 0:
            y_priority = self.priority_encoder.transform(df['priority'])
            self.priority_model.fit(df['processed_text'], y_priority)
        
        # Save models
        self._save_models()
        return True
    
    def connect_email(self):
        """Connect to the email server"""
        try:
            self.email_client = imaplib.IMAP4_SSL(
                self.config['email']['imap_server'],
                self.config['email']['port']
            )
            self.email_client.login(
                self.config['email']['username'],
                self.config['email']['password']
            )
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def connect_smtp(self):
        """Connect to SMTP server for sending emails"""
        try:
            self.smtp_client = smtplib.SMTP(
                self.config['email']['smtp_server'],
                self.config['email']['smtp_port']
            )
            self.smtp_client.starttls()
            self.smtp_client.login(
                self.config['email']['username'],
                self.config['email']['password']
            )
            return True
        except Exception as e:
            print(f"SMTP connection error: {e}")
            return False
    
    def fetch_unread_emails(self):
        """Fetch unread emails from the inbox"""
        if not self.email_client:
            if not self.connect_email():
                return []
        
        self.email_client.select('INBOX')
        status, messages = self.email_client.search(None, 'UNSEEN')
        
        email_list = []
        if status == 'OK':
            for num in messages[0].split():
                status, data = self.email_client.fetch(num, '(RFC822)')
                if status == 'OK':
                    email_message = email.message_from_bytes(data[0][1])
                    email_list.append(email_message)
        
        return email_list
    
    def predict_group(self, subject, body):
        """Predict the appropriate group using ML model"""
        # Preprocess input text
        processed_text = self._preprocess_text(subject + " " + body)
        
        # Make prediction
        try:
            # Get probability distribution
            group_probs = self.group_model.predict_proba([processed_text])[0]
            
            # Get predicted class and its probability
            group_idx = np.argmax(group_probs)
            confidence = group_probs[group_idx]
            
            # Get group name
            predicted_group = self.group_encoder.inverse_transform([group_idx])[0]
            
            # Check if confidence is below threshold
            if confidence < self.config['ml']['confidence_threshold']:
                return "Needs_Review", confidence
            
            return predicted_group, confidence
        except:
            # If model fails or isn't trained yet
            return "Needs_Review", 0.0
    
    def predict_priority(self, subject, body):
        """Predict the priority using ML model"""
        # Preprocess input text
        processed_text = self._preprocess_text(subject + " " + body)
        
        # Make prediction
        try:
            # Get probability distribution
            priority_probs = self.priority_model.predict_proba([processed_text])[0]
            
            # Get predicted class and its probability
            priority_idx = np.argmax(priority_probs)
            confidence = priority_probs[priority_idx]
            
            # Get priority name
            predicted_priority = self.priority_encoder.inverse_transform([priority_idx])[0]
            
            # Check if confidence is below threshold
            if confidence < self.config['ml']['confidence_threshold']:
                return "NORMAL", confidence
            
            return predicted_priority, confidence
        except:
            # If model fails or isn't trained yet
            return "NORMAL", 0.0
    
    def create_service_request(self, email_message):
        """Create a service request from an email using ML predictions"""
        sender = email_message['From']
        subject = email_message['Subject'] or ""
        date = email_message['Date']
        
        body = ""
        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode()
                    except:
                        body = part.get_payload(decode=True).decode('latin-1', 'ignore')
                    break
        else:
            try:
                body = email_message.get_payload(decode=True).decode()
            except:
                body = email_message.get_payload(decode=True).decode('latin-1', 'ignore')
        
        # Use ML models to predict group and priority
        assigned_group, group_confidence = self.predict_group(subject, body)
        priority, priority_confidence = self.predict_priority(subject, body)
        
        # Generate request ID
        request_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        service_request = {
            "id": request_id,
            "sender": sender,
            "subject": subject,
            "date_received": date,
            "body": body,
            "priority": priority,
            "priority_confidence": float(priority_confidence),
            "assigned_group": assigned_group,
            "group_confidence": float(group_confidence),
            "status": "New",
            "created_at": datetime.now().isoformat(),
            "needs_review": assigned_group == "Needs_Review" or priority_confidence < self.config['ml']['confidence_threshold']
        }
        
        self._save_request(service_request)
        self._notify_group(service_request)
        self._confirm_receipt(service_request)
        
        # Add to training data
        self._add_to_training_data(service_request)
        
        return service_request
    
    def _save_request(self, service_request):
        """Save the service request to database"""
        # In a real implementation, this would save to an actual database
        # For this example, we'll just append to a JSON file
        try:
            with open(self.config['database'], 'r') as f:
                requests = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            requests = []
        
        requests.append(service_request)
        
        with open(self.config['database'], 'w') as f:
            json.dump(requests, f, indent=2)
    
    def _add_to_training_data(self, service_request):
        """Add a new request to training data for future model improvement"""
        training_item = {
            "id": service_request["id"],
            "subject": service_request["subject"],
            "body": service_request["body"],
            "assigned_group": service_request["assigned_group"],
            "priority": service_request["priority"]
        }
        
        try:
            with open(self.config['training_data'], 'r') as f:
                training_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            training_data = []
        
        # Only add to training data if confidence is high
        if (service_request['group_confidence'] >= self.config['ml']['confidence_threshold'] and
            service_request['priority_confidence'] >= self.config['ml']['confidence_threshold']):
            training_data.append(training_item)
            
            with open(self.config['training_data'], 'w') as f:
                json.dump(training_data, f, indent=2)
            
            # Check if we need to retrain models
            if len(training_data) % self.config['ml']['retrain_threshold'] == 0:
                print(f"Retraining models with {len(training_data)} examples")
                self.train_models()
    
    def _notify_group(self, service_request):
        """Notify the assigned group about the new service request"""
        if not self.smtp_client:
            if not self.connect_smtp():
                print("Failed to connect to SMTP server")
                return
        
        # In a real implementation, this would fetch group email addresses
        # from a user management system
        group_email = f"{service_request['assigned_group'].lower()}@example.com"
        
        msg = MIMEMultipart()
        msg['From'] = self.config['email']['username']
        msg['To'] = group_email
        
        confidence_info = ""
        if service_request['needs_review']:
            msg['Subject'] = f"New Service Request #{service_request['id']} - NEEDS REVIEW"
            confidence_info = f"\nML Confidence: Group ({service_request['group_confidence']:.2%}), Priority ({service_request['priority_confidence']:.2%})"
        else:
            msg['Subject'] = f"New Service Request #{service_request['id']} - {service_request['priority']} Priority"
        
        body = f"""
        A new service request has been assigned to your group:
        
        Request ID: {service_request['id']}
        Sender: {service_request['sender']}
        Subject: {service_request['subject']}
        Priority: {service_request['priority']}{confidence_info}
        
        Please review and assign to an agent.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            self.smtp_client.send_message(msg)
        except Exception as e:
            print(f"Failed to send notification: {e}")
    
    def _confirm_receipt(self, service_request):
        """Send confirmation of receipt to the requester"""
        if not self.smtp_client:
            if not self.connect_smtp():
                print("Failed to connect to SMTP server")
                return
        
        sender_email = re.search(r'<(.+?)>', service_request['sender'])
        if sender_email:
            sender_email = sender_email.group(1)
        else:
            sender_email = service_request['sender']
        
        msg = MIMEMultipart()
        msg['From'] = self.config['email']['username']
        msg['To'] = sender_email
        msg['Subject'] = f"Service Request Confirmation #{service_request['id']}"
        
        body = f"""
        Thank you for your service request. It has been received and assigned to our {service_request['assigned_group']} team.
        
        Request ID: {service_request['id']}
        Subject: {service_request['subject']}
        Priority: {service_request['priority']}
        
        Please reference this ID in any future communications regarding this request.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            self.smtp_client.send_message(msg)
        except Exception as e:
            print(f"Failed to send confirmation: {e}")
    
    def update_assignment(self, request_id, new_group, new_priority=None):
        """Update the assignment of a service request and use it for model improvement"""
        try:
            with open(self.config['database'], 'r') as f:
                requests = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        
        for i, request in enumerate(requests):
            if request['id'] == request_id:
                old_group = request['assigned_group']
                old_priority = request['priority']
                
                # Update group
                request['assigned_group'] = new_group
                request['needs_review'] = False
                
                # Update priority if provided
                if new_priority:
                    request['priority'] = new_priority
                
                # Save updated requests
                with open(self.config['database'], 'w') as f:
                    json.dump(requests, f, indent=2)
                
                # Update training data
                self._update_training_data(request_id, new_group, new_priority)
                
                # Send notification if group changed
                if old_group != new_group:
                    self._notify_group(request)
                
                return True
        
        return False
    
    def _update_training_data(self, request_id, new_group, new_priority=None):
        """Update training data with corrected assignment"""
        try:
            with open(self.config['training_data'], 'r') as f:
                training_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            training_data = []
        
        # Find if this request is already in training data
        request_found = False
        for item in training_data:
            if item['id'] == request_id:
                request_found = True
                item['assigned_group'] = new_group
                if new_priority:
                    item['priority'] = new_priority
                break
        
        # If not found, retrieve from database and add to training data
        if not request_found:
            try:
                with open(self.config['database'], 'r') as f:
                    requests = json.load(f)
                
                for request in requests:
                    if request['id'] == request_id:
                        training_item = {
                            "id": request["id"],
                            "subject": request["subject"],
                            "body": request["body"],
                            "assigned_group": new_group,
                            "priority": new_priority or request["priority"]
                        }
                        training_data.append(training_item)
                        break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        
        # Save updated training data
        with open(self.config['training_data'], 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Check if we need to retrain models
        if len(training_data) % self.config['ml']['retrain_threshold'] == 0:
            print(f"Retraining models with {len(training_data)} examples")
            self.train_models()
    
    def process_new_emails(self):
        """Process all unread emails and create service requests"""
        emails = self.fetch_unread_emails()
        requests = []
        
        for email_message in emails:
            request = self.create_service_request(email_message)
            requests.append(request)
        
        return requests
    
    def get_model_performance(self):
        """Get performance metrics for the ML models"""
        df = self._load_training_data()
        
        if len(df) < 20:
            return {
                "status": "Insufficient data",
                "training_samples": len(df)
            }
        
        # Split data for validation
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Preprocess text
        train_df['processed_text'] = train_df.apply(
            lambda row: self._preprocess_text(row['subject'] + " " + row['body']), 
            axis=1
        )
        test_df['processed_text'] = test_df.apply(
            lambda row: self._preprocess_text(row['subject'] + " " + row['body']), 
            axis=1
        )
        
        # Evaluate group model
        group_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        y_train_group = self.group_encoder.transform(train_df['assigned_group'])
        group_model.fit(train_df['processed_text'], y_train_group)
        
        y_test_group = self.group_encoder.transform(test_df['assigned_group'])
        y_pred_group = group_model.predict(test_df['processed_text'])
        
        group_accuracy = accuracy_score(y_test_group, y_pred_group)
        group_report = classification_report(
            y_test_group, 
            y_pred_group,
            target_names=self.group_encoder.classes_,
            output_dict=True
        )
        
        # Evaluate priority model
        priority_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        y_train_priority = self.priority_encoder.transform(train_df['priority'])
        priority_model.fit(train_df['processed_text'], y_train_priority)
        
        y_test_priority = self.priority_encoder.transform(test_df['priority'])
        y_pred_priority = priority_model.predict(test_df['processed_text'])
        
        priority_accuracy = accuracy_score(y_test_priority, y_pred_priority)
        priority_report = classification_report(
            y_test_priority, 
            y_pred_priority,
            target_names=self.priority_encoder.classes_,
            output_dict=True
        )
        
        return {
            "status": "Trained",
            "training_samples": len(df),
            "group_model": {
                "accuracy": group_accuracy,
                "report": group_report
            },
            "priority_model": {
                "accuracy": priority_accuracy,
                "report": priority_report
            }
        }
    
    def close_connections(self):
        """Close connections to email servers"""
        if self.email_client:
            self.email_client.logout()
        
        if self.smtp_client:
            self.smtp_client.quit()

# Example usage
if __name__ == "__main__":
    system = MLEmailServiceRequest()
    
    # Train models if they don't exist
    system.train_models()
    
    # Process new emails
    new_requests = system.process_new_emails()
    print(f"Processed {len(new_requests)} new service requests")
    
    # Get model performance
    performance = system.get_model_performance()
    print(f"Group classification accuracy: {performance.get('group_model', {}).get('accuracy', 'N/A')}")
    print(f"Priority classification accuracy: {performance.get('priority_model', {}).get('accuracy', 'N/A')}")
    
    # Close connections
    system.close_connections()
