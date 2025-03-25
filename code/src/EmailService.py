import os
import email
import extract_msg  # For .msg files
import re
import json
import io
import zipfile
import docx
import pdfplumber
import tempfile

def extract_text_from_attachment(attachment_content, attachment_name):
    """Extracts text from various attachment types."""
    try:
        if attachment_name.lower().endswith(".txt"):
            return attachment_content.decode()
        elif attachment_name.lower().endswith(".docx"):
            doc = docx.Document(io.BytesIO(attachment_content))
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return "\n".join(full_text)
        elif attachment_name.lower().endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(attachment_content)) as pdf:
                full_text = []
                for page in pdf.pages:
                    full_text.append(page.extract_text())
                return "\n".join(full_text)
        elif attachment_name.lower().endswith(".zip"):
            extracted_text = ""
            with zipfile.ZipFile(io.BytesIO(attachment_content), 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if not file_info.is_dir():
                        with zip_ref.open(file_info) as file:
                            file_content = file.read()
                            extracted_text += extract_text_from_attachment(file_content, file_info.filename) + "\n"
            return extracted_text
        else:
            return ""  # Unsupported attachment type

    except Exception as e:
        print(f"Error extracting text from attachment {attachment_name}: {e}")
        return ""

def extract_email_data(file_path):
    """Extracts email data and attachment content."""
    try:
        attachments = []
        if file_path.lower().endswith(".msg"):
            with extract_msg.Message(file_path) as msg:
                subject = msg.subject
                from_addr = msg.sender
                to_addr = ", ".join([str(recipient) for recipient in msg.to])
                body = msg.body
                for attachment in msg.attachments:
                    attachments.append({
                        "name": attachment.filename,
                        "content": attachment.data,
                    })

        elif file_path.lower().endswith(".eml"):
            with open(file_path, "rb") as f:
                msg = email.message_from_binary_file(f)
                subject = msg["Subject"]
                from_addr = msg["From"]
                to_addr = msg["To"]
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        if "text/plain" in content_type and "attachment" not in content_disposition:
                            body = part.get_payload(decode=True).decode()
                        elif "text/html" in content_type and "attachment" not in content_disposition:
                            body = part.get_payload(decode=True).decode()
                        elif "application/" in content_type or "image/" in content_type or "text/calendar" in content_type:
                            attachments.append({
                                "name": part.get_filename(),
                                "content": part.get_payload(decode=True),
                            })
                else:
                    body = msg.get_payload(decode=True).decode()

        else:
            return None

        email_data = {
            "subject": subject,
            "from": from_addr,
            "to": to_addr,
            "body": body,
            "attachments": attachments,
        }
        return email_data

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_keywords(email_data, keywords):
    """Extracts keyword-related content from email body and attachments."""
    if not email_data:
        return {}

    extracted_data = {}
    full_text = email_data["body"]
    for attachment in email_data.get("attachments", []):
        full_text += "\n" + extract_text_from_attachment(attachment["content"], attachment["name"])

    for keyword in keywords:
        matches = re.finditer(rf"(?i)(.{0,100}{keyword}.{{0,100}})", full_text)
        extracted_data[keyword] = [match.group(0).strip() for match in matches]

    return extracted_data

def process_email_folder(folder_path, keywords, output_file="extracted_data.json"):
    """Processes emails and saves results to a JSON file."""
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".msg", ".eml")):
            file_path = os.path.join(folder_path, filename)
            try:
                email_data = extract_email_data(file_path)
                if email_data:
                    keyword_data = extract_keywords(email_data, keywords)
                    results.append({
                        "file": filename,
                        "email_data": email_data,
                        "keyword_data": keyword_data,
                    })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    f.close()
    print(f"Extracted data saved to {output_file}")

# # Example usage:
# if __name__ == "__main__":
#     folder_path = "email_folder"
#     keywords = ["Problem", "Issue", "request", "Amount", "Ëxpiration Date", "Name", "Deal Name"]

#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)

#     process_email_folder(folder_path, keywords)
