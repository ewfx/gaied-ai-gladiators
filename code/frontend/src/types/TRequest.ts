type Request = {
    request_id: string;
    timestamp: string;
    request_type: string;
    sub_request_type: string;
    support_group: string;
    urgency: string;
    confidence: number;
    summary: string;
};

export default Request;
