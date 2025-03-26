import { Box, Button, Typography } from "@mui/material";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { useEffect, useState } from "react";
import TRequest from "../../types/TRequest";


const Details = () => {
    const location = useLocation();
    const { id } = useParams();
    console.log(location.state.row)
    const navigate = useNavigate();
    const [details, setDetails] = useState<Partial<TRequest>>();


    useEffect(() => {
        if(location.state?.row) {
            setDetails(location.state.row as Partial<TRequest>);
        } 
        else {
            setTimeout(() => {
                navigate("/");
            }, 1500);
        }
    }, [location]);

    const handleSupportSeek = () => {
        alert("Support seek request sent");
    };

    if (!details?.request_id) {
        return <Box>No ticket data available. Redirecting...</Box>;
    }

    return (
        <Box sx={{ p: 3 }}>
            <Typography variant="h4" gutterBottom>
                {`Request ${details.request_id} details`}
            </Typography>
            <Box sx={{ display: 'grid', gap: 2, maxWidth: 600, marginTop: 2 }}>
                <Typography><strong>Request Type:</strong> {details?.request_type}</Typography>
                <Typography><strong>Sub Request Type:</strong> {details?.sub_request_type}</Typography>
                <Typography><strong>Support Group:</strong> {details?.support_group}</Typography>
                <Typography><strong>Timestamp:</strong> {details?.timestamp}</Typography>
                <Typography><strong>Urgency:</strong> {details?.urgency}</Typography>
                <Typography><strong>Summary:</strong></Typography>
                <Box sx={{ 
                    backgroundColor: '#f5f5f5',
                    padding: 2,
                    borderRadius: 1,
                    color: 'black',
                    border: '1px solid #e0e0e0'
                }}>
                    {details?.summary?.split('\n').map((point, index) => (
                        <Box key={index} sx={{ marginLeft: 2 }}>
                            {point.replace(/\*/g, '').split(' ').map((word, wordIndex) => {
                                if (word.startsWith('*') && word.endsWith('*')) {
                                    return <strong key={wordIndex}>{word.replace(/\*/g, '')} </strong>;
                                }
                                return word + ' ';
                            })}
                        </Box>
                    ))}
                </Box>
            </Box>
            <Button
                sx={{ 
                    marginTop: 3, 
                    marginRight: 2,
                    borderRadius: 5, 
                    color: "black", 
                    backgroundColor: "white" 
                }}
                variant="contained"
                onClick={() => navigate("/")}
            >
                Go Back
            </Button>
            <Button
                sx={{ 
                    marginTop: 3, 
                    borderRadius: 5, 
                    color: "black", 
                    backgroundColor: "white" 
                }}
                variant="contained"
                disabled={details?.confidence === undefined || details?.confidence === null || details?.confidence >= 0.6}
                onClick={handleSupportSeek}
            >
                Seek Manual Support
            </Button>
        </Box>
    );
};

export default Details;
