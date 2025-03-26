import { Box, Link, Typography } from '@mui/material';
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import TRequest from '../../types/TRequest';

const Grid = () => {
    const [rows, setRows] = useState<TRequest[]>([]);

    const navigate = useNavigate();
    const columns: GridColDef<(typeof rows)[number]>[] = [
        { field: 'request_id', headerName: 'Request ID', width: 150 },
        { field: 'request_type', headerName: 'Request Type', width: 200 },
        { field: 'sub_request_type', headerName: 'Sub Request Type', width: 200 },
        { field: 'support_group', headerName: 'Support Group', width: 200 },
        { field: 'urgency', headerName: 'Urgency', width: 100 },
        { field: 'confidence', headerName: 'Confidence', width: 150},
        { field: 'details',
          headerName: 'Details', 
          width: 250,
          renderCell: (params) => (
            <Link
                sx={{ fontWeight: 'bold'}}
                component="button"
                underline='hover'
                onClick={() => {
                    navigate(`/details/${params.row.request_id}`, {
                        state: {
                            row: params.row
                        }
                    })
                }}
            >
                Details
            </Link>
          )
        }
      ];
    
    const requestMethod: RequestInit['method'] = "GET";
    const url = `http://127.0.0.1:8000/tickets/`;


    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<Error | null>(null);
    const [data, setData] = useState<TRequest[]>([]);

    useEffect(() => {
      fetch(url, {
        method: requestMethod,
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
      })
        .then(response => {
          if (!response.ok) {
        throw new Error('Network response was not ok');
          }
          return response.json();
        })
        .then(data => setData(data))
        .catch(error => setError(error))
        .finally(() => setLoading(false));
    }, [url]);

    useEffect(() => {
        if (data && Array.isArray(data)) {
            setRows((data));
            console.log(data)
          }
      }, [data]);

    if(error) {
        return <Box>Error: {error.message}</Box>;
    }

    return(
        <>
            <Typography variant='h2' sx={{ padding: '1rem', textAlign: 'initial'}}>Service Email Tickets</Typography>
            <Box sx={{ height: 400, width: '100%' }}>
                <DataGrid
                    getRowId={(row) => row.request_id}
                    sx={{
                        '& .MuiDataGrid-cell': {
                          
                          color: 'white',
                          '@media (prefers-color-scheme: light)': {
                            color: 'black',
                          },
                        },
                      }}
                    rows={rows}
                    columns={columns}
                    disableRowSelectionOnClick
                    loading={loading}
                    disableColumnFilter
                    disableColumnSorting
                    
                />
            </Box>
        </>
    )
}

export default Grid;