import { lazy, Suspense } from 'react';
import './App.css'
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { CircularProgress } from '@mui/material';

const Details = lazy(() => import('./components/Details/Details'));
const Grid = lazy(() => import('./components/Grid/Grid'));

function App() {

  return (
    <BrowserRouter>
      <Suspense fallback={<CircularProgress/>}>
        <Routes>
          <Route path="/" element={<Grid />} />
          <Route path="/details/:id" element={<Details />} />
          <Route path="/details" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

export default App
