import React from 'react';
import { Box, Container } from '@mui/material';
import MultiExchange from '../components/MultiExchange';
import { useAuth } from '../contexts/AuthContext';
import { Navigate } from 'react-router-dom';

const MultiExchangePage: React.FC = () => {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return (
    <Container maxWidth={false}>
      <Box sx={{ py: 3 }}>
        <MultiExchange />
      </Box>
    </Container>
  );
};

export default MultiExchangePage;