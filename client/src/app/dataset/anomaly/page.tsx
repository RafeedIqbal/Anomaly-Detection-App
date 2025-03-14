"use client";

import React from 'react';
import Image from 'next/image';
import { Container, Grid, Box, Typography, Paper } from '@mui/material';

const GalleryPage: React.FC = () => {
  return (
    <Container sx={{ mt: 4 }}>
      {/* Header Paper */}
      <Paper elevation={3} sx={{ p: 2, mb: 4 }}>
        <Typography variant="h4" component="h1">
          Anomaly Detection Analysis
        </Typography>
      </Paper>

      {/* Image Grid Paper */}
      <Paper elevation={3} sx={{ p: 2 }}>
        <Grid container spacing={2}>
          {/* Image 0 */}
          <Grid item xs={12} sm={6}>
            <Box sx={{ position: 'relative', width: '100%', height: 300 }}>
              <Image
                src="/image.png"
                alt="Image"
                layout="fill"
                objectFit="contain"
              />
            </Box>
          </Grid>
          {/* Image 1 */}
          <Grid item xs={12} sm={6}>
            <Box sx={{ position: 'relative', width: '100%', height: 300 }}>
              <Image
                src="/image1.png"
                alt="Image 1"
                layout="fill"
                objectFit="contain"
              />
            </Box>
          </Grid>
          {/* Image 2 */}
          <Grid item xs={12} sm={6}>
            <Box sx={{ position: 'relative', width: '100%', height: 300 }}>
              <Image
                src="/image2.png"
                alt="Image 2"
                layout="fill"
                objectFit="contain"
              />
            </Box>
          </Grid>
          {/* Image 3 */}
          <Grid item xs={12} sm={6}>
            <Box sx={{ position: 'relative', width: '100%', height: 300 }}>
              <Image
                src="/image3.png"
                alt="Image 3"
                layout="fill"
                objectFit="contain"
              />
            </Box>
          </Grid>
        </Grid>
      </Paper>
    </Container>
  );
};

export default GalleryPage;