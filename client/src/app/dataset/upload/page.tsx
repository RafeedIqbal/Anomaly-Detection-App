"use client";

import React, { useState, ChangeEvent, useContext } from "react";
import { Box, Button, Container, Paper, Typography } from "@mui/material";
import { useRouter } from "next/navigation";
import { CsvContext } from "@/context/CsvContext";

const DatasetUploadPage: React.FC = () => {
  const { setCsvData } = useContext(CsvContext);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<string>("");
  const router = useRouter();

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      if (file.type !== "text/csv" && !file.name.endsWith(".csv")) {
        setUploadError("Please upload a valid CSV file.");
        return;
      }
      setSelectedFile(file);
      setUploadError("");
    }
  };

  const handleUpload = () => {
    if (!selectedFile) {
      setUploadError("No file selected.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result;
      if (typeof text === "string") {
        // Store CSV data in context
        setCsvData(text);
        // Navigate to next page where CSV data can be used (adjust the route as needed)
        router.push("/models");
      } else {
        setUploadError("Error reading file.");
      }
    };
    reader.onerror = () => {
      setUploadError("Error reading file.");
    };

    reader.readAsText(selectedFile);
  };

  return (
    <Container maxWidth="sm">
      <Box
        sx={{
            minHeight: "100vh",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 4,
            backgroundColor: "black", // Set the background to black
        }}
      >
        <Paper elevation={4} sx={{ padding: 4, width: "100%" }}>
          <Typography variant="h4" component="h1" gutterBottom align="center">
            Upload Dataset
          </Typography>
          {uploadError && (
            <Typography variant="body1" color="error" align="center">
              {uploadError}
            </Typography>
          )}
          <Box
            component="div"
            sx={{
              display: "flex",
              flexDirection: "column",
              gap: 2,
              mt: 2,
            }}
          >
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={handleFileChange}
              style={{
                color: "white",
                backgroundColor: "black",
                border: "none",
              }}
            />
            <Button variant="contained" color="primary" onClick={handleUpload}>
              Upload CSV
            </Button>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default DatasetUploadPage;
