"use client";

import React, { useState, useContext } from "react";
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  CircularProgress,
} from "@mui/material";
import axios from "axios";
import { useRouter } from "next/navigation";
import { CsvContext } from "@/context/CsvContext";

const CreateDatasetPage: React.FC = () => {
  const { setCsvData } = useContext(CsvContext);
  const router = useRouter();

  // Form state
  const [energyRepo, setEnergyRepo] = useState<string>("IESO");
  const [predictorRepo, setPredictorRepo] = useState<string>("climate");
  const [targetZone, setTargetZone] = useState<string>("Toronto");
  const [firstYear, setFirstYear] = useState<string>("2010");
  const [lastYear, setLastYear] = useState<string>("2020");

  // UI state
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [success, setSuccess] = useState<string>("");

  const handleGenerateCSV = async () => {
    setError("");
    setSuccess("");
    setLoading(true);

    try {
      // Convert year inputs to numbers
      const first = parseInt(firstYear);
      const last = parseInt(lastYear);

      // POST request to generate CSV
      const response = await axios.post(
        "http://localhost:5000/generate_csv",
        {
          energy_repo: energyRepo,
          predictor_repo: predictorRepo,
          target_zone: targetZone,
          first_year: first,
          last_year: last,
        },
        { responseType: "text" } // Expect CSV text
      );

      // Store the CSV data in context so it can be used later
      setCsvData(response.data);
      setSuccess("CSV generated and stored successfully.");
    } catch (err: unknown) {
      if (err instanceof Error) {
        console.error(err.message);
      } else {
        console.error(err);
      }
      setError("Failed to generate CSV. Please check your inputs and try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleProceed = () => {
    // Navigate to the next page where the CSV data is used
    router.push("/dataset/next");
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
          p: 2,
        }}
      >
        <Paper elevation={4} sx={{ p: 4, width: "100%" }}>
          <Typography variant="h4" component="h1" gutterBottom align="center">
            Create Dataset
          </Typography>
          {error && (
            <Typography variant="body1" color="error" align="center">
              {error}
            </Typography>
          )}
          {success && (
            <Typography variant="body1" color="primary" align="center">
              {success}
            </Typography>
          )}
          <Box
            component="form"
            noValidate
            autoComplete="off"
            sx={{
              display: "flex",
              flexDirection: "column",
              gap: 2,
              mt: 2,
            }}
          >
            <TextField
              label="Energy Repository"
              variant="outlined"
              fullWidth
              value={energyRepo}
              onChange={(e) => setEnergyRepo(e.target.value)}
              helperText="Default: IESO"
              InputProps={{ sx: { backgroundColor: "white" } }}
            />
            <TextField
              label="Predictor Repository"
              variant="outlined"
              fullWidth
              value={predictorRepo}
              onChange={(e) => setPredictorRepo(e.target.value)}
              helperText="Default: climate"
              InputProps={{ sx: { backgroundColor: "white" } }}
            />
            <TextField
              label="Target Zone"
              variant="outlined"
              fullWidth
              value={targetZone}
              onChange={(e) => setTargetZone(e.target.value)}
              helperText="Default: Toronto"
              InputProps={{ sx: { backgroundColor: "white" } }}
            />
            <TextField
              label="First Year"
              variant="outlined"
              fullWidth
              value={firstYear}
              onChange={(e) => setFirstYear(e.target.value)}
              helperText="Default: 2010"
              InputProps={{ sx: { backgroundColor: "white" } }}
            />
            <TextField
              label="Last Year"
              variant="outlined"
              fullWidth
              value={lastYear}
              onChange={(e) => setLastYear(e.target.value)}
              helperText="Default: 2020"
              InputProps={{ sx: { backgroundColor: "white" } }}
            />

            <Button
              variant="contained"
              color="primary"
              onClick={handleGenerateCSV}
              disabled={loading}
              fullWidth
              sx={{ mt: 2 }}
            >
              {loading ? <CircularProgress size={24} color="inherit" /> : "Generate CSV"}
            </Button>
          </Box>
          {success && (
            <Button
              variant="outlined"
              color="secondary"
              onClick={handleProceed}
              fullWidth
              sx={{ mt: 2 }}
            >
              Proceed
            </Button>
          )}
        </Paper>
      </Box>
    </Container>
  );
};

export default CreateDatasetPage;
