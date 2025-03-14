"use client";

import React, { useState, useContext } from "react";
import { useRouter } from "next/navigation";
import {
  Box,
  Button,
  Grid,
  Typography,
  Paper,
  CircularProgress,
} from "@mui/material";
import axios from "axios";
import Image from "next/image";
import { CsvContext } from "@/context/CsvContext";

interface XgbResult {
  train_loss: number;
  test_loss: number;
  train_accuracy: number;
  test_accuracy: number;
  loss_curve: string;         // base64-encoded image
  performance_plot: string;   // base64-encoded image
}

interface LstmResult {
  train_loss: number;
  test_loss: number;
  train_accuracy: number;
  test_accuracy: number;
  train_loss_curve: string;       // base64-encoded image
  test_predictions_plot: string;  // base64-encoded image
}

export default function AnalysisPage() {
  const { csvData } = useContext(CsvContext);
  const [xgbResult, setXgbResult] = useState<XgbResult | null>(null);
  const [lstmResult, setLstmResult] = useState<LstmResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const router = useRouter();
  const targetColumn = "Toronto";

  const handleRunModels = async () => {
    if (!csvData) {
      setError("No CSV data found in context.");
      return;
    }
    setError("");
    setLoading(true);

    try {
      // Create a Blob from CSV data for FormData
      const blob = new Blob([csvData], { type: "text/csv" });

      const formDataXGB = new FormData();
      formDataXGB.append("file", blob, "merged_data.csv");
      formDataXGB.append("target", targetColumn);

      const formDataLSTM = new FormData();
      formDataLSTM.append("file", blob, "merged_data.csv");
      formDataLSTM.append("target", targetColumn);

      // Run both APIs in parallel
      const [xgbResp, lstmResp] = await Promise.all([
        axios.post("http://localhost:5000/xgb", formDataXGB, {
          headers: { "Content-Type": "multipart/form-data" },
        }),
        axios.post("http://localhost:5000/lstm", formDataLSTM, {
          headers: { "Content-Type": "multipart/form-data" },
        }),
      ]);

      setXgbResult(xgbResp.data);
      setLstmResult(lstmResp.data);
    } catch (err) {
      console.error(err);
      setError("Error running models. Check console or server logs.");
    } finally {
      setLoading(false);
    }
  };

  const handleHome = () => {
    router.push("/");
  };

  const handleBack = () => {
    router.back();
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        backgroundColor: "black",
        color: "white",
        p: 2,
      }}
    >
      <Typography variant="h5" sx={{ mb: 2 }}>
        #4 Analysis Dashboard
      </Typography>

      {error && (
        <Typography variant="body1" color="error" sx={{ mb: 2 }}>
          {error}
        </Typography>
      )}
      {loading && (
        <Box display="flex" justifyContent="center" mb={2}>
          <CircularProgress color="inherit" />
        </Box>
      )}

      <Button
        variant="contained"
        color="primary"
        onClick={handleRunModels}
        disabled={loading}
        sx={{ mb: 4 }}
      >
        Run Models
      </Button>

      <Grid container spacing={2} alignItems="stretch">
        {/* Left Column: All Graphs */}
        <Grid item xs={12} md={9}>

          {xgbResult && (
            <>
              <Paper
                sx={{
                  backgroundColor: "#333",
                  color: "white",
                  p: 2,
                  mb: 2,
                }}
              >
                <Typography variant="h6" sx={{ mb: 1 }}>
                  Model #1 (XGB) Loss Curve
                </Typography>
                <Image
                  src={`data:image/png;base64,${xgbResult.loss_curve}`}
                  alt="XGB Loss Curve"
                  unoptimized
                  width={800}
                  height={400}
                  style={{ width: "100%", maxHeight: "400px", objectFit: "contain" }}
                />
              </Paper>

              <Paper
                sx={{
                  backgroundColor: "#333",
                  color: "white",
                  p: 2,
                  mb: 2,
                }}
              >
                <Typography variant="h6" sx={{ mb: 1 }}>
                  Model #1 (XGB) Performance on Test Data
                </Typography>
                <Image
                  src={`data:image/png;base64,${xgbResult.performance_plot}`}
                  alt="XGB Performance Plot"
                  unoptimized
                  width={800}
                  height={400}
                  style={{ width: "100%", maxHeight: "400px", objectFit: "contain" }}
                />
              </Paper>
            </>
          )}

          {lstmResult && (
            <>
              <Paper
                sx={{
                  backgroundColor: "#333",
                  color: "white",
                  p: 2,
                  mb: 2,
                }}
              >
                <Typography variant="h6" sx={{ mb: 1 }}>
                  Model #2 (LSTM) Loss Curve
                </Typography>
                <Image
                  src={`data:image/png;base64,${lstmResult.train_loss_curve}`}
                  alt="LSTM Training Loss"
                  unoptimized
                  width={800}
                  height={400}
                  style={{ width: "100%", maxHeight: "400px", objectFit: "contain" }}
                />
              </Paper>

              <Paper
                sx={{
                  backgroundColor: "#333",
                  color: "white",
                  p: 2,
                  mb: 2,
                }}
              >
                <Typography variant="h6" sx={{ mb: 1 }}>
                  Model #2 (LSTM) Performance on Test Data
                </Typography>
                <Image
                  src={`data:image/png;base64,${lstmResult.test_predictions_plot}`}
                  alt="LSTM Test Predictions"
                  unoptimized
                  width={800}
                  height={400}
                  style={{ width: "100%", maxHeight: "400px", objectFit: "contain" }}
                />
              </Paper>

            </>
          )}
        </Grid>

        {/* Right Column: Metrics */}
        <Grid item xs={12} md={3}>
          {xgbResult && (
            <Paper
              sx={{
                backgroundColor: "#333",
                color: "white",
                p: 2,
                mb: 2,
              }}
            >
              <Typography variant="h6" sx={{ mb: 1 }}>
                Model #1 (XGB) Metrics
              </Typography>
              <Typography variant="body2">
                Train Loss: {xgbResult.train_loss.toFixed(3)}
              </Typography>
              <Typography variant="body2">
                Test Loss: {xgbResult.test_loss.toFixed(3)}
              </Typography>
              <Typography variant="body2">
                Train Accuracy: {xgbResult.train_accuracy.toFixed(3)}
              </Typography>
              <Typography variant="body2">
                Test Accuracy: {xgbResult.test_accuracy.toFixed(3)}
              </Typography>
            </Paper>
          )}

          {lstmResult && (
            <Paper
              sx={{
                backgroundColor: "#333",
                color: "white",
                p: 2,
              }}
            >
              <Typography variant="h6" sx={{ mb: 1 }}>
                Model #2 (LSTM) Metrics
              </Typography>
              <Typography variant="body2">
                Train Loss: {lstmResult.train_loss.toFixed(3)}
              </Typography>
              <Typography variant="body2">
                Test Loss: {lstmResult.test_loss.toFixed(3)}
              </Typography>
              <Typography variant="body2">
                Train Accuracy: {lstmResult.train_accuracy.toFixed(3)}
              </Typography>
              <Typography variant="body2">
                Test Accuracy: {lstmResult.test_accuracy.toFixed(3)}
              </Typography>
            </Paper>
          )}
        </Grid>
      </Grid>

      {/* Navigation Buttons */}
      <Box display="flex" justifyContent="space-between" mt={4}>
        <Button variant="outlined" color="inherit" onClick={handleHome}>
          HOME
        </Button>
        <Button variant="outlined" color="inherit" onClick={handleBack}>
          BACK
        </Button>
      </Box>
    </Box>
  );
}
