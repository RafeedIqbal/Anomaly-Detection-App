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
  anomaly_csv: string;        // CSV string for anomaly detection
}

interface LstmResult {
  train_loss: number | null;
  test_loss: number;
  train_accuracy: number | null;
  test_accuracy: number;
  train_loss_curve: string;       // base64-encoded image
  test_predictions_plot: string;  // base64-encoded image
  anomaly_csv: string;            // CSV string for anomaly detection
}

export default function AnalysisPage() {
  const { csvData, setCsvData } = useContext(CsvContext);
  const [xgbResult, setXgbResult] = useState<XgbResult | null>(null);
  const [lstmResult, setLstmResult] = useState<LstmResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const router = useRouter();
  const targetColumn = "Toronto";

  // Handler for running only the XGBoost model
  const handleRunXGB = async () => {
    if (!csvData || !csvData.original) {
      setError("No CSV data found in context.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      // Use the CSV stored under the "original" key
      const blob = new Blob([csvData.original], { type: "text/csv" });
      const formData = new FormData();
      formData.append("file", blob, "merged_data.csv");
      formData.append("target", targetColumn);
      const response = await axios.post("http://localhost:5000/xgb", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setXgbResult(response.data);
      // Merge in the anomaly CSV from XGB into the context
      setCsvData((prev) => ({ ...prev, xgb: response.data.anomaly_csv }));
    } catch (err) {
      console.error(err);
      setError("Error running XGBoost model. Check console or server logs.");
    } finally {
      setLoading(false);
    }
  };

  // Handler for running only the LSTM model
  const handleRunLSTM = async () => {
    if (!csvData || !csvData.original) {
      setError("No CSV data found in context.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const blob = new Blob([csvData.original], { type: "text/csv" });
      const formData = new FormData();
      formData.append("file", blob, "merged_data.csv");
      formData.append("target", targetColumn);
      const response = await axios.post("http://localhost:5000/lstm", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setLstmResult(response.data);
      // Merge in the anomaly CSV from LSTM into the context
      setCsvData((prev) => ({ ...prev, lstm: response.data.anomaly_csv }));
    } catch (err) {
      console.error(err);
      setError("Error running LSTM model. Check console or server logs.");
    } finally {
      setLoading(false);
    }
  };

  const handleNext = () => {
    router.push("/dataset/anomaly");
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
      {/* Centered Header and Separate Run Buttons */}
      <Box sx={{ textAlign: "center", mb: 4 }}>
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
        <Box
          sx={{
            display: "flex",
            justifyContent: "center",
            gap: 2,
            mb: 4,
          }}
        >
          <Button
            variant="contained"
            color="primary"
            onClick={handleRunXGB}
            disabled={loading}
            sx={{ fontSize: "1.2rem", padding: "12px 24px" }}
          >
            Run XGBoost
          </Button>
          <Button
            variant="contained"
            color="secondary"
            onClick={handleRunLSTM}
            disabled={loading}
            sx={{ fontSize: "1.2rem", padding: "12px 24px" }}
          >
            Run LSTM
          </Button>
        </Box>
      </Box>

      <Grid container spacing={2} alignItems="stretch">
        {/* Left Column: Graphs */}
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
                  style={{
                    width: "100%",
                    maxHeight: "400px",
                    objectFit: "contain",
                  }}
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
                  style={{
                    width: "100%",
                    maxHeight: "400px",
                    objectFit: "contain",
                  }}
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
                  style={{
                    width: "100%",
                    maxHeight: "400px",
                    objectFit: "contain",
                  }}
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
                  style={{
                    width: "100%",
                    maxHeight: "400px",
                    objectFit: "contain",
                  }}
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
                Train Loss:{" "}
                {lstmResult.train_loss !== null
                  ? lstmResult.train_loss.toFixed(3)
                  : "N/A"}
              </Typography>
              <Typography variant="body2">
                Test Loss:{" "}
                {lstmResult.test_loss !== null
                  ? lstmResult.test_loss.toFixed(3)
                  : "N/A"}
              </Typography>
              <Typography variant="body2">
                Train Accuracy:{" "}
                {lstmResult.train_accuracy !== null
                  ? lstmResult.train_accuracy.toFixed(3)
                  : "N/A"}
              </Typography>
              <Typography variant="body2">
                Test Accuracy:{" "}
                {lstmResult.test_accuracy !== null
                  ? lstmResult.test_accuracy.toFixed(3)
                  : "N/A"}
              </Typography>
            </Paper>
          )}
        </Grid>
      </Grid>

      {/* Navigation Buttons */}
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        mt={4}
        gap={2}
      >
        <Button
          variant="outlined"
          color="inherit"
          onClick={handleBack}
          size="large"
          sx={{ fontSize: "1.2rem", padding: "12px 24px" }}
        >
          BACK
        </Button>
        <Button
          variant="outlined"
          color="inherit"
          onClick={handleNext}
          size="large"
          sx={{ fontSize: "1.2rem", padding: "12px 24px" }}
        >
          NEXT
        </Button>
      </Box>
    </Box>
  );
}
