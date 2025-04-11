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
import { CsvContext } from "@/context/CsvContext";
import Plot from "react-plotly.js";

// Updated: No base64 decoding needed
function decodePlotData(jsonString: string) {
  return JSON.parse(jsonString);
}

interface XgbResult {
  train_loss: number;
  test_loss: number;
  train_accuracy: number;
  test_accuracy: number;
  loss_curve: string;
  performance_plot: string;
  anomaly_csv: string;
  target_column_used: string;
}

interface LstmResult {
  train_loss: number;
  test_loss: number;
  train_accuracy: number;
  test_accuracy: number;
  loss_curve: string;
  performance_plot: string;
  anomaly_csv: string;
  target_column_used: string;
}

export default function AnalysisPage() {
  const { csvData, setCsvData } = useContext(CsvContext);
  const [xgbResult, setXgbResult] = useState<XgbResult | null>(null);
  const [lstmResult, setLstmResult] = useState<LstmResult | null>(null);
  const [loadingXgb, setLoadingXgb] = useState(false);
  const [loadingLstm, setLoadingLstm] = useState(false);
  const [error, setError] = useState("");

  const router = useRouter();

  const handleRunXGB = async () => {
    if (!csvData || !csvData.original) {
      setError("No CSV data found. Please generate data in the previous step.");
      return;
    }
    setError("");
    setLoadingXgb(true);
    setXgbResult(null);
    try {
      const blob = new Blob([csvData.original], { type: "text/csv" });
      const formData = new FormData();
      formData.append("file", blob, "input_data.csv");

      const response = await axios.post<XgbResult>("http://localhost:5000/xgb", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setXgbResult(response.data);
      setCsvData((prev) => ({ ...prev, xgb: response.data.anomaly_csv }));
    } catch (err: any) {
      console.error("XGBoost Error:", err);
      setError(err.response?.data?.error || err.message || "Error running XGBoost model.");
    } finally {
      setLoadingXgb(false);
    }
  };

  const handleRunLSTM = async () => {
    if (!csvData || !csvData.original) {
      setError("No CSV data found. Please generate data in the previous step.");
      return;
    }
    setError("");
    setLoadingLstm(true);
    setLstmResult(null);
    try {
      const blob = new Blob([csvData.original], { type: "text/csv" });
      const formData = new FormData();
      formData.append("file", blob, "input_data.csv");

      const response = await axios.post<LstmResult>("http://localhost:5000/lstm", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setLstmResult(response.data);
      setCsvData((prev) => ({ ...prev, lstm: response.data.anomaly_csv }));
    } catch (err: any) {
      console.error("LSTM Error:", err);
      setError(err.response?.data?.error || err.message || "Error running LSTM model.");
    } finally {
      setLoadingLstm(false);
    }
  };

  const handleNext = () => {
    if (!csvData?.xgb && !csvData?.lstm) {
      setError("Please run at least one model (XGBoost or LSTM) before proceeding to anomaly detection.");
      return;
    }
    setError("");
    router.push("/dataset/anomaly");
  };

  const handleBack = () => {
    router.back();
  };

  return (
    <Box sx={{ minHeight: "100vh", backgroundColor: "black", color: "white", p: 2, display: "flex", flexDirection: "column" }}>
      <Box sx={{ textAlign: "center", mb: 3 }}>
        <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
          #4 Model Training & Analysis
        </Typography>
        <Typography variant="body1" sx={{ color: 'grey.400' }}>
          Train XGBoost and LSTM models using the generated dataset.
        </Typography>
      </Box>

      {error && (
        <Typography variant="body1" color="error" sx={{ mb: 2, textAlign: 'center' }}>
          {error}
        </Typography>
      )}

      <Box sx={{ display: "flex", justifyContent: "center", gap: 3, mb: 4 }}>
        <Box sx={{ position: 'relative', display: 'inline-flex' }}>
          <Button variant="contained" color="primary" onClick={handleRunXGB} disabled={loadingXgb || loadingLstm} sx={{ fontSize: "1.1rem", p: "10px 20px" }}>
            Run XGBoost
          </Button>
          {loadingXgb && <CircularProgress size={24} color="inherit" sx={{ position: 'absolute', top: '50%', left: '50%', mt: '-12px', ml: '-12px' }} />}
        </Box>

        <Box sx={{ position: 'relative', display: 'inline-flex' }}>
          <Button variant="contained" color="secondary" onClick={handleRunLSTM} disabled={loadingXgb || loadingLstm} sx={{ fontSize: "1.1rem", p: "10px 20px" }}>
            Run LSTM
          </Button>
          {loadingLstm && <CircularProgress size={24} color="inherit" sx={{ position: 'absolute', top: '50%', left: '50%', mt: '-12px', ml: '-12px' }} />}
        </Box>
      </Box>

      <Grid container spacing={3} alignItems="stretch" sx={{ flexGrow: 1 }}>
        <Grid item xs={12} md={8}>
          <Grid container spacing={3}>
            {xgbResult && (
              <>
                <Grid item xs={12}>
                  <Paper sx={{ backgroundColor: "#222", color: "white", p: 2 }}>
                    <Typography variant="h6" sx={{ mb: 2, textAlign: 'center' }}>
                      XGBoost Loss Curve (Target: {xgbResult.target_column_used})
                    </Typography>
                    <Plot
                      data={decodePlotData(xgbResult.loss_curve)}
                      layout={{
                        paper_bgcolor: "#222",
                        plot_bgcolor: "#222",
                        font: { color: "white" },
                        xaxis: { title: 'Epoch' },
                        yaxis: { title: 'RMSE' }
                      }}
                      style={{ width: "100%", height: "400px" }}
                    />
                  </Paper>
                </Grid>

                <Grid item xs={12}>
                  <Paper sx={{ backgroundColor: "#222", color: "white", p: 2 }}>
                    <Typography variant="h6" sx={{ mb: 2, textAlign: 'center' }}>
                      XGBoost Performance Plot (Target: {xgbResult.target_column_used})
                    </Typography>
                    <Plot
                      data={decodePlotData(xgbResult.performance_plot)}
                      layout={{
                        paper_bgcolor: "#222",
                        plot_bgcolor: "#222",
                        font: { color: "white" },
                        xaxis: { title: 'Time' },
                        yaxis: { title: xgbResult.target_column_used }
                      }}
                      style={{ width: "100%", height: "400px" }}
                    />
                  </Paper>
                </Grid>
              </>
            )}

            {lstmResult && (
              <>
                <Grid item xs={12}>
                  <Paper sx={{ backgroundColor: "#222", color: "white", p: 2 }}>
                    <Typography variant="h6" sx={{ mb: 2, textAlign: 'center' }}>
                      LSTM Loss Curve (Target: {lstmResult.target_column_used})
                    </Typography>
                    <Plot
                      data={decodePlotData(lstmResult.loss_curve)}
                      layout={{
                        paper_bgcolor: "#222",
                        plot_bgcolor: "#222",
                        font: { color: "white" },
                        xaxis: { title: 'Epoch' },
                        yaxis: { title: 'RMSE' }
                      }}
                      style={{ width: "100%", height: "400px" }}
                    />
                  </Paper>
                </Grid>

                <Grid item xs={12}>
                  <Paper sx={{ backgroundColor: "#222", color: "white", p: 2 }}>
                    <Typography variant="h6" sx={{ mb: 2, textAlign: 'center' }}>
                      LSTM Performance Plot (Target: {lstmResult.target_column_used})
                    </Typography>
                    <Plot
                      data={decodePlotData(lstmResult.performance_plot)}
                      layout={{
                        paper_bgcolor: "#222",
                        plot_bgcolor: "#222",
                        font: { color: "white" },
                        xaxis: { title: 'Time' },
                        yaxis: { title: lstmResult.target_column_used }
                      }}
                      style={{ width: "100%", height: "400px" }}
                    />
                  </Paper>
                </Grid>
              </>
            )}
          </Grid>
        </Grid>

        <Grid item xs={12} md={4}>
          <Grid container spacing={3} direction="column">
            {xgbResult && (
              <Grid item>
                <Paper sx={{ backgroundColor: "#222", color: "white", p: 2 }}>
                  <Typography variant="h6">XGBoost Metrics</Typography>
                  <Typography>Train RMSE: {xgbResult.train_loss.toFixed(4)}</Typography>
                  <Typography>Test RMSE: <strong>{xgbResult.test_loss.toFixed(4)}</strong></Typography>
                  <Typography>Train R²: {xgbResult.train_accuracy.toFixed(4)}</Typography>
                  <Typography>Test R²: <strong>{xgbResult.test_accuracy.toFixed(4)}</strong></Typography>
                </Paper>
              </Grid>
            )}
            {lstmResult && (
              <Grid item>
                <Paper sx={{ backgroundColor: "#222", color: "white", p: 2 }}>
                  <Typography variant="h6">LSTM Metrics</Typography>
                  <Typography>Train RMSE: {lstmResult.train_loss.toFixed(4)}</Typography>
                  <Typography>Test RMSE: <strong>{lstmResult.test_loss.toFixed(4)}</strong></Typography>
                  <Typography>Train R²: {lstmResult.train_accuracy.toFixed(4)}</Typography>
                  <Typography>Test R²: <strong>{lstmResult.test_accuracy.toFixed(4)}</strong></Typography>
                </Paper>
              </Grid>
            )}
          </Grid>
        </Grid>
      </Grid>

      <Box display="flex" justifyContent="center" alignItems="center" mt="auto" pt={4} gap={3}>
        <Button variant="outlined" color="inherit" onClick={handleBack} size="large">
          BACK
        </Button>
        <Button variant="contained" color="success" onClick={handleNext} size="large" disabled={!csvData?.xgb && !csvData?.lstm}>
          NEXT: Anomalies
        </Button>
      </Box>
    </Box>
  );
}
