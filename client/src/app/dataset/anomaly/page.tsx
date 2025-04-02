"use client";

import React, { useState } from "react";
import {
  Box,
  Button,
  Card,
  CardContent,
  Grid,
  Typography,
} from "@mui/material";

export default function AnomalyDetectionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string>("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("target", "Toronto");

    try {
      const res = await fetch("http://localhost:5000/anomaly_detection", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setResult(data);
        setError("");
      } else {
        setError(data.error || "An error occurred");
      }
    } catch (err) {
      setError("An error occurred");
    }
  };

  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h4" gutterBottom>
        Anomaly Detection
      </Typography>

      <form onSubmit={handleSubmit}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 2,
            flexWrap: "wrap",
            my: 2,
          }}
        >
          <Button
            variant="outlined"
            component="label"
            color="primary"
            sx={{ textTransform: "none" }}
          >
            📁 Upload CSV
            <input
              type="file"
              accept=".csv"
              hidden
              onChange={handleFileChange}
            />
          </Button>

          <Typography variant="body2" sx={{ minWidth: 200 }}>
            {file ? `Selected: ${file.name}` : "No file selected"}
          </Typography>

          <Button
            type="submit"
            variant="contained"
            color="primary"
            disabled={!file}
          >
            🚀 Run Detection
          </Button>
        </Box>
      </form>

      {error && (
        <Typography color="error" sx={{ mt: 2 }}>
          {error}
        </Typography>
      )}

      {result && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Number of anomalies detected: {result.num_anomalies}
          </Typography>

          <Card sx={{ mb: 4, p: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Top 10 Anomalies
            </Typography>
            <pre
              style={{
                background: "#111",
                color: "#0f0",
                padding: "1rem",
                borderRadius: "8px",
                overflowX: "auto",
              }}
            >
              {result.best_ten_anomalies}
            </pre>
          </Card>

          <Grid container spacing={3}>
            {[
              { label: "Scatter Plot", key: "scatter_plot" },
              { label: "Anomalies by Month", key: "month_plot" },
              { label: "Longest Anomalous Streak", key: "longest_anomalous_plot" },
              { label: "Longest Clean Streak", key: "longest_clean_plot" },
            ].map(
              (plot) =>
                result[plot.key] && (
                  <Grid item xs={12} md={6} key={plot.key}>
                    <Card elevation={3}>
                      <CardContent>
                        <Typography
                          variant="subtitle1"
                          gutterBottom
                          sx={{ fontWeight: 600 }}
                        >
                          {plot.label}
                        </Typography>
                        <Box
                          component="img"
                          src={`data:image/png;base64,${result[plot.key]}`}
                          alt={plot.label}
                          sx={{
                            width: "100%",
                            borderRadius: 2,
                            boxShadow: 2,
                          }}
                        />
                      </CardContent>
                    </Card>
                  </Grid>
                )
            )}
          </Grid>
        </Box>
      )}
    </Box>
  );
}
