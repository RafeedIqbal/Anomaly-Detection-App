"use client";

import React, { useState } from "react";
import { Container, Paper, TextField, Button, Typography, Box } from "@mui/material";
import axios from "axios";
import { useRouter } from "next/navigation";

const Register: React.FC = () => {
  const [username, setUsername] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [confirmPassword, setConfirmPassword] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [success, setSuccess] = useState<string>("");
  const router = useRouter();

  const handleRegister = async () => {
    setError("");
    setSuccess("");

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    try {
      // Update the URL as needed to point to your Flask API endpoint.
      const response = await axios.post("http://localhost:5000/register", {
        username,
        password,
      });
      setSuccess("Registration successful! Redirecting to login...");
      // Redirect to login page after a short delay.
      setTimeout(() => {
        router.push("/login");
      }, 2000);
    } catch (err) {
      console.error(err);
      setError("Registration failed. Please try again.");
    }
  };

  const handleNavigateToLogin = () => {
    router.push("/login");
  };

  return (
    <Container maxWidth="sm">
      <Box
        display="flex"
        alignItems="center"
        justifyContent="center"
        minHeight="100vh"
        sx={{ padding: 2 }}
      >
        <Paper elevation={4} sx={{ padding: 4, width: "100%" }}>
          <Typography variant="h4" component="h1" gutterBottom align="center">
            Register
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
          <Box component="form" noValidate autoComplete="off">
            <TextField
              label="Username"
              variant="outlined"
              margin="normal"
              fullWidth
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
            <TextField
              label="Password"
              variant="outlined"
              margin="normal"
              type="password"
              fullWidth
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <TextField
              label="Confirm Password"
              variant="outlined"
              margin="normal"
              type="password"
              fullWidth
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
            />
            <Button
              variant="contained"
              color="primary"
              onClick={handleRegister}
              fullWidth
              sx={{ mt: 3 }}
            >
              Register
            </Button>
            <Button
              variant="outlined"
              color="secondary"
              onClick={handleNavigateToLogin}
              fullWidth
              sx={{ mt: 2 }}
            >
              Back to Login
            </Button>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default Register;
