"use client";

import React, { useState } from "react";
import { Container, Paper, TextField, Button, Typography, Box } from "@mui/material";
import axios from "axios";
import { useRouter } from "next/navigation";

const Login: React.FC = () => {
  const [username, setUsername] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [error, setError] = useState<string>("");
  const router = useRouter();

  const handleLogin = async () => {
    try {
      // Update the URL as needed to point to your Flask API endpoint.
      const response = await axios.post("http://localhost:5000/login", {
        username,
        password,
      });
      const { access_token } = response.data;
      // Save the token (using localStorage here; adjust as needed)
      localStorage.setItem("access_token", access_token);
      console.log("Login successful!");
      router.push("/dataset")
      // Optionally, navigate to a protected page after successful login
      // router.push("/profile");
    } catch (err) {
      console.error(err);
      setError("Invalid credentials. Please try again.");
    }
  };

  const handleNavigateToRegister = () => {
    router.push("/register");
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
            Login
          </Typography>
          {error && (
            <Typography variant="body1" color="error" align="center">
              {error}
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
            <Button
              variant="contained"
              color="primary"
              onClick={handleLogin}
              fullWidth
              sx={{ mt: 3 }}
            >
              Login
            </Button>
            <Button
              variant="outlined"
              color="secondary"
              onClick={handleNavigateToRegister}
              fullWidth
              sx={{ mt: 2 }}
            >
              Register
            </Button>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default Login;
