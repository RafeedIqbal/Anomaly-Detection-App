"use client";

import React from "react";
import { ThemeProvider, createTheme, CssBaseline } from "@mui/material";

const theme = createTheme({
  palette: {
    mode: "light",
    // Customize your palette here if needed
  },
});

interface MuiProviderProps {
  children: React.ReactNode;
}

export default function MuiProvider({ children }: MuiProviderProps) {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {children}
    </ThemeProvider>
  );
}
