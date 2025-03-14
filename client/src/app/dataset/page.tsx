"use client";

import React from "react";
import { useRouter } from "next/navigation";
import { Box, Button, Typography } from "@mui/material";

export default function DatasetChoicePage() {
  const router = useRouter();

  const handleUpload = () => {
    router.push("/dataset/upload");
  };

  const handleCreate = () => {
    router.push("/dataset/create");
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        backgroundColor: "black",
        color: "white",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 4,
      }}
    >
      {/* Optional progress or header at the top */}
      {/* <LinearProgress variant="determinate" value={50} sx={{ width: "100%", position: "absolute", top: 0 }} /> */}
      
      <Typography variant="h5" component="h1">
        DATASET CHOICE
      </Typography>
      
      <Box
        sx={{
          display: "flex",
          flexDirection: { xs: "column", sm: "row" },
          alignItems: "center",
          justifyContent: "center",
          gap: 4,
        }}
      >
        <Button
          variant="contained"
          sx={{ minWidth: 200 }}
          onClick={handleUpload}
        >
          UPLOAD DATASET
        </Button>

        {/* A simple vertical divider can be added if desired:
            <Box sx={{ width: 2, height: "50px", backgroundColor: "white" }} />
        */}

        <Button
          variant="contained"
          sx={{ minWidth: 200 }}
          onClick={handleCreate}
        >
          CREATE NEW DATASET
        </Button>
      </Box>
    </Box>
  );
}
