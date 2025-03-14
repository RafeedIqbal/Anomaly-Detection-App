"use client";

import React, { createContext, useState, ReactNode } from "react";

interface CsvContextType {
  csvData: string | null;
  setCsvData: (data: string | null) => void;
}

export const CsvContext = createContext<CsvContextType>({
  csvData: null,
  setCsvData: () => {},
});

export const CsvProvider = ({ children }: { children: ReactNode }) => {
  const [csvData, setCsvData] = useState<string | null>(null);

  return (
    <CsvContext.Provider value={{ csvData, setCsvData }}>
      {children}
    </CsvContext.Provider>
  );
};
