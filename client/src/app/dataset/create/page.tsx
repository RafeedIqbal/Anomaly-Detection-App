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
  MenuItem,
  FormControl,
  InputLabel,
  Select,
} from "@mui/material";
import axios from "axios";
import { useRouter } from "next/navigation";
import { CsvContext } from "@/context/CsvContext";

// Options for zonal dataset
const zonalOptions = [
  "Northwest",
  "Northeast",
  "Ottawa",
  "East",
  "Toronto",
  "Essa",
  "Bruce",
  "Southwest",
  "Niagara",
  "West",
  "Zone Total",
];

// Options for FSA dataset (loaded from message.txt)
const fsaOptions = [
  "Toronto", "Ottawa", "Hamilton", "Mississauga", "Brampton", "Kitchener", "London", "Markham", "Oshawa", "Vaughan",
  "Windsor", "St. Catharines", "Oakville", "Richmond Hill", "Burlington", "Sudbury", "Barrie", "Guelph", "Whitby",
  "Cambridge", "Milton", "Ajax", "Waterloo", "Thunder Bay", "Brantford", "Chatham", "Clarington", "Pickering",
  "Niagara Falls", "Newmarket", "Peterborough", "Kawartha Lakes", "Caledon", "Belleville", "Sarnia",
  "Sault Ste. Marie", "Welland", "Halton Hills", "Aurora", "North Bay", "Stouffville", "Cornwall", "Georgina",
  "Woodstock", "Quinte West", "St. Thomas", "New Tecumseth", "Innisfil", "Bradford West Gwillimbury", "Timmins",
  "Lakeshore", "Brant", "Leamington", "East Gwillimbury", "Orangeville", "Orillia", "Stratford", "Fort Erie",
  "LaSalle", "Centre Wellington", "Grimsby", "King", "Woolwich", "Clarence-Rockland", "Midland", "Lincoln",
  "Wasaga Beach", "Collingwood", "Strathroy-Caradoc", "Thorold", "Amherstburg", "Tecumseh", "Essa", "Owen Sound",
  "Brockville", "Kingsville", "Springwater", "Scugog", "Uxbridge", "Wilmot", "Essex", "Oro-Medonte", "Cobourg",
  "South Frontenac", "Port Colborne", "Huntsville", "Russell", "Niagara-on-the-Lake", "Middlesex Centre", "Selwyn",
  "Tillsonburg", "Pelham", "Petawawa", "North Grenville", "Loyalist", "Port Hope", "Pembroke", "Bracebridge",
  "Greater Napanee", "Kenora", "Mississippi Mills", "St. Clair", "West Lincoln", "West Nipissing / Nipissing Ouest",
  "Clearview", "Thames Centre", "Carleton Place", "Guelph/Eramosa", "Central Elgin", "Saugeen Shores",
  "Ingersoll", "South Stormont", "Severn", "South Glengarry", "North Perth", "Trent Hills", "The Nation / La Nation",
  "West Grey", "Gravenhurst", "Perth East", "Wellington North", "Brighton", "Tiny", "Hawkesbury", "Brock",
  "Erin", "Kincardine", "Elliot Lake", "Arnprior", "North Dundas", "Wellesley", "Georgian Bluffs", "Norwich",
  "Meaford", "Adjala-Tosorontio", "Hamilton Township", "South Dundas", "Lambton Shores", "North Dumfries",
  "Mapleton", "Rideau Lakes", "North Glengarry", "South Huron", "Penetanguishene", "Tay", "Cavan Monaghan",
  "Temiskaming Shores", "Grey Highlands", "Alfred and Plantagenet", "Elizabethtown-Kitley", "Smiths Falls",
  "Ramara", "Leeds and the Thousand Islands", "Brockton", "Laurentian Valley", "Mono", "Malahide", "Huron East",
  "Beckwith", "Shelburne", "West Perth", "Champlain", "Minto", "South Bruce Peninsula", "Renfrew", "Plympton-Wyoming",
  "Kapuskasing", "Zorra", "Kirkland Lake", "Aylmer", "Puslinch", "Drummond/North Elmsley", "Hanover", "Dryden",
  "Fort Frances", "Goderich", "Stone Mills", "South-West Oxford", "Douro-Dummer", "McNab/Braeside",
  "Central Huron", "Blandford-Blenheim", "Bayham", "Augusta", "St. Marys", "Southgate", "Bluewater",
  "East Zorra-Tavistock", "Huron-Kinloss", "The Blue Mountains", "Whitewater Region", "Edwardsburgh/Cardinal",
  "Wainfleet", "North Stormont", "Alnwick/Haldimand", "Arran-Elderslie", "Parry Sound", "Muskoka Falls",
  "Perth", "Cramahe", "North Middlesex", "Dysart et al", "Hindon Hill", "Tweed", "Oliver Paipoonge", "Petrolia",
  "Southwest Middlesex", "Front of Yonge", "Tay Valley", "South Bruce", "Ashfield-Colborne-Wawanosh", "Trent Lakes",
  "Gananoque", "Lanark Highlands", "Cochrane", "Sioux Lookout", "Hearst", "Breslau", "Stirling-Rawdon",
  "Espanola", "West Elgin", "East Ferris", "North Huron", "Southwold", "Centre Hastings", "Lucan Biddulph",
  "Greenstone", "Tyendinaga", "Iroquois Falls", "Havelock-Belmont-Methuen", "Central Frontenac", "Seguin",
  "Madawaska Valley", "Deep River", "Asphodel-Norwood", "Red Lake", "Hastings Highlands", "Prescott",
  "Northern Bruce Peninsula", "Casselman", "Callander", "Amaranth", "Marmora and Lake", "Bancroft", "Howick",
  "Dutton/Dunwich", "Perth South", "Montague", "Warwick", "Bonnechere Valley", "Morris-Turnberry", "Mulmur",
  "Blind River", "Powassan", "Highlands East", "East Hawkesbury", "Marathon", "Shuniah", "Sables-Spanish Rivers",
  "Lake of Bays", "Merrickville", "Adelaide-Metcalfe", "Melancthon", "Laurentian Hills", "Grand Valley",
  "Admaston/Bromley", "North Algona Wilberforce", "Wawa", "Horton", "Enniskillen", "Atikokan", "Markstay",
  "Northeastern Manitoulin and the Islands", "McDougall", "French River / Rivière des Français", "East Garafraxa",
  "Greater Madawaska", "Georgian Bay", "North Kawartha", "Perry", "Black River-Matheson", "Killaloe, Hagarty and Richards",
  "Alvinston", "Algonquin Highlands", "Addington Highlands", "Neebing", "Bonfield", "Central Manitoulin", "Madoc",
  "Mattawa", "Dawn-Euphemia", "Chapleau", "Manitouwadge", "Wellington", "Frontenac Islands", "Point Edward",
  "North Frontenac", "Komoka", "Deseronto", "Nipissing", "Huron Shores", "Nipigon", "Burford", "Terrace Bay",
  "Macdonald, Meredith and Aberdeen Additional", "Brudenell, Lyndoch and Raglan", "Moosonee", "Englehart",
  "Strong", "Lappe", "Armour", "Faraday", "Bayfield", "St.-Charles", "Emo", "Smooth Rock Falls", "Chisholm",
  "Thessalon", "Conestogo", "St. Joseph", "Moonbeam", "Claremont", "Ignace", "Armstrong", "Hillsburgh",
  "Sagamok", "Hensall", "Carling", "Laird", "Tara", "Cobalt", "South River", "McKellar", "South Algonquin",
  "Sioux Narrows-Nestor Falls", "Beachburg", "Schreiber", "Plantagenet", "Papineau-Cameron", "Assiginack", "Prince",
  "Athens", "Chatsworth", "Magnetawan"
];

const CreateDatasetPage: React.FC = () => {
  const { setCsvData } = useContext(CsvContext);
  const router = useRouter();

  // Form state
  const [datasetType, setDatasetType] = useState<string>("FSA");
  const [repository, setRepository] = useState<string>("climate");
  const [target, setTarget] = useState<string>(
    datasetType === "FSA" ? fsaOptions[0] : zonalOptions[0]
  );
  // Date fields for FSA (month inputs) and Zonal (year inputs)
  const [startMonthFSA, setStartMonthFSA] = useState<string>("2018-01");
  const [endMonthFSA, setEndMonthFSA] = useState<string>("2024-11");
  const [startYearZonal, setStartYearZonal] = useState<string>("2003");
  const [endYearZonal, setEndYearZonal] = useState<string>("2024");

  // UI state
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [success, setSuccess] = useState<string>("");

  // When datasetType changes, reset the target field to the first available option.
  const handleDatasetTypeChange = (
    event: React.ChangeEvent<{ value: unknown }>
  ) => {
    const newType = event.target.value as string;
    setDatasetType(newType);
    if (newType === "FSA") {
      setTarget(fsaOptions[0]);
    } else {
      setTarget(zonalOptions[0]);
    }
  };

  const handleGenerateCSV = async () => {
    setError("");
    setSuccess("");
    setLoading(true);
    try {
      let postData: any = {
        dataset_type: datasetType,
        predictor_repo: repository,
      };

      if (datasetType === "FSA") {
        // For FSA, split the month fields into year and month
        const [sYear, sMonth] = startMonthFSA.split("-");
        const [eYear, eMonth] = endMonthFSA.split("-");
        postData = {
          ...postData,
          target_city: target,
          start_year: parseInt(sYear),
          start_month: parseInt(sMonth),
          end_year: parseInt(eYear),
          end_month: parseInt(eMonth),
        };
      } else {
        // For Zonal, use year inputs
        postData = {
          ...postData,
          target_zone: target,
          start_year: parseInt(startYearZonal),
          end_year: parseInt(endYearZonal),
        };
      }

      const response = await axios.post(
        "http://localhost:5000/generate_csv",
        postData,
        { responseType: "text" } // Expect CSV text
      );
      // Save the generated CSV in context under "original"
      setCsvData({ original: response.data });
      setSuccess("CSV generated and stored successfully.");
    } catch (err: unknown) {
      console.error(err);
      setError("Failed to generate CSV. Please check your inputs and try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleProceed = () => {
    router.push("/dataset/analysis");
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
            {/* 1. Dataset Type */}
            <FormControl fullWidth>
              <InputLabel id="dataset-type-label">Type</InputLabel>
              <Select
                labelId="dataset-type-label"
                value={datasetType}
                label="Type"
                onChange={handleDatasetTypeChange}
              >
                <MenuItem value="FSA">FSA</MenuItem>
                <MenuItem value="Zonal">Zonal</MenuItem>
              </Select>
            </FormControl>

            {/* 2. Repository */}
            <TextField
              label="Repository"
              variant="outlined"
              fullWidth
              value={repository}
              onChange={(e) => setRepository(e.target.value)}
              helperText="Default: climate"
              InputProps={{ sx: { backgroundColor: "white" } }}
            />

            {/* 3. Target: City for FSA or Zone for Zonal */}
            {datasetType === "FSA" ? (
              <FormControl fullWidth>
                <InputLabel id="fsa-city-label">City</InputLabel>
                <Select
                  labelId="fsa-city-label"
                  value={target}
                  label="City"
                  onChange={(e) => setTarget(e.target.value as string)}
                >
                  {fsaOptions.map((city) => (
                    <MenuItem key={city} value={city}>
                      {city}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            ) : (
              <FormControl fullWidth>
                <InputLabel id="zonal-zone-label">Zone</InputLabel>
                <Select
                  labelId="zonal-zone-label"
                  value={target}
                  label="Zone"
                  onChange={(e) => setTarget(e.target.value as string)}
                >
                  {zonalOptions.map((zone) => (
                    <MenuItem key={zone} value={zone}>
                      {zone}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}

            {/* 4. Date Range Fields */}
            {datasetType === "FSA" ? (
              <>
                <TextField
                  label="Start Year/Month"
                  type="month"
                  fullWidth
                  value={startMonthFSA}
                  onChange={(e) => setStartMonthFSA(e.target.value)}
                  InputProps={{ inputProps: { min: "2018-01", max: "2024-11" } }}
                />
                <TextField
                  label="End Year/Month"
                  type="month"
                  fullWidth
                  value={endMonthFSA}
                  onChange={(e) => setEndMonthFSA(e.target.value)}
                  InputProps={{ inputProps: { min: "2018-01", max: "2024-11" } }}
                />
              </>
            ) : (
              <>
                <TextField
                  label="Start Year"
                  type="number"
                  fullWidth
                  value={startYearZonal}
                  onChange={(e) => setStartYearZonal(e.target.value)}
                  InputProps={{ inputProps: { min: 2003, max: 2024 } }}
                />
                <TextField
                  label="End Year"
                  type="number"
                  fullWidth
                  value={endYearZonal}
                  onChange={(e) => setEndYearZonal(e.target.value)}
                  InputProps={{ inputProps: { min: 2003, max: 2024 } }}
                />
              </>
            )}

            <Button
              variant="contained"
              color="primary"
              onClick={handleGenerateCSV}
              disabled={loading}
              fullWidth
              sx={{ mt: 2 }}
            >
              {loading ? (
                <CircularProgress size={24} color="inherit" />
              ) : (
                "Generate CSV"
              )}
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
