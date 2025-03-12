#######
# Paste code below into your script to access our API/functions
#######
# import urllib.request
# api_url = 'https://raw.githubusercontent.com/tanmayyb/ele70_bv03/refs/heads/main/api/datasets.py'
# exec(urllib.request.urlopen(api_url).read())

import pandas as pd

# Dataloader for Energy Demand (IESO Zonal Ontario-wide)
def load_ieso_dataset( first_year:int, last_year:int, join:bool=False) -> dict|pd.DataFrame:
  # loads dataset from multiple years

  # assertion checks
  assert first_year <= last_year, "invalid entry for first/last year"
  assert first_year>=2003, "invalid entry for first_year, data before 2003 is N/A"
  assert last_year<=2024, "invalid entry for last year"

  # prefix and suffix defined for IESO api
  url_prefix = 'http://reports.ieso.ca/public/DemandZonal/PUB_DemandZonal_'
  url_suffix = '.csv'

  if first_year == last_year:
    # only for 1 year

    url = url_prefix+str(first_year)+url_suffix
    df = pd.read_csv(url, header=3) # returns record
    # add datetime
    df['DateTime'] = pd.to_datetime(df['Date'], utc=False)+pd.to_timedelta(df['Hour'], unit='h')

    return df

  else:
    # multiple years
    # decide dataset data type based on
    # if user wants a joined dataset
    if not join:
      dataset = dict()
    else:
      dataset = pd.DataFrame()

    # multiple year data
    for year in range(first_year, last_year+1):

      url = url_prefix+str(year)+url_suffix
      df = pd.read_csv(url, header=3) # returns record
      # add datetime
      df['DateTime'] = pd.to_datetime(df['Date'], utc=False)+pd.to_timedelta(df['Hour'], unit='h')

      if not join:
        dataset[year] = df
      else:
        dataset = pd.concat([dataset, df])

    return dataset

from tqdm import tqdm

# Dataloader for Weather (Canada-Wide)
def load_climate_dataset(first_year:int, last_year:int, station_id:int=31688, join:bool=False) -> dict|pd.DataFrame:

  # assertion checks
  assert first_year <= last_year, "invalid entry for first/last year"
  assert first_year>=2003, "invalid entry for first_year, please select >=2003"
  assert last_year<=2024, "invalid entry for last year"

  # hourly data for a given year, and station
  # f'https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&time=UTC&stationID=31688&Year=2008&Month={i}&timeframe=1&submit=Download+Data'

  num_years = last_year - first_year

  url_prefix = 'https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv'
  timeframe = 1

  # add 1 year data
  if num_years<=0:
    tmp = []
    for month in range(1, 12+1):
      url = url_prefix+f'&time=UTC&stationID={station_id}&Year={first_year}&Month={month}&timeframe={timeframe}&submit=Data'
      df = pd.read_csv(url)      
      tmp.append(df)
    df = pd.concat(tmp) # concat 12 months to 1 year df

    # add datetime
    df['DateTime'] = pd.to_datetime(df['Date/Time (UTC)'])

    return df

  # add multiple year data
  else:
    if not join:
      data = dict()
    else:
      data = pd.DataFrame()

    for year in tqdm(range(first_year, last_year+1)):
      tmp = []
      for month in range(1, 12+1):
        url = url_prefix+f'&time=UTC&stationID={station_id}&Year={year}&Month={month}&timeframe={timeframe}&submit=Data'
        tmp.append(pd.read_csv(url))
      df = pd.concat(tmp) # concat 12 months to 1 year df
      
      # add datetime
      df['DateTime'] = pd.to_datetime(df['Date/Time (UTC)'])

      # add year data
      if not join:
        data[year] = df
      else:
        data = pd.concat([data, df])
    return data
  

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import urllib
import concurrent.futures
import os
from tqdm import tqdm
import zipfile
import json


class Dataset():
  def __init__(self, region : str, country : str = "CA"):
    if country != "CA":
      raise ValueError("Currently only supports 'CA'")
    self.country = country.upper()
    self.region = region.upper()
    self.postal_codes = None
    self.cities = self.load_cities()

  def load_cities(self):
    if self.country == "CA":
      url = 'https://raw.githubusercontent.com/tanmayyb/ele70_bv03/refs/heads/main/api/canadacities.txt'
      df = pd.read_csv(url)

      # grab the cities
      cities = df[df["province_id"] == self.region]["city"].tolist()

      # grab the postal codes
      postal_codes = df[df["province_id"] == self.region]["postal"].tolist()
      self.postal_codes = [str(code).split() if pd.notna(code) else [] for code in postal_codes]
      return cities
    else:
      raise ValueError("Currently only supports 'CA'")



class IESODataset(Dataset):
  def __init__(self, dataset_type: str, region: str = "ON"):
    if region != "ON":
      raise ValueError("IESO dataset only supports Ontario ('ON') region")

    self.dataset_type = dataset_type.lower()
    if self.dataset_type not in ["zonal", "fsa"]:
      raise ValueError("Dataset type must be either 'zonal' or 'fsa'")

    super().__init__(region, "CA")

    self.dataset_name = f"ieso_{self.dataset_type}"
    self.data_dir = "./data/ieso"
    self.target_idx = None # user input
    self.target_name = None # auto-generated
    self.target_options = self.generate_target_options()
    self.target_val = None # auto-generated
    self.available_files, self.filetype, self.available_dates, self.date_type = self.get_metadata()
    self.selected_local_files, self.selected_dates = None, None
    self.default_filename = "ieso_dataset.json"

  def generate_target_options(self):
    # called by app to display region selection options
    if self.dataset_type == "zonal":
      self.target_options = ["Northwest", "Northeast", "Ottawa", "East", "Toronto", "Essa", "Bruce", "Southwest", "Niagara", "West", "Zone Total"]
    else:
      self.target_options = self.cities
    return self.target_options

  def get_target_options(self):
    return self.target_options

  def set_target(self, target_idx:int):
    self.target_idx = target_idx
    self.target_name = self.target_options[target_idx]
    if self.dataset_type == "zonal":
      self.target_val = self.target_options[target_idx]
    elif self.dataset_type == "fsa":
      self.target_val = self.postal_codes[target_idx]

  def get_metadata(self):
    if self.dataset_type == "zonal":
      base_url = "https://reports-public.ieso.ca/public/DemandZonal/"
      response = requests.get(base_url)
      soup = BeautifulSoup(response.text, 'html.parser')
      
      # Find all links on the page
      links = soup.find_all('a')
      
      # Filter for PUB_DemandZonal files and extract years
      files = []
      years = []
      for link in links:
          href = link.get('href')
          if href and 'PUB_DemandZonal_' in href and href.endswith('.csv'):
              # Extract year from filename
              match = re.search(r'PUB_DemandZonal_(\d{4})\.csv', href)
              if match:
                  year = int(match.group(1))
                  years.append(year)
                  files.append(base_url + href)
      
      # Sort both lists by year
      years.sort()
      files.sort()
      years = years[:-1]
      files = files[:-1]
      print(f"Available years: {min(years)} to {max(years)}")
      
      available_files = files
      filetype = "csv"
      dates = years
      date_type = "yearly"

    else:
      base_url = "https://reports-public.ieso.ca/public/HourlyConsumptionByFSA/"
      response = requests.get(base_url)
      soup = BeautifulSoup(response.text, 'html.parser')
      
      # Find all links on the page
      links = soup.find_all('a')
      
      # Filter for zip files
      files = []
      for link in links:
          href = link.get('href')
          if href and href.endswith('.zip'):
              files.append(base_url + href)

      dates = []
      for zip_file in files:
          # Extract YYYYMM from filename using regex
          match = re.search(r'_(\d{6})_', zip_file)
          if match:
              dates.append(match.group(1))

      if dates:
          dates.sort()
          print(f"Time range: {dates[0]} to {dates[-1]}")              
      
      available_files = files[1:]
      filetype = "zip"
      dates = dates
      date_type = "monthly"

    return available_files, filetype, dates, date_type

  def get_dates(self):
    return self.available_dates

  def download_dataset(self, start_date: int, end_date: int):
    if isinstance(start_date, str):
        start_date = int(start_date)
    if isinstance(end_date, str):
        end_date = int(end_date)
    if start_date > end_date:
      raise ValueError("Start date must be before or equal to end date")
    
    # Create data directory if it doesn't exist
    os.makedirs(self.data_dir, exist_ok=True)

    files = self.available_files
    file_type = self.filetype
    self.selected_dates = (start_date, end_date)

    # Filter files based on date range
    selected_files = [f for f, d in zip(files, self.available_dates) if start_date <= int(d) <= end_date]
    self.selected_files = selected_files

    if len(selected_files) == 0:
      raise ValueError("No files found for the given date range, please check the date range and try again.\
                        For FSA data, the date range should be in the format YYYYMM.\
                        For Zonal data, the date range should be in the format YYYY.")

    if file_type == "zip":
      self.download_zip_dataset(selected_files)
    elif file_type == "csv":
      self.download_csv_dataset(selected_files) 

  def download_zip_dataset(self, files):
    def download_and_extract(url, pbar):
        filename = url.split('/')[-1]
        extracted_files = []
        try:
            # Download
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            zip_path = os.path.join(self.data_dir, filename)
            
            # Download with progress
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Extract and store filenames
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
                extracted_files.extend(zip_ref.namelist())
            
            # Delete zip file
            os.remove(zip_path)
            
            # Store the extracted filenames as instance variable
            if not hasattr(self, 'extracted_filenames'):
                self.extracted_filenames = []
            self.extracted_filenames.extend(extracted_files)
            
            # Update progress bar
            pbar.update(1)
            return True
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            return False

    # Create progress bar for total operations
    with tqdm(total=len(files), desc="Downloading and extracting ZIP files") as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for url in files:
                futures.append(executor.submit(download_and_extract, url, pbar))
            concurrent.futures.wait(futures)
    
    self.selected_local_files = self.extracted_filenames
    self.selected_local_files.sort()
    del self.extracted_filenames

  def download_csv_dataset(self, files):
    def download_file(url, pbar):
        filename = url.split('/')[-1]
        try:
            # Download
            response = requests.get(url, stream=True)
            file_path = os.path.join(self.data_dir, filename)
            
            # Download with progress
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Store the filename as instance variable
            if not hasattr(self, 'downloaded_filenames'):
                self.downloaded_filenames = []
            self.downloaded_filenames.append(filename)
            
            # Update progress bar
            pbar.update(1)
            return True
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            return False

    # Create progress bar for total operations
    with tqdm(total=len(files), desc="Downloading CSV files") as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for url in files:
                futures.append(executor.submit(download_file, url, pbar))
            concurrent.futures.wait(futures)
    
    self.selected_local_files = self.downloaded_filenames
    self.selected_local_files.sort()
    del self.downloaded_filenames
  
  def parse_fsa_file(self, filepath, target_val, target_name):
    df = pd.read_csv(filepath, header=3)
    df = df[df['FSA'].isin(target_val)]
    df = df.groupby(['DATE', 'HOUR'])['TOTAL_CONSUMPTION'].sum().reset_index()
    df['DateTime'] = pd.to_datetime(df['DATE'], utc=False)+pd.to_timedelta(df['HOUR'], unit='h')
    df = df.rename(columns={'TOTAL_CONSUMPTION': target_name})
    df = df[['DateTime', target_name]]
    return df

  def parse_zonal_file(self, filepath, target_val, target_name):
    df = pd.read_csv(filepath, header=3)
    df['DateTime'] = pd.to_datetime(df['Date'], utc=False)+pd.to_timedelta(df['Hour'], unit='h')
    df = df[['DateTime', target_name]]
    return df

  def parse_dataset(self, chunk_size=4):
      """
      Parse FSA CSV files in sequential chunks and concatenate results
      
      Args:
          files (list): List of CSV filenames to parse
          chunk_size (int): Number of files to process in each chunk
      
      Returns:
          pd.DataFrame: Concatenated dataframe of all parsed files
      """
      files = self.selected_local_files
      data_dir  = self.data_dir
      target_val = self.target_val
      target_name = self.target_name
      # Check if target_val exists and is set
      if not hasattr(self, 'target_val') or self.target_val is None:
          raise ValueError("No target value set. Please call set_target() first.")

      # Split files into chunks
      file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
      
      results = []
      # Process chunks sequentially with progress bar
      for chunk in tqdm(file_chunks, desc="Processing chunks"):
          chunk_dfs = []
          for file in chunk:
              try:
                  filepath = os.path.join(data_dir, file)
                  if self.dataset_type == "zonal":
                    chunk_dfs.append(self.parse_zonal_file(filepath, target_val, target_name))
                  elif self.dataset_type == "fsa":
                    chunk_dfs.append(self.parse_fsa_file(filepath, target_val, target_name))
              except Exception as e:
                  print(f"Error processing {file}: {str(e)}")
          
          if chunk_dfs:
              results.append(pd.concat(chunk_dfs))
      
      return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
  
  def load_dataset(self, start_date: int=None, end_date: int=None, target_idx: int=None, download: bool = True, filepath: str=None, chunk_size: int = 4):
    if download:
      if start_date is None or end_date is None:
          raise ValueError("start_date and end_date must be provided")
      self.download_dataset(start_date, end_date)
      if target_idx is not None:
        self.set_target(target_idx)
        df = self.parse_dataset(chunk_size)
        self.df = df
        return df
    else:
      self.load_from_json(filepath)

  def save_dataset(self, filepath=None):
    if not hasattr(self, 'df'):
        raise ValueError("No dataset loaded to save. Call load_dataset first.")
        
    # Generate default filepath if none provided
    if filepath is None:
        filepath = os.path.join(self.data_dir, self.default_filename)
    
    # Prepare metadata and data as a single JSON
    df_to_save = self.df.copy()
    df_to_save['DateTime'] = df_to_save['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    combined_data = {
        'metadata': {
            'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_type': self.dataset_type,
            'target_name': self.target_name,
            'target_val': self.target_val,
            'date_range': self.selected_dates,
            'filetype': self.filetype,
            'files': self.selected_local_files,
            'column_types': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        },
        'data': df_to_save.to_dict(orient='records')
    }
    
    # Save combined metadata and data
    with open(filepath, 'w') as f:
        json.dump(combined_data, f, indent=2)

  def load_from_json(self, filepath=None):
    """
    Load dataset and metadata from a saved JSON file.
    
    Args:
        filepath (str, optional): Path to the JSON file. If None, uses default path.
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    # Use default filepath if none provided
    if filepath is None:
        filepath = os.path.join(self.data_dir, self.default_filename)
    
    # Load the JSON file
    with open(filepath, 'r') as f:
        combined_data = json.load(f)
    
    # Extract metadata and data
    metadata = combined_data['metadata']
    data = combined_data['data']
    
    # Restore metadata to class attributes
    self.dataset_type = metadata['dataset_type']
    self.target_name = metadata['target_name']
    self.target_val = metadata['target_val']
    self.selected_dates = metadata['date_range']
    self.filetype = metadata['filetype']
    self.selected_local_files = metadata['files']
    
    # Convert data back to DataFrame
    df = pd.DataFrame(data)
    
    # Convert DateTime back to datetime type
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Store the DataFrame
    self.df = df
    
    return df