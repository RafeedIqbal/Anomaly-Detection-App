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
from io import BytesIO  # added for in-memory processing

def create_train_test_split(dataset: pd.DataFrame, target: str = None, split_coeff: float = 0.8, dt=None) -> tuple:
    training_cutoff = int(split_coeff * len(dataset))
    train = dataset.iloc[:training_cutoff]
    test = dataset.iloc[training_cutoff:]

    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]

    (train_idx, test_idx) = None, None
    if dt is not None:
        train_idx = dt[:training_cutoff]
        test_idx = dt[training_cutoff:]

    return (X_train, X_test, y_train, y_test), (train_idx, test_idx)

class Dataset():
    def __init__(self, region: str, country: str = "CA"):
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
        self.target_idx = None  # user input
        self.target_name = None  # auto-generated
        self.target_options = self.generate_target_options()
        self.target_val = None  # auto-generated
        self.available_files, self.filetype, self.available_dates, self.date_type = self.get_metadata()
        self.selected_local_files, self.selected_dates = None, None
        self.default_filename = "ieso_dataset.json"
        # For FSA in-memory processing
        self.in_memory_files = []

    def generate_target_options(self):
        # called by app to display region selection options
        if self.dataset_type == "zonal":
            self.target_options = ["Northwest", "Northeast", "Ottawa", "East", "Toronto", "Essa", "Bruce", "Southwest", "Niagara", "West", "Zone Total"]
        else:
            self.target_options = self.cities
        return self.target_options

    def get_target_options(self):
        return self.target_options

    def set_target(self, target_idx: int):
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
                print(f"Available Time range for IESO {self.dataset_type} dataset: {dates[0]} to {dates[-1]}")

            available_files = files[1:]
            filetype = "zip"
            dates = dates
            date_type = "monthly"

        return available_files, filetype, dates, date_type

    def get_dates(self):
        return self.available_dates

    def set_datetime(self, start_date: str, end_date: str):
        if len(start_date) == 4:
            start_dt = pd.to_datetime(start_date, format='%Y')
            end_dt = pd.to_datetime(end_date, format='%Y')
            end_dt = (end_dt + pd.offsets.YearEnd(0)).replace(hour=23, minute=0, second=0)
        else:
            start_dt = pd.to_datetime(start_date, format='%Y%m')
            end_dt = pd.to_datetime(end_date, format='%Y%m')
            end_dt = (end_dt + pd.offsets.MonthEnd(0)).replace(hour=23, minute=0, second=0)
        self.datetime_range = (start_dt, end_dt)

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
        """
        Download and process ZIP files in memory without saving to disk.
        """
        self.in_memory_files = []
        def download_and_extract(url, pbar):
            result = []
            try:
                response = requests.get(url)
                response.raise_for_status()
                zip_bytes = BytesIO(response.content)
                with zipfile.ZipFile(zip_bytes) as zip_ref:
                    for filename in zip_ref.namelist():
                        with zip_ref.open(filename) as file:
                            file_content = file.read()
                            result.append((filename, file_content))
                pbar.update(1)
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")
                pbar.update(1)
            return result

        with tqdm(total=len(files), desc="Downloading and processing ZIP files") as pbar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(download_and_extract, url, pbar) for url in files]
                concurrent.futures.wait(futures)
                for future in futures:
                    self.in_memory_files.extend(future.result())

    def download_csv_dataset(self, files):
        def download_file(url, pbar):
            filename = url.split('/')[-1]
            try:
                response = requests.get(url, stream=True)
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                if not hasattr(self, 'downloaded_filenames'):
                    self.downloaded_filenames = []
                self.downloaded_filenames.append(filename)
                pbar.update(1)
                return True
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                return False

        with tqdm(total=len(files), desc="Downloading CSV files") as pbar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for url in files:
                    futures.append(executor.submit(download_file, url, pbar))
                concurrent.futures.wait(futures)
        self.selected_local_files = self.downloaded_filenames
        self.selected_local_files.sort()
        del self.downloaded_filenames

    def parse_fsa_file(self, file_content, target_val, target_name):
        """
        Parse an FSA CSV file from in-memory content.
        """
        import io
        file_obj = io.BytesIO(file_content)
        df = pd.read_csv(file_obj, header=3)
        df = df[df['FSA'].isin(target_val)]
        df = df.groupby(['DATE', 'HOUR'])['TOTAL_CONSUMPTION'].sum().reset_index()
        df['DateTime'] = pd.to_datetime(df['DATE'], utc=False) + pd.to_timedelta(df['HOUR'], unit='h')
        df = df.rename(columns={'TOTAL_CONSUMPTION': target_name})
        df = df[['DateTime', target_name]]
        return df

    def parse_zonal_file(self, filepath, target_val, target_name):
        df = pd.read_csv(filepath, header=3)
        df['DateTime'] = pd.to_datetime(df['Date'], utc=False) + pd.to_timedelta(df['Hour'], unit='h')
        df = df[['DateTime', target_name]]
        return df

    def parse_dataset(self, chunk_size=4):
        """
        Parse dataset files in sequential chunks and concatenate results.
        For FSA datasets, process in-memory files.
        """
        target_val = self.target_val
        target_name = self.target_name
        if not hasattr(self, 'target_val') or self.target_val is None:
            raise ValueError("No target value set. Please call set_target() first.")

        results = []
        if self.dataset_type == "zonal":
            files = self.selected_local_files
            data_dir = self.data_dir
            file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
            for chunk in tqdm(file_chunks, desc="Processing chunks"):
                chunk_dfs = []
                for file in chunk:
                    try:
                        filepath = os.path.join(data_dir, file)
                        chunk_dfs.append(self.parse_zonal_file(filepath, target_val, target_name))
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")
                if chunk_dfs:
                    results.append(pd.concat(chunk_dfs))
        elif self.dataset_type == "fsa":
            files = self.in_memory_files
            file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
            for chunk in tqdm(file_chunks, desc="Processing in-memory chunks"):
                chunk_dfs = []
                for filename, file_content in chunk:
                    try:
                        chunk_dfs.append(self.parse_fsa_file(file_content, target_val, target_name))
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                if chunk_dfs:
                    results.append(pd.concat(chunk_dfs))
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def load_dataset(self, start_date: str = None, end_date: str = None, target_idx: int = None, download: bool = True, filepath: str = None, chunk_size: int = 4):
        if not isinstance(start_date, str) or not isinstance(end_date, str):
            raise ValueError("start_date and end_date must be strings!")

        if target_idx is None:
            target_idx = self.target_idx
        if download:
            if start_date is None or end_date is None:
                raise ValueError("start_date and end_date must be provided")

            print(f"Selected Time Range for IESO {self.dataset_type} dataset: {start_date} to {end_date}")

            # store datetime range       
            self.set_datetime(start_date, end_date)
            self.download_dataset(start_date, end_date)

            if target_idx is not None:
                self.set_target(target_idx)
                df = self.parse_dataset(chunk_size)
                df['DateTime'] = df['DateTime'] - pd.Timedelta(hours=1)
                df.set_index('DateTime', inplace=True)
                self.df = df
                return df
            else:
                print("No target value set. Please call set_target() first to generate a df.")
        else:
            self.load_from_json(filepath)

    def save_dataset(self, filepath=None):
        if not hasattr(self, 'df'):
            raise ValueError("No dataset loaded to save. Call load_dataset first.")

        if filepath is None:
            filepath = os.path.join(self.data_dir, self.default_filename)

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
                'files': self.selected_local_files if self.dataset_type == "zonal" else "in-memory",
                'column_types': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            },
            'data': df_to_save.to_dict(orient='records')
        }

        with open(filepath, 'w') as f:
            json.dump(combined_data, f, indent=2)

    def load_from_json(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.data_dir, self.default_filename)

        with open(filepath, 'r') as f:
            combined_data = json.load(f)

        metadata = combined_data['metadata']
        data = combined_data['data']

        self.dataset_type = metadata['dataset_type']
        self.target_name = metadata['target_name']
        self.target_val = metadata['target_val']
        self.selected_dates = metadata['date_range']
        self.filetype = metadata['filetype']
        self.selected_local_files = metadata['files']

        start_date = str(self.selected_dates[0])
        end_date = str(self.selected_dates[1])
        self.set_datetime(start_date, end_date)

        df = pd.DataFrame(data)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        self.df = df

        return df

from meteostat import Stations, Hourly
from geopy.geocoders import Nominatim
import geopandas as gpd
from shapely.geometry import Point
import random
from collections import OrderedDict

class ClimateDataset(Dataset):
    def __init__(self, iesodata: IESODataset, region: str = "ON"):
        super().__init__(region, "CA")
        self.dataset_type = "climate"
        self.data_dir = "./data/climate"
        self.default_filename = "climate_dataset.json"
        self.dataset_name = 'climate'

        self.ieso_dataset = iesodata
        if hasattr(self.ieso_dataset, 'target_name'):
            self.target_name = self.ieso_dataset.target_name
        else:
            raise ValueError("IESO dataset does not have target name")
        if self.target_name is None:
            raise ValueError("IESO dataset target name is not set")

        self.weather_station_ids = None
        self.ieso_dataset_type = self.ieso_dataset.dataset_type
        self.ieso_target_name = self.ieso_dataset.target_name
        self.ieso_target_val = self.ieso_dataset.target_val

        self.datetime_range = self.ieso_dataset.datetime_range
        self.selected_station_ids = []

    def load_dataset(self, sample_num: int = 5, sampling_seed: int = 42,
                     download: bool = True, filepath: str = None):

        self.sample_num = sample_num
        self.sampling_seed = sampling_seed

        if download:
            self.get_weather_stations()
            self.select_weather_stations()
            self.combine_station_data()
        else:
            self.load_from_json(filepath)

    def select_weather_stations(self):
        dataset_type = self.ieso_dataset_type
        stations = self.weather_stations
        sample_num = self.sample_num
        seed = self.sampling_seed
        num_available_stations = len(stations)
        station_ids = stations.index.to_list()

        sample_num = min(num_available_stations, sample_num)
        num_selected_stations = 0
        station_data = OrderedDict()

        start_dt, end_dt = self.datetime_range

        random.seed(seed)
        while num_selected_stations < sample_num:
            if dataset_type == 'zonal':
                station_id = random.choice(station_ids)
                station_ids.remove(station_id)
            elif dataset_type == 'fsa':
                station_id = station_ids.pop(0)

            df = self.load_station_data(station_id)
            is_good = self.perform_checks(df, start_dt, end_dt)
            if is_good:
                num_selected_stations += 1
                station_data[station_id] = df[(df.index >= start_dt) & (df.index <= end_dt)]

            if len(station_ids) == 0:
                break

        if num_selected_stations == 0:
            raise RuntimeError('WTF, no weather stations passed checks!')

        self.selected_station_ids = list(station_data.keys())
        self.station_data = station_data

    def perform_checks(self, df: pd.DataFrame, start_dt, end_dt) -> bool:
        if df.empty:
            return False

        if start_dt not in df.index or end_dt not in df.index:
            return False

        return True

    def load_station_data(self, station_id):
        df = Hourly(station_id).fetch()
        return df

    def combine_station_data(self):
        def add_prefix_to_columns(df, prefix):
            df = df.add_prefix(f"{prefix}_")
            df.rename(columns={f"{prefix}_time": "time"}, inplace=True)
            return df

        combined_df = None
        for station_id, df in self.station_data.items():
            df = add_prefix_to_columns(df, station_id)
            if combined_df is None:
                combined_df = df
            else:
                combined_df = combined_df.merge(df, on="time", how="outer")

        self.df = combined_df

    def get_weather_stations(self):
        dataset_type = self.ieso_dataset_type

        if dataset_type == 'zonal':
            self.get_zone_based_weather_stations()
        elif dataset_type == 'fsa':
            self.get_location_based_weather_stations()
        else:
            raise ValueError("Unknown dataset type, cannot resolve weather stations!")

    def get_zone_based_weather_stations(self):
        target = self.target_name
        geojson_url = 'https://raw.githubusercontent.com/tanmayyb/ele70_bv03/refs/heads/main/api/ieso_zones.geojson'

        zones = gpd.read_file(geojson_url).to_crs("EPSG:4326")
        zones.set_index('Name', inplace=True)
        def convert_to_tuple(coord_str):
            return tuple(map(float, coord_str.strip('()').split(', ')))
        zones['top_left'] = zones['top_left'].apply(convert_to_tuple)
        zones['bottom_right'] = zones['bottom_right'].apply(convert_to_tuple)

        top_left = zones.loc[target].top_left
        bottom_right = zones.loc[target].bottom_right
        zone_geometry = zones.loc[target].geometry
        def is_within_zone(station):
            point = Point(station['longitude'], station['latitude'])
            return point.within(zone_geometry)

        stations = Stations().region(self.country)
        stations = stations.bounds(top_left, bottom_right)
        stations = stations.fetch()
        filtered_stations = stations[stations.apply(is_within_zone, axis=1)]
        self.weather_stations = filtered_stations

    def get_location_based_weather_stations(self):
        target = self.target_name

        geolocator = Nominatim(user_agent="geoapi")
        location = geolocator.geocode(f"{target}, {self.region}")
        stations = Stations().region(self.country)
        stations = stations.nearby(location.latitude, location.longitude)
        stations = stations.fetch()
        self.weather_stations = stations

    def save_dataset(self, filepath=None):
        if not hasattr(self, 'df'):
            raise ValueError("No dataset loaded to save. Call load_dataset first.")

        if filepath is None:
            os.makedirs(self.data_dir, exist_ok=True)
            filepath = os.path.join(self.data_dir, self.default_filename)

        df_to_save = self.df.copy()
        df_to_save.reset_index(inplace=True)
        df_to_save.rename(columns={'time': 'DateTime'}, inplace=True)
        df_to_save['DateTime'] = df_to_save['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')

        combined_data = {
            'metadata': {
                'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_type': self.dataset_type,
                'column_types': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            },
            'data': df_to_save.to_dict(orient='records')
        }

        with open(filepath, 'w') as f:
            json.dump(combined_data, f, indent=2)

    def load_from_json(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.data_dir, self.default_filename)

        with open(filepath, 'r') as f:
            combined_data = json.load(f)

        metadata = combined_data['metadata']
        data = combined_data['data']

        self.dataset_type = metadata['dataset_type']
        self.target_name = metadata['target_name']
        self.target_val = metadata['target_val']
        self.selected_dates = metadata['date_range']
        self.filetype = metadata['filetype']
        self.selected_local_files = metadata['files']

        start_date = str(self.selected_dates[0])
        end_date = str(self.selected_dates[1])
        self.set_datetime(start_date, end_date)

        df = pd.DataFrame(data)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        self.df = df

        return df


class DatasetPreprocessor():
    def __init__(self, ieso_dataset: IESODataset, climate_dataset: ClimateDataset):
        self.ieso_dataset = ieso_dataset
        self.climate_dataset = climate_dataset
        self.target_name = ieso_dataset.target_name

    def preprocess(self, delete_leap_day: bool = False):
        df = self.ieso_dataset.df.merge(self.climate_dataset.df, left_index=True, right_index=True, how='inner')
        df = df.reset_index().rename(columns={'index': 'DateTime'})
        df['Y'] = df['DateTime'].dt.year
        df['M'] = df['DateTime'].dt.month
        df['D'] = df['DateTime'].dt.day
        df['H'] = df['DateTime'].dt.hour

        dt = df.pop('DateTime')

        if delete_leap_day:
            df = df[~((df.DateTime.dt.month == 2) & (df.DateTime.dt.day == 29))]

        df = df.fillna(df.mean())

        return self.target_name, df, dt

    def save_dataset(self, filepath: str = None):
        target_name, df, dt = self.preprocess()
        df['DateTime'] = dt

        metadata = {
            'target_name': target_name,
            'dataset_type': self.ieso_dataset.dataset_type,
            'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        with open(filepath, 'w') as f:
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            df.to_csv(f, index=False)

    @staticmethod
    def load_dataset(filepath: str = None) -> tuple[str, pd.DataFrame, pd.DatetimeIndex]:
        metadata = {}
        with open(filepath, 'r') as f:
            lines = f.readlines()

        data_start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('#'):
                key_value = line[1:].strip().split(':', 1)
                if len(key_value) == 2:
                    key, value = key_value
                    metadata[key.strip()] = value.strip()
            else:
                data_start_idx = i
                break

        from io import StringIO
        csv_data = ''.join(lines[data_start_idx:])
        df = pd.read_csv(StringIO(csv_data))

        target_name = metadata['target_name']
        dt = df.pop('DateTime')

        return target_name, df, dt

    @staticmethod
    def load_dataset_from_filepath(filepath: str = None) -> tuple[str, pd.DataFrame, pd.DatetimeIndex]:
        return DatasetPreprocessor.load_dataset(filepath)

    @staticmethod
    def load_dataset_from_file(file: str = None) -> tuple[str, pd.DataFrame, pd.DatetimeIndex]:
        metadata = {}
        lines = [line.decode('utf-8') if isinstance(line, bytes) else line for line in file.readlines()]

        data_start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('#'):
                key_value = line[1:].strip().split(':', 1)
                if len(key_value) == 2:
                    key, value = key_value
                    metadata[key.strip()] = value.strip()
            else:
                data_start_idx = i
                break

        from io import StringIO
        csv_data = ''.join(lines[data_start_idx:])
        df = pd.read_csv(StringIO(csv_data))

        target_name = metadata['target_name']
        dt = df.pop('DateTime')

        return target_name, df, dt

def load_dataset_from_file(file):
    return DatasetPreprocessor.load_dataset_from_file(file)
