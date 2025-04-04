from flask import Flask, jsonify, request, Response
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from flask_cors import CORS  # Enable CORS for cross-origin requests
from auth import authenticate_user, register_user
# Import the modified model training functions
from models import XGB_MT1R1, LSTM_FINAL
from dataset_loader import IESODataset, ClimateDataset  # Updated import for new dataset loader
import datetime
import pandas as pd
import io, base64
import matplotlib.pyplot as plt
# Import the modified anomaly detection module
from anomaly_detection import AnomalyDetection
import traceback # For detailed error logging


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.config['JWT_SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key in production
jwt = JWTManager(app)

@app.route('/')
def index():
    return jsonify(message="Flask API is running!")

@app.route('/register', methods=['POST'])
def register():
    return register_user()

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    user = authenticate_user(username, password)
    if not user:
        return jsonify(message="Invalid credentials"), 401

    access_token = create_access_token(identity=username, expires_delta=datetime.timedelta(hours=1))
    return jsonify(access_token=access_token), 200

@app.route('/profile', methods=['GET'])
@jwt_required()
def profile():
    current_user = get_jwt_identity()
    return jsonify(message=f"Hello, {current_user}!"), 200

# Route for training XGBoost model (modified: no target form field)
@app.route('/xgb', methods=['POST'])
def train_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']

    try:
        df = pd.read_csv(file, low_memory=False)
        # Basic check for enough columns before passing to model
        if len(df.columns) < 2:
             return jsonify({"error": f"CSV must have at least 2 columns. Found: {df.columns.tolist()}"}), 400
        # --- Target no longer needed from request ---
        # target = request.form.get('target', 'Toronto') # REMOVED
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV file: {str(e)}"}), 400

    try:
        # Call model function without target argument
        result = XGB_MT1R1(df)
    except Exception as e:
        print(f"Error during XGBoost training: {traceback.format_exc()}") # Log detailed error
        return jsonify({"error": str(e)}), 500

    return jsonify(result)

# Route for training LSTM model (modified: no target form field)
@app.route('/lstm', methods=['POST'])
def lstm_train_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    try:
        df = pd.read_csv(file, low_memory=False)
        if len(df.columns) < 2:
             return jsonify({"error": f"CSV must have at least 2 columns. Found: {df.columns.tolist()}"}), 400
        # --- Target no longer needed from request ---
        # target = request.form.get('target', 'Toronto') # REMOVED
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV file: {str(e)}"}), 400

    try:
         # Call model function without target argument
        result = LSTM_FINAL(df) # Assuming default look_back is okay
        # If look_back needs to be configurable, get it from request.form
        # look_back = int(request.form.get('look_back', 6))
        # result = LSTM_FINAL(df, look_back=look_back)
    except Exception as e:
        print(f"Error during LSTM training: {traceback.format_exc()}") # Log detailed error
        return jsonify({"error": str(e)}), 500
    return jsonify(result)

# Route to generate a merged CSV file from energy and climate datasets
@app.route('/generate_csv', methods=['POST'])
def generate_csv():
    # This route remains unchanged as it generates the input CSV,
    # not consumes one where the target needs inferring.
    # The target *zone/city* specified here determines which column
    # becomes the second column in the *output* CSV, which the
    # /xgb and /lstm routes will then automatically use.
    data = request.get_json() if request.is_json else request.form

    dataset_type = data.get('dataset_type')
    if not dataset_type or dataset_type.lower() not in ['zonal', 'fsa']:
        return jsonify({"error": "dataset_type must be provided and be either 'Zonal' or 'FSA'."}), 400
    dataset_type = dataset_type.lower()

    predictor_repo = data.get('predictor_repo', 'climate').lower()

    # --- Target specification for data *generation* ---
    # This target name is crucial for selecting the correct data *before* saving the CSV.
    # It will become the second column header in the generated CSV.
    if dataset_type == 'zonal':
        target_name = data.get('target_zone')
        if not target_name:
             return jsonify({"error": "For zonal dataset, target_zone must be provided."}), 400
        try:
            start_year = int(data.get('start_year'))
            end_year = int(data.get('end_year'))
        except (TypeError, ValueError):
            return jsonify({"error": "For zonal dataset, start_year and end_year must be provided as integers."}), 400

        ds = IESODataset('zonal', region="ON")
        target_options = ds.get_target_options()
        try:
            # Find the index corresponding to the requested target zone name
            target_idx = next(i for i, option in enumerate(target_options) if option.lower() == target_name.lower())
        except StopIteration:
            return jsonify({"error": f"Target zone '{target_name}' not found. Available options: {target_options}."}), 400

        try:
            energy_df = ds.load_dataset(start_date=str(start_year), end_date=str(end_year), target_idx=target_idx, download=True)
            # Rename the selected target column to the user-provided name for clarity
            original_col_name = energy_df.columns[-1] # Assumes target is the last column added
            if original_col_name != target_name:
                 energy_df = energy_df.rename(columns={original_col_name: target_name})

        except Exception as e:
            print(f"Error loading zonal dataset: {traceback.format_exc()}")
            return jsonify({"error": f"Error loading zonal dataset: {str(e)}"}), 500

    elif dataset_type == 'fsa':
        target_name = data.get('target_city')
        if not target_name:
            return jsonify({"error": "For FSA dataset, target_city must be provided."}), 400
        try:
            start_year = int(data.get('start_year'))
            start_month = int(data.get('start_month'))
            end_year = int(data.get('end_year'))
            end_month = int(data.get('end_month'))
        except (TypeError, ValueError):
            return jsonify({"error": "For FSA dataset, start_year, start_month, end_year, and end_month must be provided as integers."}), 400

        start_date = int(f"{start_year}{start_month:02d}")
        end_date = int(f"{end_year}{end_month:02d}")

        ds = IESODataset('fsa', region="ON")
        try:
            import ast
            # Ensure message.txt exists or handle error
            try:
                with open('message.txt', 'r') as f:
                     cities = ast.literal_eval(f.read())
            except FileNotFoundError:
                 return jsonify({"error": "'message.txt' not found. Cannot determine available FSA cities."}), 500
            ds.target_options = cities
        except Exception as e:
            print(f"Error loading target cities from message.txt: {traceback.format_exc()}")
            return jsonify({"error": f"Error loading target cities from message.txt: {str(e)}"}), 500

        try:
            target_idx = next(i for i, city in enumerate(ds.target_options) if city.lower() == target_name.lower())
        except StopIteration:
            return jsonify({"error": f"Target city '{target_name}' not found. Available options might be outdated if message.txt is old. Found options: {ds.target_options}."}), 400

        try:
            energy_df = ds.load_dataset(start_date=str(start_date), end_date=str(end_date), target_idx=target_idx, download=True)
            # Rename the selected target column
            original_col_name = energy_df.columns[-1]
            if original_col_name != target_name:
                 energy_df = energy_df.rename(columns={original_col_name: target_name})

        except Exception as e:
            print(f"Error loading FSA dataset: {traceback.format_exc()}")
            return jsonify({"error": f"Error loading FSA dataset: {str(e)}"}), 500
    else:
        # This case should be caught earlier, but included for completeness
        return jsonify({"error": "Invalid dataset_type provided."}), 400

    # --- Predictor (Climate) Data Loading ---
    if predictor_repo in ['climate', 'weather']:
        try:
            # Instantiate and load the climate dataset using the IESO dataset instance
            climate_ds = ClimateDataset(ds, region="ON")

             # Ensure the date range is set on the IESO dataset instance *before* loading climate data
            if not hasattr(ds, 'datetime_range') or ds.datetime_range is None:
                 ds.set_datetime() # Call set_datetime if not already done by load_dataset

            # Check if datetime_range was successfully set
            if not hasattr(ds, 'datetime_range') or ds.datetime_range is None:
                 raise ValueError("datetime_range could not be determined from the energy dataset.")

            climate_ds.datetime_range = ds.datetime_range # Pass the range explicitly

            climate_ds.load_dataset(download=True)
            predictor_df = climate_ds.df

            # --- Ensure DateTime column in predictor_df ---
            if not isinstance(predictor_df.index, pd.DatetimeIndex) and "DateTime" not in predictor_df.columns and "time" not in predictor_df.columns :
                 # If index is not datetime and no obvious time columns exist, error out
                 return jsonify({"error": f"Predictor dataset does not have a DatetimeIndex or a recognizable 'DateTime'/'time' column. Columns: {predictor_df.columns.tolist()}"}), 500

            # If index IS datetime, reset it to become a column
            if isinstance(predictor_df.index, pd.DatetimeIndex):
                index_name = predictor_df.index.name if predictor_df.index.name else 'index_time' # Use existing name or default
                predictor_df = predictor_df.reset_index()
                # Rename the new column to 'DateTime' if it's not already called that
                if index_name in predictor_df.columns and index_name != 'DateTime':
                     predictor_df = predictor_df.rename(columns={index_name: "DateTime"})
                elif 'index' in predictor_df.columns and 'DateTime' not in predictor_df.columns: # Handle default 'index' name from reset_index
                     predictor_df = predictor_df.rename(columns={'index': 'DateTime'})

            # If 'time' column exists but 'DateTime' doesn't, rename 'time'
            elif 'time' in predictor_df.columns and 'DateTime' not in predictor_df.columns:
                 predictor_df = predictor_df.rename(columns={"time": "DateTime"})

            # Final check
            if "DateTime" not in predictor_df.columns:
                 return jsonify({"error": f"Failed to establish 'DateTime' column in predictor dataset. Columns found: {predictor_df.columns.tolist()}"}), 500

            # Convert predictor DateTime column to datetime objects for robust merging
            predictor_df['DateTime'] = pd.to_datetime(predictor_df['DateTime'])

        except Exception as e:
            print(f"Error during climate data loading/processing: {traceback.format_exc()}")
            return jsonify({"error": f"Error loading or processing climate dataset: {str(e)}"}), 500
    else:
        return jsonify({"error": f"Predictor repository '{predictor_repo}' is not supported."}), 400

    # --- Merging Logic ---
    try:
        # Ensure 'DateTime' column exists and is datetime type in energy_df
        if isinstance(energy_df.index, pd.DatetimeIndex):
             energy_df = energy_df.reset_index()
             # Rename if index wasn't named 'DateTime' or was default 'index'
             if 'index' in energy_df.columns and 'DateTime' not in energy_df.columns:
                 energy_df = energy_df.rename(columns={'index': 'DateTime'})
             # Add other potential index names if necessary

        if 'DateTime' not in energy_df.columns:
             return jsonify({"error": f"Energy dataset missing 'DateTime' column before merge. Columns: {energy_df.columns.tolist()}"}), 500

        energy_df['DateTime'] = pd.to_datetime(energy_df['DateTime'])


        # --- Perform the merge ---
        # Use outer join initially to diagnose mismatches, then switch back to inner if preferred
        # merged_df = pd.merge(energy_df, predictor_df, on="DateTime", how="outer", indicator=True)
        # print(merged_df['_merge'].value_counts()) # See how many rows match
        # merged_df = merged_df[merged_df['_merge'] == 'both'].drop(columns=['_merge']) # Keep only matching rows

        merged_df = pd.merge(energy_df, predictor_df, on="DateTime", how="inner")


        # Check if merge resulted in empty dataframe
        if merged_df.empty:
            print("Warning: Merged dataframe is empty. Check date ranges, timezones, and column names.")
            # Provide more info for debugging
            min_energy_date = energy_df['DateTime'].min()
            max_energy_date = energy_df['DateTime'].max()
            min_pred_date = predictor_df['DateTime'].min()
            max_pred_date = predictor_df['DateTime'].max()
            return jsonify({
                "error": "Merging resulted in an empty dataset. Potential date range mismatch or no overlapping DateTime values.",
                "energy_date_range": f"{min_energy_date} to {max_energy_date}",
                "predictor_date_range": f"{min_pred_date} to {max_pred_date}",
                "energy_columns": energy_df.columns.tolist(),
                "predictor_columns": predictor_df.columns.tolist()
                }), 400

        # Ensure the target column is indeed the second column after the merge
        cols = merged_df.columns.tolist()
        if len(cols) > 1 and cols[1] != target_name:
            # If target isn't second, try to reorder: DateTime, Target, rest...
            if target_name in cols:
                 print(f"Reordering columns to place '{target_name}' second.")
                 other_cols = [col for col in cols if col not in ['DateTime', target_name]]
                 merged_df = merged_df[['DateTime', target_name] + other_cols]
            else:
                 # This shouldn't happen if merge was successful and renaming worked
                 print(f"Warning: Target column '{target_name}' not found after merge. Columns: {cols}")


    except KeyError as e:
         missing_in_energy = 'DateTime' not in energy_df.columns
         missing_in_predictor = 'DateTime' not in predictor_df.columns
         return jsonify({
             "error": f"Error merging datasets: KeyError on column {e}. Ensure 'DateTime' column exists and is correctly named in both datasets.",
             "'DateTime' in energy_df?": not missing_in_energy,
             "'DateTime' in predictor_df?": not missing_in_predictor,
             "energy_cols": energy_df.columns.tolist(),
             "predictor_cols": predictor_df.columns.tolist()
             }), 500
    except Exception as e:
        print(f"Error during dataset merging: {traceback.format_exc()}")
        return jsonify({"error": f"Error merging datasets: {str(e)}"}), 500

    # --- Convert to CSV and return ---
    # Ensure DateTime is formatted consistently if needed (optional)
    # merged_df['DateTime'] = merged_df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    csv_data = merged_df.to_csv(index=False)
    filename = f"merged_{dataset_type}_{target_name.replace(' ','_')}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )


# Route for anomaly detection (modified: no target form field)
@app.route('/anomaly_detection', methods=['POST'])
def anomaly_detection_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']

    try:
        # This CSV is the *output* from /xgb or /lstm
        # It should have columns like: DateTime, TargetName, pred, error
        df = pd.read_csv(file, low_memory=False)
        # Check structure expected by AnomalyDetection class
        if len(df.columns) < 4:
             return jsonify({"error": f"Input CSV for anomaly detection must have at least 4 columns (DateTime, Target, pred, error). Found: {df.columns.tolist()}"}), 400
        if 'pred' not in df.columns or 'error' not in df.columns:
             return jsonify({"error": f"Input CSV must contain 'pred' and 'error' columns. Found: {df.columns.tolist()}"}), 400

        # --- Target no longer needed from request ---
        # target = request.form.get('target', 'Toronto') # REMOVED
    except Exception as e:
        return jsonify({"error": f"Failed to read results CSV file: {str(e)}"}), 400

    try:
        # Instantiate AnomalyDetection without target argument
        ad = AnomalyDetection(df)

        # Get summary plots and other info
        plots = ad.summary_plots() # This generates the base64 plot strings
        num_anoms = ad.num_anomalies() # This prints and returns the number
        best_table_str = ad.best_ten_anomalies() # This prints and returns the table string

        # Return results
        return jsonify({
            "num_anomalies": num_anoms,
            "best_ten_anomalies": best_table_str, # Return the string representation
            "inferred_target": ad.target, # Confirm which target was used
            **plots  # Add scatter_plot, month_plot, etc.
        })
    except Exception as e:
        print(f"Error during anomaly detection processing: {traceback.format_exc()}") # Log detailed error
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Make sure matplotlib runs headlessly if needed, Agg backend set in models.py
    app.run(debug=True, host='0.0.0.0') # Listen on all interfaces if running in Docker/VM