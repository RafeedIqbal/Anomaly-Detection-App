# app.py
from flask import Flask, jsonify, request, Response
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from flask_cors import CORS  # Enable CORS for cross-origin requests
from auth import authenticate_user, register_user
from models import XGB_MT1R1, LSTM_FINAL  # Import the model training function
from dataset_loader import load_ieso_dataset, load_climate_dataset  # Import data loaders
import datetime
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.config['JWT_SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key
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

# New route for training the model
@app.route('/xgb', methods=['POST'])
def train_route():
    # Check if a file was provided in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']

    # Read CSV file into a DataFrame
    try:
        df = pd.read_csv(file, low_memory=False)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV file: {str(e)}"}), 400

    # Use the provided target column or default to 'Toronto'
    target = request.form.get('target', 'Toronto')

    try:
        result = XGB_MT1R1(df, target)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)

@app.route('/lstm', methods=['POST'])
def lstm_train_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    try:
        df = pd.read_csv(file, low_memory=False)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV file: {str(e)}"}), 400
    target = request.form.get('target', 'Toronto')
    try:
        result = LSTM_FINAL(df, target)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(result)

# New route to generate a CSV for modeling
@app.route('/generate_csv', methods=['POST'])
def generate_csv():
    # Parse input parameters from JSON or form-data
    data = request.get_json() if request.is_json else request.form

    # Validate and get dataset type (should be either "Zonal" or "FSA")
    dataset_type = data.get('dataset_type')
    if not dataset_type or dataset_type.lower() not in ['zonal', 'fsa']:
        return jsonify({"error": "dataset_type must be provided and be either 'Zonal' or 'FSA'."}), 400
    dataset_type = dataset_type.lower()

    # Get predictor repository (default: climate)
    predictor_repo = data.get('predictor_repo', 'climate').lower()

    # Prepare variables to hold the predictor dataset's year range
    if dataset_type == 'zonal':
        # For zonal datasets, require target_zone, start_year, and end_year.
        target_zone = data.get('target_zone', 'Toronto')
        try:
            start_year = int(data.get('start_year'))
            end_year = int(data.get('end_year'))
        except (TypeError, ValueError):
            return jsonify({"error": "For zonal dataset, start_year and end_year must be provided as integers."}), 400

        # Create a zonal IESODataset instance for Ontario
        ds = IESODataset('zonal', region="ON")
        target_options = ds.get_target_options()  # e.g., ['Northwest', 'Northeast', 'Ottawa', 'East', 'Toronto', ...]
        try:
            target_idx = next(i for i, option in enumerate(target_options) if option.lower() == target_zone.lower())
        except StopIteration:
            return jsonify({"error": f"Target zone '{target_zone}' not found. Available options: {target_options}."}), 400

        # Load the zonal energy dataset using years as the date range
        try:
            energy_df = ds.load_dataset(start_date=start_year, end_date=end_year, target_idx=target_idx, download=True)
        except Exception as e:
            return jsonify({"error": f"Error loading zonal dataset: {str(e)}"}), 500

        # Use the same year range for the climate dataset
        climate_start_year = start_year
        climate_end_year = end_year

    elif dataset_type == 'fsa':
        # For FSA datasets, require target_city, start_year, start_month, end_year, and end_month.
        target_city = data.get('target_city')
        if not target_city:
            return jsonify({"error": "For FSA dataset, target_city must be provided."}), 400
        try:
            start_year = int(data.get('start_year'))
            start_month = int(data.get('start_month'))
            end_year = int(data.get('end_year'))
            end_month = int(data.get('end_month'))
        except (TypeError, ValueError):
            return jsonify({"error": "For FSA dataset, start_year, start_month, end_year, and end_month must be provided as integers."}), 400

        # Combine year and month into a YYYYMM integer for the dataset loader
        start_date = int(f"{start_year}{start_month:02d}")
        end_date = int(f"{end_year}{end_month:02d}")

        ds = IESODataset('fsa', region="ON")
        # Override the default target options with the list from message.txt
        try:
            import ast
            with open('message.txt', 'r') as f:
                cities = ast.literal_eval(f.read())
            ds.target_options = cities
        except Exception as e:
            return jsonify({"error": f"Error loading target cities from message.txt: {str(e)}"}), 500

        try:
            target_idx = next(i for i, city in enumerate(ds.target_options) if city.lower() == target_city.lower())
        except StopIteration:
            return jsonify({"error": f"Target city '{target_city}' not found. Available options: {ds.target_options}."}), 400

        # Load the FSA energy dataset using YYYYMM as the date range
        try:
            energy_df = ds.load_dataset(start_date=start_date, end_date=end_date, target_idx=target_idx, download=True)
        except Exception as e:
            return jsonify({"error": f"Error loading FSA dataset: {str(e)}"}), 500

        # For the predictor dataset, use the provided year range (ignoring month details)
        climate_start_year = start_year
        climate_end_year = end_year

    # Load the predictor (climate) dataset
    if predictor_repo in ['climate', 'weather']:
        try:
            predictor_df = load_climate_dataset(climate_start_year, climate_end_year, join=True)
        except Exception as e:
            return jsonify({"error": f"Error loading climate dataset: {str(e)}"}), 500
    else:
        return jsonify({"error": f"Predictor repository '{predictor_repo}' is not supported."}), 400

    # Merge the energy and predictor datasets on the "DateTime" column
    try:
        merged_df = pd.merge(energy_df, predictor_df, on="DateTime")
    except Exception as e:
        return jsonify({"error": f"Error merging datasets: {str(e)}"}), 500

    # Convert the merged DataFrame to CSV
    csv_data = merged_df.to_csv(index=False)

    # Return the CSV as a downloadable file
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=merged_data.csv"}
    )

if __name__ == '__main__':
    app.run(debug=True)
