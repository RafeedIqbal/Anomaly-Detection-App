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
        df = pd.read_csv(file)
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
        df = pd.read_csv(file)
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
    # Parse input parameters (either as JSON or form-data)
    data = request.get_json() if request.is_json else request.form
    # Energy repository: default "IESO"
    energy_repo = data.get('energy_repo', 'IESO').upper()
    # Predictor repository: default "climate"
    predictor_repo = data.get('predictor_repo', 'climate').lower()
    # Target zone: default "Toronto"
    target_zone = data.get('target_zone', 'Toronto')
    # Timeframe: first and last years (default 2010 and 2020)
    try:
        first_year = int(data.get('first_year', 2010))
        last_year = int(data.get('last_year', 2020))
    except ValueError:
        return jsonify({"error": "Invalid timeframe provided."}), 400

    # Load the energy dataset (currently only IESO is supported)
    if energy_repo == 'IESO':
        energy_df = load_ieso_dataset(first_year, last_year, join=True)
    else:
        return jsonify({"error": f"Energy repository '{energy_repo}' is not supported."}), 400

    # Load the predictor dataset (currently only climate data is supported)
    if predictor_repo in ['climate', 'weather']:
        predictor_df = load_climate_dataset(first_year, last_year, join=True)
    else:
        return jsonify({"error": f"Predictor repository '{predictor_repo}' is not supported."}), 400

    # Merge the datasets on the "DateTime" column
    merged_df = pd.merge(energy_df, predictor_df, on="DateTime")

    # Check if the target zone exists in the merged dataset
    if target_zone not in merged_df.columns:
        return jsonify({"error": f"Target zone '{target_zone}' not found in the energy dataset."}), 400

    # (Optionally) select only relevant columns:
    # For instance, you might want DateTime, the target zone column, and all predictor columns.
    # Here we return the full merged dataframe.
    csv_data = merged_df.to_csv(index=False)

    # Return CSV as a downloadable file
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=merged_data.csv"}
    )

if __name__ == '__main__':
    app.run(debug=True)
