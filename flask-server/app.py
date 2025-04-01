from flask import Flask, jsonify, request, Response
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from flask_cors import CORS  # Enable CORS for cross-origin requests
from auth import authenticate_user, register_user
from models import XGB_MT1R1, LSTM_FINAL  # Import the model training functions
from dataset_loader import load_ieso_dataset, load_climate_dataset, IESODataset  # Import data loaders
import datetime
import pandas as pd
import io, base64
import matplotlib.pyplot as plt

# Import the anomaly detection module
from anomaly_detection import AnomalyDetection

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

# Route for training XGBoost model
@app.route('/xgb', methods=['POST'])
def train_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']

    try:
        df = pd.read_csv(file, low_memory=False)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV file: {str(e)}"}), 400

    target = request.form.get('target', 'Toronto')

    try:
        result = XGB_MT1R1(df, target)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)

# Route for training LSTM model
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

# Route to generate a merged CSV file from energy and climate datasets
@app.route('/generate_csv', methods=['POST'])
def generate_csv():
    data = request.get_json() if request.is_json else request.form

    dataset_type = data.get('dataset_type')
    if not dataset_type or dataset_type.lower() not in ['zonal', 'fsa']:
        return jsonify({"error": "dataset_type must be provided and be either 'Zonal' or 'FSA'."}), 400
    dataset_type = dataset_type.lower()

    predictor_repo = data.get('predictor_repo', 'climate').lower()

    if dataset_type == 'zonal':
        target_zone = data.get('target_zone', 'Toronto')
        try:
            start_year = int(data.get('start_year'))
            end_year = int(data.get('end_year'))
        except (TypeError, ValueError):
            return jsonify({"error": "For zonal dataset, start_year and end_year must be provided as integers."}), 400

        ds = IESODataset('zonal', region="ON")
        target_options = ds.get_target_options()
        try:
            target_idx = next(i for i, option in enumerate(target_options) if option.lower() == target_zone.lower())
        except StopIteration:
            return jsonify({"error": f"Target zone '{target_zone}' not found. Available options: {target_options}."}), 400

        try:
            energy_df = ds.load_dataset(start_date=start_year, end_date=end_year, target_idx=target_idx, download=True)
        except Exception as e:
            return jsonify({"error": f"Error loading zonal dataset: {str(e)}"}), 500

        climate_start_year = start_year
        climate_end_year = end_year

    elif dataset_type == 'fsa':
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

        start_date = int(f"{start_year}{start_month:02d}")
        end_date = int(f"{end_year}{end_month:02d}")

        ds = IESODataset('fsa', region="ON")
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

        try:
            energy_df = ds.load_dataset(start_date=start_date, end_date=end_date, target_idx=target_idx, download=True)
        except Exception as e:
            return jsonify({"error": f"Error loading FSA dataset: {str(e)}"}), 500

        climate_start_year = start_year
        climate_end_year = end_year

    if predictor_repo in ['climate', 'weather']:
        try:
            predictor_df = load_climate_dataset(climate_start_year, climate_end_year, join=True)
        except Exception as e:
            return jsonify({"error": f"Error loading climate dataset: {str(e)}"}), 500
    else:
        return jsonify({"error": f"Predictor repository '{predictor_repo}' is not supported."}), 400

    try:
        merged_df = pd.merge(energy_df, predictor_df, on="DateTime")
    except Exception as e:
        return jsonify({"error": f"Error merging datasets: {str(e)}"}), 500

    csv_data = merged_df.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=merged_data.csv"}
    )

@app.route('/anomaly_detection', methods=['POST'])
def anomaly_detection_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']

    try:
        df = pd.read_csv(file, low_memory=False)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV file: {str(e)}"}), 400

    target = request.form.get('target', 'Toronto')

    try:
        ad = AnomalyDetection(df, target)

        # Get all 4 plots separately
        plots = ad.summary_plots()

        num_anoms = len(ad.anomalies)
        best_ten = ad.anomalies.sort_values(by=['Anomaly Score'], ascending=True).head(10)
        best_table = best_ten.to_string(index=False)

        return jsonify({
            "num_anomalies": num_anoms,
            "best_ten_anomalies": best_table,
            **plots  # returns: scatter_plot, month_plot, longest_anomalous_plot, longest_clean_plot
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
