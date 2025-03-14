# app.py
from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from flask_cors import CORS  # Enable CORS for cross-origin requests
from auth import authenticate_user, register_user
from models import XGB_MT1R1  # Import the model training function
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
@app.route('/XGB', methods=['POST'])
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

if __name__ == '__main__':
    app.run(debug=True)
