# app.py
from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from auth import authenticate_user, register_user
import datetime

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key
jwt = JWTManager(app)

# Health check route
@app.route('/')
def index():
    return jsonify(message="Flask API is running!")

# Authentication endpoints will be added below...

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


