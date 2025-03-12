# auth.py
from flask import jsonify, request
from flask_jwt_extended import create_access_token
import datetime

# In-memory user store (for demonstration)
users = {}

def register_user():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    
    if username in users:
        return jsonify(message="User already exists"), 400
    
    # In production, hash the password before storing!
    users[username] = {"password": password}
    return jsonify(message="User registered successfully"), 201

def authenticate_user(username, password):
    user = users.get(username)
    if not user or user["password"] != password:
        return None
    return user
