from flask import Blueprint, request, jsonify
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token
from pymongo import MongoClient
from datetime import timedelta

auth_bp = Blueprint('auth', __name__)
bcrypt = Bcrypt()
client = MongoClient("your_mongodb_connection_string")
db = client["ASLApp"]
users = db["users"]

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.json
    if users.find_one({'email': data['email']}):
        return jsonify({"msg": "Email already exists"}), 400
    hashed_pw = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    users.insert_one({'email': data['email'], 'password': hashed_pw})
    return jsonify({"msg": "User created"}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    user = users.find_one({'email': data['email']})
    if user and bcrypt.check_password_hash(user['password'], data['password']):
        token = create_access_token(identity=user['email'], expires_delta=timedelta(days=1))
        return jsonify(access_token=token), 200
    return jsonify({"msg": "Invalid credentials"}), 401
