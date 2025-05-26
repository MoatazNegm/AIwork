import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # DON'T CHANGE THIS !!!

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Get form data from AJAX request
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')
    
    # In a real application, you would process this data (e.g., save to database)
    # For this example, we'll just return a success response
    response = {
        'status': 'success',
        'message': 'Form submitted successfully',
        'data': {
            'name': name,
            'email': email,
            'message': message
        }
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
