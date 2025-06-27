import sys
import os
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

app = Flask(__name__)

# Register blueprints
from routes.upload import upload_bp
from routes.delete import delete_bp
app.register_blueprint(upload_bp)
app.register_blueprint(delete_bp)

# Initialize data structures
switches = []
lan_leaves = []
lan_spines = []
compute_entries = []
storage_entries = []
cables_entries = []
rack_rows_entries = []
project_name = "Data Center Configuration"

# Create data directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)

# Default LAN roles
lan_roles = {}

# Function to save configuration to file
def save_configuration():
    config = {
        "project_name": project_name,
        "switches": switches,
        "lan_leaves": lan_leaves,
        "lan_spines": lan_spines,
        "compute_nodes": compute_entries,
        "storage_blocks": storage_entries,
        "cables": cables_entries,
        "rack_rows": rack_rows_entries
    }
    
    # Create safe filename from project name
    safe_filename = "".join([c if c.isalnum() else "_" for c in project_name])
    
    # Save to file
    filepath = os.path.join(os.path.dirname(__file__), 'data', f"{safe_filename}.json")
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    return filepath

# Function to load configuration from file
def load_configuration():
    global project_name, switches, lan_leaves, lan_spines, compute_entries, storage_entries, cables_entries, rack_rows_entries, lan_roles

    # Create safe filename from project name
    safe_filename = "".join([c if c.isalnum() else "_" for c in project_name])
    filepath = os.path.join(os.path.dirname(__file__), 'data', f"{safe_filename}.json")

    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
                project_name = config.get('project_name', 'Data Center Configuration')
                switches = config.get('switches', [])
                lan_leaves = config.get('lan_leaves', [])
                lan_spines = config.get('lan_spines', [])
                compute_entries = config.get('compute_nodes', [])
                storage_entries = config.get('storage_blocks', [])
                cables_entries = config.get('cables', [])
                rack_rows_entries = config.get('rack_rows', [])
                
                # Update LAN roles
                lan_roles.clear()
                for leaf in lan_leaves:
                    for lan_name, details in leaf.items():
                        lan_roles[lan_name] = details.get('role', '')

        except Exception as e:
            print(f"Error loading configuration: {e}")
            # Reset to default if loading fails
            project_name = "Data Center Configuration"
            switches = []
            lan_leaves = []
            lan_spines = []
            compute_entries = []
            storage_entries = []
            cables_entries = []
            rack_rows_entries = []
            lan_roles = {}


@app.route('/')
def index():
    load_configuration() # Load configuration when index page is accessed
    # Get next LAN number
    next_lan_number = 1
    existing_lan_numbers = []
    
    for leaf in lan_leaves:
        for lan_name in leaf.keys():
            if lan_name.startswith('LAN_'):
                try:
                    num = int(lan_name.split('_')[1])
                    existing_lan_numbers.append(num)
                except (IndexError, ValueError):
                    pass
    
    if existing_lan_numbers:
        next_lan_number = max(existing_lan_numbers) + 1
    
    return render_template('index.html',
                          project_name=project_name,
                          switches=switches,
                          lan_leaves=lan_leaves,
                          lan_spines=lan_spines,
                          compute_entries=compute_entries,
                          storage_entries=storage_entries,
                          cables_entries=cables_entries,
                          rack_rows_entries=rack_rows_entries,
                          next_lan_number=next_lan_number,
                          lan_roles=lan_roles)

@app.route('/update_project_name', methods=['POST'])
def update_project_name():
    global project_name
    project_name = request.form.get('project_name', 'Data Center Configuration')
    
    # Save configuration with new project name
    save_configuration()
    
    return jsonify({
        "status": "success",
        "message": "Project name updated successfully"
    })

@app.route('/submit_switch', methods=['POST'])
def submit_switch():
    global switches
    
    # Get form data
    model = request.form.get('model')
    ports = int(request.form.get('ports', 0))
    speed = int(request.form.get('speed', 0))
    height = float(request.form.get('height', 0))
    wattage = float(request.form.get('wattage', 0))
    weight = float(request.form.get('weight', 0))
    
    # Optional fields
    uplink_count = request.form.get('uplink_count', '')
    uplink_speed = request.form.get('uplink_speed', '')
    
    # Create switch entry
    switch_entry = {
        "model": model,
        "ports": ports,
        "speed": speed,
        "height": height,
        "wattage": wattage,
        "weight": weight
    }
    
    # Add optional fields if provided
    if uplink_count and uplink_speed:
        switch_entry["uplink_count"] = int(uplink_count)
        switch_entry["uplink_speed"] = int(uplink_speed)
    
    # Check if editing existing entry
    edit_index = request.form.get('edit_index', '')
    if edit_index and edit_index.isdigit():
        index = int(edit_index)
        if 0 <= index < len(switches):
            switches[index] = switch_entry
            message = "Switch updated successfully"
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid switch index"
            })
    else:
        # Add new entry
        switches.append(switch_entry)
        message = "Switch added successfully"
    
    # Save configuration
    save_configuration()
    
    return jsonify({
        "status": "success",
        "message": message,
        "all_entries": switches
    })

@app.route('/submit_lan_leaf', methods=['POST'])
def submit_lan_leaf():
    global lan_leaves, lan_roles
    
    # Get form data
    lan_name = request.form.get('lan_name')
    role = request.form.get('role')
    fixed_qty = int(request.form.get('fixed_qty', 0))
    lan_type = request.form.get('lan_type')
    topology = request.form.get('topology')
    selected_switches = request.form.getlist('selected_switches[]')
    
    # Create LAN leaf entry
    lan_leaf_entry = {
        lan_name: {
            "role": role,
            "fixedqty": fixed_qty,
            "type": lan_type,
            "topology": topology,
            "switches": selected_switches
        }
    }
    
    # Update LAN roles
    lan_roles[lan_name] = role
    
    # Check if editing existing entry
    edit_index = request.form.get('edit_index', '')
    if edit_index and edit_index.isdigit():
        index = int(edit_index)
        if 0 <= index < len(lan_leaves):
            lan_leaves[index] = lan_leaf_entry
            message = "LAN leaf updated successfully"
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid LAN leaf index"
            })
    else:
        # Add new entry
        lan_leaves.append(lan_leaf_entry)
        message = "LAN leaf added successfully"
    
    # Save configuration
    save_configuration()
    
    return jsonify({
        "status": "success",
        "message": message,
        "all_entries": lan_leaves
    })

@app.route('/submit_lan_spine', methods=['POST'])
def submit_lan_spine():
    global lan_spines
    
    # Get form data
    spine_name = request.form.get('spine_name')
    role = request.form.get('role')
    fixed_qty = int(request.form.get('fixed_qty', 0))
    spine_type = request.form.get('spine_type')
    topology = request.form.get('topology')
    selected_switches = request.form.getlist('selected_switches_spine[]')
    
    # Create LAN spine entry
    lan_spine_entry = {
        spine_name: {
            "role": role,
            "fixedqty": fixed_qty,
            "type": spine_type,
            "topology": topology,
            "switches": selected_switches
        }
    }
    
    # Check if editing existing entry
    edit_index = request.form.get('edit_index', '')
    if edit_index and edit_index.isdigit():
        index = int(edit_index)
        if 0 <= index < len(lan_spines):
            lan_spines[index] = lan_spine_entry
            message = "LAN spine updated successfully"
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid LAN spine index"
            })
    else:
        # Add new entry
        lan_spines.append(lan_spine_entry)
        message = "LAN spine added successfully"
    
    # Save configuration
    save_configuration()
    
    return jsonify({
        "status": "success",
        "message": message,
        "all_entries": lan_spines
    })

@app.route('/submit_compute', methods=['POST'])
def submit_compute():
    global compute_entries
    # Get form data
    compute_name = request.form.get('compute_name')
    count = int(request.form.get('count', 0))
    wattage = float(request.form.get('wattage', 0))
    height = float(request.form.get('height', 0))
    weight = float(request.form.get('weight', 0))
    
    # Create compute entry
    compute_entry = {
        compute_name: {
            "count": count,
            "wattage": wattage,
            "height": height,
            "weight": weight
        }
    }
    
    # Add LAN details
    for i in range(1, 7):
        lan_count = request.form.get(f'lan_{i}_count', '')
        lan_speed = request.form.get(f'lan_{i}_speed', '')
        if lan_count and lan_speed:
            compute_entry[compute_name][f'LAN_{i}'] = {
                "count": int(lan_count),
                "speed": int(lan_speed)
            }
            
    # Check if editing existing entry
    edit_index = request.form.get('edit_index', '')
    if edit_index and edit_index.isdigit():
        index = int(edit_index)
        if 0 <= index < len(compute_entries):
            compute_entries[index] = compute_entry
            message = "Compute node updated successfully"
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid compute node index"
            })
    else:
        # Add new entry
        compute_entries.append(compute_entry)
        message = "Compute node added successfully"
    
    # Save configuration
    save_configuration()
    
    return jsonify({
        "status": "success",
        "message": message,
        "all_entries": compute_entries
    })

@app.route('/submit_storage', methods=['POST'])
def submit_storage():
    global storage_entries
    
    # Get form data
    storage_name = request.form.get('storage_name')
    count = int(request.form.get('count', 0))
    wattage = float(request.form.get('wattage', 0))
    height = float(request.form.get('height', 0))
    weight = float(request.form.get('weight', 0))
    capacity = float(request.form.get('capacity', 0))
    storage_type = request.form.get('type')
    
    # Create storage entry
    storage_entry = {
        storage_name: {
            "count": count,
            "wattage": wattage,
            "height": height,
            "weight": weight,
            "capacity": capacity,
            "type": storage_type
        }
    }
    
    # Check if editing existing entry
    edit_index = request.form.get('edit_index', '')
    if edit_index and edit_index.isdigit():
        index = int(edit_index)
        if 0 <= index < len(storage_entries):
            storage_entries[index] = storage_entry
            message = "Storage block updated successfully"
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid storage block index"
            })
    else:
        # Add new entry
        storage_entries.append(storage_entry)
        message = "Storage block added successfully"
    
    # Save configuration
    save_configuration()
    
    return jsonify({
        "status": "success",
        "message": message,
        "all_entries": storage_entries
    })

@app.route('/submit_cable', methods=['POST'])
def submit_cable():
    global cables_entries
    
    # Get form data
    switch_model = request.form.get('switch_model')
    cable_model = request.form.get('cable_model')
    server_port_speed = int(request.form.get('server_port_speed', 0))
    split = int(request.form.get('split', 0))
    length = float(request.form.get('length', 0))
    
    # Create cable entry
    cable_entry = {
        "switch_model": switch_model,
        "cable_model": cable_model,
        "server_port_speed": server_port_speed,
        "split": split,
        "length": length
    }
    
    # Check if editing existing entry
    edit_index = request.form.get('edit_index', '')
    if edit_index and edit_index.isdigit():
        index = int(edit_index)
        if 0 <= index < len(cables_entries):
            cables_entries[index] = cable_entry
            message = "Cable updated successfully"
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid cable index"
            })
    else:
        # Add new entry
        cables_entries.append(cable_entry)
        message = "Cable added successfully"
    
    # Save configuration
    save_configuration()
    
    return jsonify({
        "status": "success",
        "message": message,
        "all_entries": cables_entries
    })

@app.route('/submit_rack_row', methods=['POST'])
def submit_rack_row():
    global rack_rows_entries
    
    # Get form data
    row_name = request.form.get('row_name')
    row_number = request.form.get('row_number')
    rack_type = request.form.get('rack_type')
    rack_count = int(request.form.get('rack_count', 0))
    
    # Create rack row entry
    rack_row_entry = {
        "row_name": row_name,
        "row_number": row_number,
        "rack_type": rack_type,
        "rack_count": rack_count
    }
    
    # Check if editing existing entry
    edit_index = request.form.get('edit_index', '')
    if edit_index and edit_index.isdigit():
        index = int(edit_index)
        if 0 <= index < len(rack_rows_entries):
            rack_rows_entries[index] = rack_row_entry
            message = "Rack row updated successfully"
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid rack row index"
            })
    else:
        # Add new entry
        rack_rows_entries.append(rack_row_entry)
        message = "Rack row added successfully"
    
    # Save configuration
    save_configuration()
    
    return jsonify({
        "status": "success",
        "message": message,
        "all_entries": rack_rows_entries
    })

@app.route('/get_current_config', methods=['GET'])
def get_current_config():
    # This route will provide the current configuration data
    config = {
        "project_name": project_name,
        "switches": switches,
        "lan_leaves": lan_leaves,
        "lan_spines": lan_spines,
        "compute_nodes": compute_entries,
        "storage_blocks": storage_entries,
        "cables": cables_entries,
        "rack_rows": rack_rows_entries
    }
    return jsonify({
        "status": "success",
        "message": "Current configuration fetched successfully",
        "config": config
    })

@app.route('/get_lan_roles', methods=['GET'])
def get_lan_roles():
    return jsonify({
        "status": "success",
        "lan_roles": lan_roles
    })

@app.route('/get_next_lan_number', methods=['GET'])
def get_next_lan_number():
    next_lan_number = 1
    existing_lan_numbers = []
    
    for leaf in lan_leaves:
        for lan_name in leaf.keys():
            if lan_name.startswith('LAN_'):
                try:
                    num = int(lan_name.split('_')[1])
                    existing_lan_numbers.append(num)
                except (IndexError, ValueError):
                    pass
    
    if existing_lan_numbers:
        next_lan_number = max(existing_lan_numbers) + 1
    
    return jsonify({
        "status": "success",
        "next_lan_number": next_lan_number
    })

@app.route('/load_config_data', methods=['GET'])
def load_config_data():
    load_configuration()
    return jsonify({
        "project_name": project_name,
        "switches": switches,
        "lan_leaves": lan_leaves,
        "lan_spines": lan_spines,
        "compute_nodes": compute_entries,
        "storage_blocks": storage_entries,
        "cables": cables_entries,
        "rack_rows": rack_rows_entries,
        "lan_roles": lan_roles,
        "next_lan_number": get_next_lan_number().json['next_lan_number'] # Access the JSON response
    })

if __name__ == '__main__':
    load_configuration() # Load configuration on startup
    app.run(debug=True)