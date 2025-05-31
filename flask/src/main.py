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

@app.route('/')
def index():
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

@app.route('/submit_cable', methods=['POST'])
def submit_cable():
    global cables_entries
    
    # Get form data
    switch_model = request.form.get('switch_model')
    original_switch_model = request.form.get('original_switch_model', switch_model)
    cable_model = request.form.get('cable_model')
    server_port_speed = int(request.form.get('server_port_speed', 0))
    split = int(request.form.get('split', 1))
    length = float(request.form.get('length', 0))
    
    # Create cable entry
    cable_entry = {
        "model": cable_model,
        "server_port_speed": server_port_speed,
        "split": split,
        "length": length
    }
    
    # Check if editing existing entry
    edit_index = request.form.get('edit_index', '')
    if edit_index and edit_index.isdigit():
        index = int(edit_index)
        
        # Find and update the existing cable
        for entry in cables_entries:
            if original_switch_model in entry:
                if index < len(entry[original_switch_model]):
                    # If switch model changed, remove from old and add to new
                    if original_switch_model != switch_model:
                        entry[original_switch_model].pop(index)
                        if len(entry[original_switch_model]) == 0:
                            cables_entries.remove(entry)
                        # Add to new switch model
                        switch_exists = False
                        for new_entry in cables_entries:
                            if switch_model in new_entry:
                                new_entry[switch_model].append(cable_entry)
                                switch_exists = True
                                break
                        if not switch_exists:
                            cables_entries.append({switch_model: [cable_entry]})
                    else:
                        # Just update the existing entry
                        entry[original_switch_model][index] = cable_entry
                    
                    # Save configuration
                    save_configuration()
                    
                    return jsonify({
                        "status": "success",
                        "message": "Cable updated successfully",
                        "all_entries": cables_entries
                    })
        
        return jsonify({
            "status": "error",
            "message": "Invalid cable index"
        })
    else:
        # Add new cable
        switch_exists = False
        for entry in cables_entries:
            if switch_model in entry:
                entry[switch_model].append(cable_entry)
                switch_exists = True
                break
        
        if not switch_exists:
            cables_entries.append({switch_model: [cable_entry]})
        
        # Save configuration
        save_configuration()
        
        return jsonify({
            "status": "success",
            "message": "Cable added successfully",
            "all_entries": cables_entries
        })

@app.route('/submit_rack_row', methods=['POST'])
def submit_rack_row():
    global rack_rows_entries
    
    # Get form data
    group_name = request.form.get('group_name')
    racks = int(request.form.get('racks', 0))
    group_count = int(request.form.get('group_count', 0))
    rack_to_rack = float(request.form.get('rack_to_rack', 0))
    row_to_next_row = float(request.form.get('row_to_next_row', 0))
    
    # Create rack row entry
    rack_row_entry = {
        group_name: {
            "Racks": racks,
            "group_count": group_count,
            "rack_to_rack": rack_to_rack,
            "Row_to_next_row": row_to_next_row
        }
    }
    
    # Add to rack rows entries
    rack_rows_entries.append(rack_row_entry)
    
    # Save configuration
    save_configuration()
    
    return jsonify({
        "status": "success",
        "message": "Rack row added successfully",
        "all_entries": rack_rows_entries
    })

@app.route('/get_switch', methods=['GET'])
def get_switch():
    index = int(request.args.get('index', -1))
    
    if 0 <= index < len(switches):
        return jsonify({
            "status": "success",
            "entry": switches[index]
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Invalid switch index"
        })

@app.route('/get_lan_leaf', methods=['GET'])
def get_lan_leaf():
    index = int(request.args.get('index', -1))
    
    if 0 <= index < len(lan_leaves):
        return jsonify({
            "status": "success",
            "entry": lan_leaves[index]
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Invalid LAN leaf index"
        })

@app.route('/get_lan_spine', methods=['GET'])
def get_lan_spine():
    index = int(request.args.get('index', -1))
    
    if 0 <= index < len(lan_spines):
        return jsonify({
            "status": "success",
            "entry": lan_spines[index]
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Invalid LAN spine index"
        })

@app.route('/get_compute', methods=['GET'])
def get_compute():
    index = int(request.args.get('index', -1))
    
    if 0 <= index < len(compute_entries):
        return jsonify({
            "status": "success",
            "entry": compute_entries[index]
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Invalid compute node index"
        })

@app.route('/delete_cable', methods=['POST'])
def delete_cable():
    global cables_entries
    
    switch_model = request.form.get('switch_model')
    cable_index = int(request.form.get('cable_index'))
    
    # Find the switch model entry
    for i, entry in enumerate(cables_entries):
        if switch_model in entry:
            # Remove the specific cable
            entry[switch_model].pop(cable_index)
            
            # If no more cables for this switch, remove the switch entry
            if len(entry[switch_model]) == 0:
                cables_entries.pop(i)
                
            # Save configuration
            save_configuration()
            
            return jsonify({
                "status": "success",
                "message": "Cable deleted successfully",
                "all_entries": cables_entries
            })
    
    return jsonify({
        "status": "error",
        "message": "Cable not found"
    })






@app.route('/clear_entries', methods=['POST'])
def clear_entries():
    global switches, lan_leaves, lan_spines, compute_entries, storage_entries, cables_entries, rack_rows_entries, lan_roles
    
    category = request.form.get('category')
    
    if category == 'switches':
        switches = []
        message = "All switches cleared"
    elif category == 'lan_leaves':
        lan_leaves = []
        lan_roles = {}
        message = "All LAN leaves cleared"
    elif category == 'lan_spines':
        lan_spines = []
        message = "All LAN spines cleared"
    elif category == 'compute':
        compute_entries = []
        message = "All compute nodes cleared"
    elif category == 'storage':
        storage_entries = []
        message = "All storage blocks cleared"
    elif category == 'cables':
        cables_entries = []
        message = "All cables cleared"
    elif category == 'rack_rows':
        rack_rows_entries = []
        message = "All rack rows cleared"
    elif category == 'all':
        switches = []
        lan_leaves = []
        lan_spines = []
        compute_entries = []
        storage_entries = []
        cables_entries = []
        rack_rows_entries = []
        lan_roles = {}
        message = "All entries cleared"
    else:
        return jsonify({
            "status": "error",
            "message": "Invalid category"
        })
    
    # Save configuration
    save_configuration()
    
    return jsonify({
        "status": "success",
        "message": message
    })

@app.route('/get_all_entries', methods=['GET'])
def get_all_entries():
    return jsonify({
        "project_name": project_name,
        "switches": switches,
        "lan_leaves": lan_leaves,
        "lan_spines": lan_spines,
        "compute_nodes": compute_entries,
        "storage_blocks": storage_entries,
        "cables": cables_entries,
        "rack_rows": rack_rows_entries
    })

@app.route('/get_lan_roles', methods=['GET'])
def get_lan_roles():
    return jsonify({
        "status": "success",
        "lan_roles": lan_roles
    })

@app.route('/get_next_lan_number', methods=['GET'])
def get_next_lan_number():
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
    
    return jsonify({
        "status": "success",
        "next_lan_number": next_lan_number
    })

@app.route('/export_configuration', methods=['GET'])
def export_configuration():
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
    filename = f"{safe_filename}.json"
    
    return jsonify({
        "status": "success",
        "message": "Configuration exported successfully",
        "config": json.dumps(config, indent=2),
        "filename": filename
    })

@app.route('/import_configuration', methods=['POST'])
def import_configuration():
    global switches, lan_leaves, lan_spines, compute_entries, storage_entries, cables_entries, rack_rows_entries, project_name, lan_roles
    
    if 'config_file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file provided"
        })
    
    file = request.files['config_file']
    
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No file selected"
        })
    
    try:
        # Try to parse as JSON
        content = file.read().decode('utf-8')
        
        # Check if content is a Python dictionary string
        if content.strip().startswith('{') and content.strip().endswith('}'):
            # Try to parse as JSON
            config = json.loads(content)
        else:
            # Try to evaluate as Python dictionary
            # This is potentially unsafe, but we're assuming trusted input
            import ast
            config_dict = {}
            
            # Parse the Python dictionary-like content
            exec_globals = {}
            exec(content, exec_globals)
            
            # Extract the dictionaries we're interested in
            if 'colors_info' in exec_globals:
                config_dict['compute_nodes'] = []
                for name, details in exec_globals['colors_info'].items():
                    config_dict['compute_nodes'].append({name: details})
            
            if 'Rack_rows' in exec_globals:
                config_dict['rack_rows'] = []
                for name, details in exec_globals['Rack_rows'].items():
                    config_dict['rack_rows'].append({name: details})
            
            if 'LANs' in exec_globals:
                config_dict['lan_leaves'] = []
                for name, details in exec_globals['LANs'].items():
                    # Extract switches from the LAN
                    switches_list = []
                    if 'switch' in details:
                        for switch in details['switch']:
                            # Add switch to switches list if not already there
                            if switch not in switches:
                                switches.append(switch)
                            switches_list.append(switch['model'])
                    
                    # Create LAN leaf entry
                    lan_leaf = {
                        name: {
                            "role": details.get('role', ''),
                            "fixedqty": details.get('fixedqty', 0),
                            "type": details.get('type', ''),
                            "topology": details.get('topology', ''),
                            "switches": switches_list
                        }
                    }
                    config_dict['lan_leaves'].append(lan_leaf)
                    
                    # Update LAN roles
                    lan_roles[name] = details.get('role', '')
            
            if 'Spines' in exec_globals:
                config_dict['lan_spines'] = []
                for name, details in exec_globals['Spines'].items():
                    # Extract switches from the spine
                    switches_list = []
                    if 'switch' in details:
                        for switch in details['switch']:
                            # Add switch to switches list if not already there
                            if switch not in switches:
                                switches.append(switch)
                            switches_list.append(switch['model'])
                    
                    # Create LAN spine entry
                    lan_spine = {
                        name: {
                            "role": details.get('role', ''),
                            "fixedqty": details.get('fixedqty', 0),
                            "type": details.get('type', ''),
                            "topology": details.get('topology', ''),
                            "switches": switches_list
                        }
                    }
                    config_dict['lan_spines'].append(lan_spine)
            
            if 'Cables' in exec_globals:
                config_dict['cables'] = []
                for switch_model, cables in exec_globals['Cables'].items():
                    config_dict['cables'].append({switch_model: cables})
            
            config = config_dict
        
        # Update data structures
        if 'project_name' in config:
            project_name = config['project_name']
        
        if 'switches' in config:
            switches = config['switches']
        
        if 'lan_leaves' in config:
            lan_leaves = config['lan_leaves']
            # Update LAN roles
            for leaf in lan_leaves:
                for lan_name, details in leaf.items():
                    if 'role' in details:
                        lan_roles[lan_name] = details['role']
        
        if 'lan_spines' in config:
            lan_spines = config['lan_spines']
        
        if 'compute_nodes' in config:
            compute_entries = config['compute_nodes']
        
        if 'storage_blocks' in config:
            storage_entries = config['storage_blocks']
        
        if 'cables' in config:
            cables_entries = config['cables']
        
        if 'rack_rows' in config:
            rack_rows_entries = config['rack_rows']
        
        # Save configuration
        save_configuration()
        
        return jsonify({
            "status": "success",
            "message": "Configuration imported successfully"
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error importing configuration: {str(e)}"
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)

