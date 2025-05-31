import os
import ast
import json
import re
from flask import Blueprint, request, jsonify, current_app

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload_config_file', methods=['POST'])
def upload_config_file():
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
    
    if not file.filename.endswith('.txt'):
        return jsonify({
            "status": "error",
            "message": "Only .txt files are supported"
        })
    
    # Get data handling option
    data_option = request.form.get('data_option', 'merge')
    
    try:
        # Read file content
        content = file.read().decode('utf-8')
        
        # Parse the content
        parsed_data = parse_config_file(content)
        
        # Process the parsed data
        result = process_parsed_data(parsed_data, data_option)
        
        return jsonify({
            "status": "success",
            "message": "File uploaded and processed successfully",
            "result": result
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing file: {str(e)}"
        })

def clean_python_code(code):
    """Clean Python code by removing comments and fixing trailing commas."""
    # Remove Python comments
    code = re.sub(r'#.*?$', '', code, flags=re.MULTILINE)
    
    # Fix trailing commas in dictionaries and lists
    # This is a simple approach - for complex cases, more sophisticated parsing might be needed
    code = re.sub(r',\s*}', '}', code)
    code = re.sub(r',\s*]', ']', code)
    
    return code

def parse_config_file(content):
    """Parse the content of the uploaded file into Python dictionaries."""
    result = {}
    
    # Extract dictionary assignments
    lines = content.split('\n')
    current_dict = ""
    current_dict_name = ""
    
    for line in lines:
        # Remove comments from the line
        line = re.sub(r'#.*?$', '', line)
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Check if this is the start of a new dictionary
        if '=' in line and '{' in line:
            # If we were building a previous dictionary, add it to the result
            if current_dict:
                try:
                    # Clean the dictionary string before evaluation
                    cleaned_dict = clean_python_code(current_dict)
                    # Use ast.literal_eval to safely evaluate the string as a Python literal
                    result[current_dict_name] = ast.literal_eval(cleaned_dict)
                except (SyntaxError, ValueError) as e:
                    # If there's an error, store the raw string for debugging
                    result[current_dict_name] = {"error": str(e), "raw": current_dict}
            
            # Start a new dictionary
            parts = line.split('=', 1)
            current_dict_name = parts[0].strip()
            current_dict = parts[1].strip()
        else:
            # Continue building the current dictionary
            if current_dict:
                current_dict += line
    
    # Add the last dictionary if there is one
    if current_dict:
        try:
            # Clean the dictionary string before evaluation
            cleaned_dict = clean_python_code(current_dict)
            result[current_dict_name] = ast.literal_eval(cleaned_dict)
        except (SyntaxError, ValueError) as e:
            result[current_dict_name] = {"error": str(e), "raw": current_dict}
    
    return result

def process_parsed_data(parsed_data, data_option):
    """Process the parsed data and update the application's data structures."""
    from main import switches, lan_leaves, lan_spines, compute_entries, storage_entries, cables_entries, rack_rows_entries, save_configuration
    
    result = {
        "switches": {"added": 0, "updated": 0},
        "lan_leaves": {"added": 0, "updated": 0},
        "lan_spines": {"added": 0, "updated": 0},
        "compute_entries": {"added": 0, "updated": 0},
        "storage_entries": {"added": 0, "updated": 0},
        "cables_entries": {"added": 0, "updated": 0},
        "rack_rows_entries": {"added": 0, "updated": 0}
    }
    
    # Clear existing data if replace option is selected
    if data_option == 'replace':
        switches.clear()
        lan_leaves.clear()
        lan_spines.clear()
        compute_entries.clear()
        storage_entries.clear()
        cables_entries.clear()
        rack_rows_entries.clear()
    
    # Process LANs
    if 'LANs' in parsed_data:
        lans_data = parsed_data['LANs']
        for lan_name, lan_info in lans_data.items():
            # Extract switches from LAN and add them to the switches section
            if 'switch' in lan_info and isinstance(lan_info['switch'], list):
                for switch_data in lan_info['switch']:
                    if isinstance(switch_data, dict):
                        # Create a copy of the switch data
                        switch_entry = switch_data.copy()
                        
                        # Check if this switch model already exists in switches
                        existing_switch_index = -1
                        for i, existing_switch in enumerate(switches):
                            if existing_switch.get('model') == switch_entry.get('model'):
                                existing_switch_index = i
                                break
                        
                        if existing_switch_index >= 0:
                            # Update existing switch
                            switches[existing_switch_index] = switch_entry
                            result["switches"]["updated"] += 1
                        else:
                            # Add new switch
                            switches.append(switch_entry)
                            result["switches"]["added"] += 1
            
            # Create LAN leaf entry in the format expected by the application
            lan_leaf_entry = {
                lan_name: {
                    "role": lan_info.get('type', ''),
                    "fixedqty": lan_info.get('fixedqty', 0),
                    "type": lan_info.get('type', ''),
                    "topology": lan_info.get('topology', ''),
                    "switch": lan_info.get('switch', [])
                }
            }
            
            # Check if this LAN already exists
            existing_index = -1
            for i, leaf in enumerate(lan_leaves):
                if lan_name in leaf:
                    existing_index = i
                    break
            
            if existing_index >= 0:
                lan_leaves[existing_index] = lan_leaf_entry
                result["lan_leaves"]["updated"] += 1
            else:
                lan_leaves.append(lan_leaf_entry)
                result["lan_leaves"]["added"] += 1
    
    # Process Spines
    if 'Spines' in parsed_data:
        spines_data = parsed_data['Spines']
        for spine_name, spine_info in spines_data.items():
            # Extract switches from Spines and add them to the switches section
            if 'switch' in spine_info and isinstance(spine_info['switch'], list):
                for switch_data in spine_info['switch']:
                    if isinstance(switch_data, dict):
                        # Create a copy of the switch data
                        switch_entry = switch_data.copy()
                        
                        # Check if this switch model already exists in switches
                        existing_switch_index = -1
                        for i, existing_switch in enumerate(switches):
                            if existing_switch.get('model') == switch_entry.get('model'):
                                existing_switch_index = i
                                break
                        
                        if existing_switch_index >= 0:
                            # Update existing switch
                            switches[existing_switch_index] = switch_entry
                            result["switches"]["updated"] += 1
                        else:
                            # Add new switch
                            switches.append(switch_entry)
                            result["switches"]["added"] += 1
            
            # Create LAN spine entry in the format expected by the application
            lan_spine_entry = {
                spine_name: {
                    "role": spine_info.get('type', ''),
                    "fixedqty": spine_info.get('fixedqty', 0),
                    "type": spine_info.get('type', ''),
                    "topology": spine_info.get('topology', ''),
                    "switch": spine_info.get('switch', [])
                }
            }
            
            # Check if this spine already exists
            existing_index = -1
            for i, spine in enumerate(lan_spines):
                if spine_name in spine:
                    existing_index = i
                    break
            
            if existing_index >= 0:
                lan_spines[existing_index] = lan_spine_entry
                result["lan_spines"]["updated"] += 1
            else:
                lan_spines.append(lan_spine_entry)
                result["lan_spines"]["added"] += 1
    
    # Process Compute Entries (colors_info)
    if 'colors_info' in parsed_data:
        colors_info = parsed_data['colors_info']
        for compute_name, compute_info in colors_info.items():
            # Create compute entry in the format expected by the application
            compute_entry = {
                compute_name: {
                    "count": compute_info.get('count', 0),
                    "wattage": compute_info.get('wattage', 0),
                    "height": compute_info.get('height', 0),
                    "weight": compute_info.get('weight', 0)
                }
            }
            
            # Add LAN details
            for i in range(1, 7):
                lan_key = f'LAN_{i}'
                if lan_key in compute_info:
                    compute_entry[compute_name][lan_key] = compute_info[lan_key]
            
            # Check if this compute entry already exists
            existing_index = -1
            for i, entry in enumerate(compute_entries):
                if compute_name in entry:
                    existing_index = i
                    break
            
            if existing_index >= 0:
                compute_entries[existing_index] = compute_entry
                result["compute_entries"]["updated"] += 1
            else:
                compute_entries.append(compute_entry)
                result["compute_entries"]["added"] += 1
    
    # Process Rack Rows
    if 'Rack_rows' in parsed_data:
        rack_rows_data = parsed_data['Rack_rows']
        for group_name, group_info in rack_rows_data.items():
            # Create rack row entry in the format expected by the application
            rack_row_entry = {
                group_name: {
                    "Racks": group_info.get('Racks', 0),
                    "group_count": group_info.get('group_count', 0),
                    "rack_to_rack": group_info.get('rack_to_rack', 0),
                    "Row_to_next_row": group_info.get('Row_to_next_row', 0)
                }
            }
            
            # Check if this rack row already exists
            existing_index = -1
            for i, entry in enumerate(rack_rows_entries):
                if group_name in entry:
                    existing_index = i
                    break
            
            if existing_index >= 0:
                rack_rows_entries[existing_index] = rack_row_entry
                result["rack_rows_entries"]["updated"] += 1
            else:
                rack_rows_entries.append(rack_row_entry)
                result["rack_rows_entries"]["added"] += 1
    
    # Process Cables
    if 'Cables' in parsed_data:
        cables_data = parsed_data['Cables']
        for switch_model, cable_list in cables_data.items():
            # Create cable entry in the format expected by the application
            cable_entry = {
                switch_model: cable_list
            }
            
            # Check if this cable entry already exists
            existing_index = -1
            for i, entry in enumerate(cables_entries):
                if switch_model in entry:
                    existing_index = i
                    break
            
            if existing_index >= 0:
                cables_entries[existing_index] = cable_entry
                result["cables_entries"]["updated"] += 1
            else:
                cables_entries.append(cable_entry)
                result["cables_entries"]["added"] += 1
    
    # Save the updated configuration
    save_configuration()
    
    return result
