import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # DON'T CHANGE THIS !!!

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Store submitted entries in memory (in a real app, you'd use a database)
# Pre-populate LAN leaves with default values from the provided dictionary
lan_leaves_entries = [
    {'LAN_1': {'fixedqty': 12, 'type': '400gbsNDR', 'topology': 'halfports_spine_leaf',
              'switch': [{'model': 'IB+400', 'ports': 64, 'speed': 400, 'height': 2, 'wattage': 2000, 'weight': 20}]}},
    {'LAN_2': {'fixedqty': 14, 'type': '400GbpsEth', 'topology': 'uplinks_spine_leaf',
              'switch': [{'model': 'Z_9xxx', 'ports': 64, 'speed': 400, 'height': 2, 'wattage': 1304, 'uplink_count': 4, 'uplink_speed': 800, 'weight': 20}]}},
    {'LAN_4': {'fixedqty': 4, 'type': '25gbps', 'topology': 'uplinks_spine_leaf',
              'switch': [{'model': 'S_xx64', 'ports': 64, 'speed': 25, 'height': 2, 'wattage': 300, 'uplink_count': 4, 'uplink_speed': 100, 'weight': 12}]}},
    {'LAN_5': {'fixedqty': 4, 'type': '25gbps', 'topology': 'uplinks_spine_leaf',
              'switch': [{'model': 'S_xx64', 'ports': 64, 'speed': 25, 'height': 2, 'wattage': 200, 'uplink_count': 4, 'uplink_speed': 100, 'weight': 12}]}},
    {'LAN_6': {'fixedqty': 1000, 'type': '1gbps', 'topology': 'uplinks_spine_leaf',
              'switch': [{'model': 'SN_24', 'ports': 24, 'speed': 1, 'height': 1, 'wattage': 200, 'uplink_count': 2, 'uplink_speed': 25, 'weight': 6}]}},
    {'LAN_3': {'fixedqty': 1, 'type': '400GbpsEth', 'topology': 'rails',
              'switch': [{'model': 'Z_9xxx', 'ports': 64, 'speed': 400, 'height': 2, 'wattage': 2200, 'uplink_count': 4, 'uplink_speed': 800, 'weight': 20}]}}
]

# Pre-populate LAN spines with default values from the provided dictionary
lan_spines_entries = [
    {'LAN_1_spines': {'fixedqty': 8, 'type': '400gbsNDR', 'topology': 'halfports_spine_leaf',
                     'switch': [{'model': 'IB+400', 'ports': 64, 'speed': 400, 'height': 2, 'wattage': 2000, 'weight': 20}]}},
    {'LAN_2_spines': {'fixedqty': 8, 'type': '400GbpsEth', 'topology': 'uplinks_spine_leaf',
                     'switch': [{'model': 'Z_9xxx', 'ports': 64, 'speed': 400, 'height': 2, 'wattage': 1304, 'uplink_count': 4, 'uplink_speed': 800, 'weight': 20}]}},
    {'LAN_4_spines': {'fixedqty': 2, 'type': '25gbps', 'topology': 'uplinks_spine_leaf',
                     'switch': [{'model': 'S_xx64', 'ports': 64, 'speed': 25, 'height': 2, 'wattage': 300, 'uplink_count': 4, 'uplink_speed': 100, 'weight': 12}]}},
    {'LAN_5_spines': {'fixedqty': 2, 'type': '25gbps', 'topology': 'uplinks_spine_leaf',
                     'switch': [{'model': 'S_xx64', 'ports': 64, 'speed': 25, 'height': 2, 'wattage': 200, 'uplink_count': 4, 'uplink_speed': 100, 'weight': 12}]}},
    {'LAN_6_spines': {'fixedqty': 1000, 'type': '1gbps', 'topology': 'uplinks_spine_leaf',
                     'switch': [{'model': 'SN_24', 'ports': 24, 'speed': 1, 'height': 1, 'wattage': 200, 'uplink_count': 2, 'uplink_speed': 25, 'weight': 6}]}},
    {'LAN_3_spines': {'fixedqty': 0, 'type': '400GbpsEth', 'topology': 'rails',
                     'switch': [{'model': 'Z_9xxx', 'ports': 64, 'speed': 400, 'height': 2, 'wattage': 2200, 'uplink_count': 4, 'uplink_speed': 800, 'weight': 20}]}}
]

compute_entries = []
storage_entries = []
cables_entries = []
rack_rows_entries = []

@app.route('/')
def index():
    return render_template('index.html', 
                          lan_leaves=lan_leaves_entries,
                          lan_spines=lan_spines_entries,
                          compute_entries=compute_entries,
                          storage_entries=storage_entries,
                          cables_entries=cables_entries,
                          rack_rows_entries=rack_rows_entries)

@app.route('/submit_compute', methods=['POST'])
def submit_compute():
    # Get form data from AJAX request
    compute_name = request.form.get('compute_name')
    count = request.form.get('count')
    wattage = request.form.get('wattage')
    height = request.form.get('height')
    weight = request.form.get('weight')
    
    # Get LAN data
    lan_data = {}
    for i in range(1, 7):  # LAN_1 through LAN_6
        lan_count = request.form.get(f'lan_{i}_count')
        lan_speed = request.form.get(f'lan_{i}_speed')
        lan_data[f'LAN_{i}'] = {
            'count': int(lan_count) if lan_count else 0,
            'speed': int(lan_speed) if lan_speed else 0
        }
    
    # Create entry dictionary
    entry = {
        compute_name: {
            'count': int(count) if count else 0,
            'wattage': float(wattage) if wattage else 0,
            'height': float(height) if height else 0,
            'weight': float(weight) if weight else 0,
            **lan_data
        }
    }
    
    # Add to entries list
    compute_entries.append(entry)
    
    # Return response with all entries
    response = {
        'status': 'success',
        'message': 'Compute entry added successfully',
        'entry': entry,
        'all_entries': compute_entries
    }
    
    return jsonify(response)

@app.route('/submit_lan_leaf', methods=['POST'])
def submit_lan_leaf():
    # Get form data from AJAX request
    lan_name = request.form.get('lan_name')
    fixed_qty = request.form.get('fixed_qty')
    lan_type = request.form.get('lan_type')
    topology = request.form.get('topology')
    
    # Get switch data
    switch_model = request.form.get('switch_model')
    switch_ports = request.form.get('switch_ports')
    switch_speed = request.form.get('switch_speed')
    switch_height = request.form.get('switch_height')
    switch_wattage = request.form.get('switch_wattage')
    switch_weight = request.form.get('switch_weight')
    
    # Optional uplink data
    uplink_count = request.form.get('uplink_count')
    uplink_speed = request.form.get('uplink_speed')
    
    # Create switch dictionary
    switch_dict = {
        'model': switch_model,
        'ports': int(switch_ports) if switch_ports else 0,
        'speed': int(switch_speed) if switch_speed else 0,
        'height': int(switch_height) if switch_height else 0,
        'wattage': int(switch_wattage) if switch_wattage else 0,
        'weight': int(switch_weight) if switch_weight else 0
    }
    
    # Add uplink data if provided
    if uplink_count and uplink_speed:
        switch_dict['uplink_count'] = int(uplink_count)
        switch_dict['uplink_speed'] = int(uplink_speed)
    
    # Create entry dictionary
    entry = {
        lan_name: {
            'fixedqty': int(fixed_qty) if fixed_qty else 0,
            'type': lan_type,
            'topology': topology,
            'switch': [switch_dict]
        }
    }
    
    # Check if editing an existing entry
    edit_index = request.form.get('edit_index')
    if edit_index and edit_index.isdigit():
        index = int(edit_index)
        if 0 <= index < len(lan_leaves_entries):
            lan_leaves_entries[index] = entry
            message = 'LAN leaf entry updated successfully'
        else:
            lan_leaves_entries.append(entry)
            message = 'LAN leaf entry added successfully'
    else:
        lan_leaves_entries.append(entry)
        message = 'LAN leaf entry added successfully'
    
    # Return response with all entries
    response = {
        'status': 'success',
        'message': message,
        'entry': entry,
        'all_entries': lan_leaves_entries
    }
    
    return jsonify(response)

@app.route('/submit_lan_spine', methods=['POST'])
def submit_lan_spine():
    # Get form data from AJAX request
    spine_name = request.form.get('spine_name')
    fixed_qty = request.form.get('fixed_qty')
    spine_type = request.form.get('spine_type')
    topology = request.form.get('topology')
    
    # Get switch data
    switch_model = request.form.get('switch_model')
    switch_ports = request.form.get('switch_ports')
    switch_speed = request.form.get('switch_speed')
    switch_height = request.form.get('switch_height')
    switch_wattage = request.form.get('switch_wattage')
    switch_weight = request.form.get('switch_weight')
    
    # Optional uplink data
    uplink_count = request.form.get('uplink_count')
    uplink_speed = request.form.get('uplink_speed')
    
    # Create switch dictionary
    switch_dict = {
        'model': switch_model,
        'ports': int(switch_ports) if switch_ports else 0,
        'speed': int(switch_speed) if switch_speed else 0,
        'height': int(switch_height) if switch_height else 0,
        'wattage': int(switch_wattage) if switch_wattage else 0,
        'weight': int(switch_weight) if switch_weight else 0
    }
    
    # Add uplink data if provided
    if uplink_count and uplink_speed:
        switch_dict['uplink_count'] = int(uplink_count)
        switch_dict['uplink_speed'] = int(uplink_speed)
    
    # Create entry dictionary
    entry = {
        spine_name: {
            'fixedqty': int(fixed_qty) if fixed_qty else 0,
            'type': spine_type,
            'topology': topology,
            'switch': [switch_dict]
        }
    }
    
    # Check if editing an existing entry
    edit_index = request.form.get('edit_index')
    if edit_index and edit_index.isdigit():
        index = int(edit_index)
        if 0 <= index < len(lan_spines_entries):
            lan_spines_entries[index] = entry
            message = 'LAN spine entry updated successfully'
        else:
            lan_spines_entries.append(entry)
            message = 'LAN spine entry added successfully'
    else:
        lan_spines_entries.append(entry)
        message = 'LAN spine entry added successfully'
    
    # Return response with all entries
    response = {
        'status': 'success',
        'message': message,
        'entry': entry,
        'all_entries': lan_spines_entries
    }
    
    return jsonify(response)

@app.route('/submit_cable', methods=['POST'])
def submit_cable():
    # Get form data from AJAX request
    switch_model = request.form.get('switch_model')
    cable_model = request.form.get('cable_model')
    server_port_speed = request.form.get('server_port_speed')
    split = request.form.get('split')
    length = request.form.get('length')
    
    # Create cable dictionary
    cable_dict = {
        'model': cable_model,
        'server_port_speed': int(server_port_speed) if server_port_speed else 0,
        'split': int(split) if split else 1,
        'length': float(length) if length else 0
    }
    
    # Check if switch model already exists in cables_entries
    switch_exists = False
    for entry in cables_entries:
        if switch_model in entry:
            entry[switch_model].append(cable_dict)
            switch_exists = True
            break
    
    # If switch model doesn't exist, create new entry
    if not switch_exists:
        entry = {
            switch_model: [cable_dict]
        }
        cables_entries.append(entry)
    
    # Return response with all entries
    response = {
        'status': 'success',
        'message': 'Cable entry added successfully',
        'all_entries': cables_entries
    }
    
    return jsonify(response)

@app.route('/submit_rack_row', methods=['POST'])
def submit_rack_row():
    # Get form data from AJAX request
    group_name = request.form.get('group_name')
    racks = request.form.get('racks')
    group_count = request.form.get('group_count')
    rack_to_rack = request.form.get('rack_to_rack')
    row_to_next_row = request.form.get('row_to_next_row')
    
    # Create entry dictionary
    entry = {
        'Group': {
            'Racks': int(racks) if racks else 0,
            'group_count': int(group_count) if group_count else 0,
            'rack_to_rack': float(rack_to_rack) if rack_to_rack else 0,
            'Row_to_next_row': float(row_to_next_row) if row_to_next_row else 0
        }
    }
    
    # Add to entries list
    rack_rows_entries.append(entry)
    
    # Return response with all entries
    response = {
        'status': 'success',
        'message': 'Rack row entry added successfully',
        'entry': entry,
        'all_entries': rack_rows_entries
    }
    
    return jsonify(response)

@app.route('/get_lan_leaf', methods=['GET'])
def get_lan_leaf():
    index = request.args.get('index')
    if index and index.isdigit():
        index = int(index)
        if 0 <= index < len(lan_leaves_entries):
            return jsonify({
                'status': 'success',
                'entry': lan_leaves_entries[index],
                'index': index
            })
    return jsonify({
        'status': 'error',
        'message': 'Invalid index or entry not found'
    })

@app.route('/get_lan_spine', methods=['GET'])
def get_lan_spine():
    index = request.args.get('index')
    if index and index.isdigit():
        index = int(index)
        if 0 <= index < len(lan_spines_entries):
            return jsonify({
                'status': 'success',
                'entry': lan_spines_entries[index],
                'index': index
            })
    return jsonify({
        'status': 'error',
        'message': 'Invalid index or entry not found'
    })

@app.route('/get_all_entries', methods=['GET'])
def get_all_entries():
    all_data = {
        'lan_leaves': lan_leaves_entries,
        'lan_spines': lan_spines_entries,
        'compute_nodes': compute_entries,
        'storage_blocks': storage_entries,
        'cables': cables_entries,
        'rack_rows': rack_rows_entries
    }
    return jsonify(all_data)

@app.route('/clear_entries', methods=['POST'])
def clear_entries():
    category = request.form.get('category', 'all')
    
    if category == 'all':
        lan_leaves_entries.clear()
        lan_spines_entries.clear()
        compute_entries.clear()
        storage_entries.clear()
        cables_entries.clear()
        rack_rows_entries.clear()
        message = 'All entries cleared'
    elif category == 'lan_leaves':
        lan_leaves_entries.clear()
        message = 'LAN leaves entries cleared'
    elif category == 'lan_spines':
        lan_spines_entries.clear()
        message = 'LAN spines entries cleared'
    elif category == 'compute':
        compute_entries.clear()
        message = 'Compute entries cleared'
    elif category == 'storage':
        storage_entries.clear()
        message = 'Storage entries cleared'
    elif category == 'cables':
        cables_entries.clear()
        message = 'Cables entries cleared'
    elif category == 'rack_rows':
        rack_rows_entries.clear()
        message = 'Rack rows entries cleared'
    
    return jsonify({'status': 'success', 'message': message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
