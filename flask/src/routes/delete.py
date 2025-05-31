from flask import Blueprint, request, jsonify

delete_bp = Blueprint('delete', __name__)

@delete_bp.route('/delete_switch', methods=['POST'])
def delete_switch():
    from main import switches, lan_leaves, lan_spines, save_configuration
    
    # Get form data
    index = request.form.get('index')
    model = request.form.get('model')
    
    if not index or not index.isdigit():
        return jsonify({
            "status": "error",
            "message": "Invalid switch index"
        })
    
    index = int(index)
    
    if index < 0 or index >= len(switches):
        return jsonify({
            "status": "error",
            "message": "Switch index out of range"
        })
    
    # Get the switch model to be deleted
    switch_to_delete = switches[index]['model']
    
    # Remove the switch
    deleted_switch = switches.pop(index)
    
    # Remove references to this switch from LAN leaves
    for leaf in lan_leaves:
        for lan_name, details in leaf.items():
            if 'switches' in details and switch_to_delete in details['switches']:
                details['switches'].remove(switch_to_delete)
    
    # Remove references to this switch from LAN spines
    for spine in lan_spines:
        for spine_name, details in spine.items():
            if 'switches' in details and switch_to_delete in details['switches']:
                details['switches'].remove(switch_to_delete)
    
    # Save configuration
    save_configuration()
    
    return jsonify({
        "status": "success",
        "message": f"Switch '{switch_to_delete}' deleted successfully and all references removed",
        "all_entries": switches
    })
