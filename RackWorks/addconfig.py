import re

# Your original variable declaration string
var_declaration = """
colors_info = LANs = Cables = max_box_wattage = max_box_height = rack_height_u = rack_weight_kg = rack_width_mm = None
rack_depth_mm = u_height_mm = rack_height_mm = rack_height_u = rack_cg_height_mm = unit_to_cm = u_height_mm = None
device_to_rackside = racktop_to_ceiling = rack_to_rack = batch_size = max_iter = colors = switch_to_add = None
color_metadata_tuple = colors_weight = colors_height = colors_wattage = total_balls = None
"""

def extract_variable_names(declaration):
    """Extract all variable names from the declaration string"""
    # Remove '= None' parts and split by '='
    cleaned = re.sub(r'=\s*None', '', declaration)
    variables = [v.strip() for v in cleaned.split('=') if v.strip()]
    return variables

def generate_refactored_code(variables):
    """Generate the config import and alias code"""
    config_import = ""
    alias_lines = [f"{var} = config.{var}" for var in variables]
    return config_import + "\n".join(alias_lines) + "\n"

def refactor_file(input_file, output_file, variables):
    """Refactor a Python file to use config.variable pattern"""
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Replace standalone variable names (not preceded by . or word chars)
    for var in variables:
        content = re.sub(rf'(?<![\.\w]){var}(?![\.\w])', f'config.{var}', content)
    
    with open(output_file, 'w') as f:
        f.write(content)

# Extract variable names
variables = extract_variable_names(var_declaration)
print("Found variables:", variables)

# Generate the alias code for your files
alias_code = generate_refactored_code(variables)
print("\nAdd this to the top of your files:\n")
print(alias_code)

# Example usage to refactor rac.py
refactor_file('RackUtils.py', 'RackUtils.py', variables)
