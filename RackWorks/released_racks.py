#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


colors_info = LANs = Cable = max_box_wattag =  max_box_height = rack_height_u =  rack_weight_kg = rack_width_mm = None
rack_depth_mm = u_height_mm =  rack_height_mm = rack_height_u =  rack_cg_height_mm =  unit_to_cm = u_height_mm = None
device_to_rackside = racktop_to_ceiling = rack_to_rack = batch_size = max_iter = None


# In[ ]:


def get_the_right_cable(switch,absolute_sw_to_dev, devspeed):
    
    cables = Cables[switch]

    
    higher_lengths =(
     sorted([item for item in cables if isinstance(item.get('length'), 
                                    (int, float)) and item.get('length') > absolute_sw_to_dev
                                    and item.get('server_port_speed') == devspeed], key=lambda x: x['length'])
    )
    if higher_lengths:
        
        return higher_lengths[0]
    else:
        return {}


# In[ ]:


def calculate_cable_lengths(rack):
    
    lans = dict()
    devices = dict()
    switches = dict()
    for dev in rack:
        dev_id = dev.split('_')[-1]
        device = '_'.join(dev.split('_')[:-1])
        if 'LAN' in device:
            lanname, switch = device.split('__')
            lans[dev] = dict()
            device_specs = [x for x in LANs[lanname]['switch'] if x['model'] == switch ][0]
            lans[dev] = device_specs.copy()
            lans[dev]['position'] = rack[dev]
            lans[dev]['id'] = dev_id
            lans[dev]['half_for_split'] = 0
            lans[dev]['full_no_split'] = 0
            lans[dev]['cables'] = dict()
            lans[dev]['minimum_cable_total_length'] = 0
            lans[dev]['actual_cable_total_length'] = 0
            if device not in switches:
                switches[device] = 0
            switches[device] +=1 
         
    for lan in lans:
        lanname = lan.split('__')[0]
        switch = lans[lan]['model']
        cable_to_ceil = 0
        for dev in rack:
            dev_id = dev.split('_')[-1]
            device = '_'.join(dev.split('_')[:-1])
            
            if 'LAN' not in device :
                if colors_info[device][lanname]['count'] > 0:
                    if dev not in devices:
                        devices[dev] = dict()
                    devices[dev]['position'] = rack[dev]
                    devices[dev]['id'] = dev_id
                    no_ports = colors_info[device][lanname]['count']
                    speed_ratio =  colors_info[device][lanname]['speed'] / lans[lan]['speed']
                    device_position = rack[dev]
                    # absolute cable lengeth in meters:
                    absolute_sw_to_dev = ((lans[lan]['position'] - device_position) + 2*device_to_rackside)*unit_to_cm/100
                    cable = get_the_right_cable(switch,absolute_sw_to_dev,colors_info[device][lanname]['speed'])
                    no_of_cables = colors_info[device][lanname]['count'] * speed_ratio
                    devices[dev][lan] = dict()
                    devices[dev][lan]['cable_count'] = colors_info[device][lanname]['count']
                    devices[dev][lan]['speed'] = colors_info[device][lanname]['speed']
                    if speed_ratio < 1:
                        lans[lan]['half_for_split'] +=1
                        devices[dev][lan]['half_for_split'] = 1
                        cable_to_ceil = cable['model']
                    else:
                        lans[lan]['full_no_split'] += 1
                        devices[dev][lan]['full_no_split'] = 1
                    devices[dev][lan]['cables_count'] = no_of_cables
                    devices[dev][lan]['cable_type'] = cable
                    if cable['model'] not in lans[lan]['cables']:
                        lans[lan]['cables'][cable['model']] = 0                    
                    lans[lan]['cables'][cable['model']] += no_of_cables
                    lans[lan]['minimum_cable_total_length'] += absolute_sw_to_dev * no_of_cables
                    lans[lan]['actual_cable_total_length'] += cable['length'] * no_of_cables
        if cable_to_ceil:
            lans[lan]['cables'][cable_to_ceil] = ceil(lans[lan]['cables'][cable_to_ceil])
    
    return (devices, lans, switches)


# rack = {'LAN_1__IB+400_1': 41, 'FrontEnd_nodes_2': 40, 'GPU_nodes_3': 32, 'compute_nodes_4': 30, 'compute_nodes_5': 28, 'compute_nodes_6': 26, 'compute_nodes_7': 24, 'compute_nodes_8': 22}
# lans, devices  = calculate_cable_lengths(rack, LANs, Cables, colors_info)
# print('devices',devices)
# print('lans',lans)

# In[ ]:





# In[ ]:


def sort_device_before_placement(all_items):
    lan_devices = sorted([item for item in all_items if 'LAN' in item['type']], key=lambda x: x['type'])
    other_devices = [item for item in all_items if 'LAN' not in item['type']]

    def sort_other(item):
        sort_key = [1] * len(lan_devices) + [item.get('type', '')] # Initialize with a lower priority and type for tie-breaking

        for i, lan_item in enumerate(lan_devices):
                for key, value in item.items():
                    if isinstance(value, dict):
                        sort_key[i] = -item[key].get('count', 0) # Higher count gets a higher (negative) priority
                        break # Move to the next LAN device after finding a match
        return tuple(sort_key)

    sorted_other_devices = sorted(other_devices, key=sort_other)
    return lan_devices + sorted_other_devices


# all_servers = [{'model': 'IB+400', 'ports': 64, 'speed': 400, 'height': 2, 'wattage': 2000, 'weight': 20, 'type': 'LAN_3__IB+400'}, {'count': 20, 'wattage': 11000, 'height': 8, 'weight': 15, 'LAN_1': {'count':0, 'speed': 400}, 'LAN_2': {'count': 1, 'speed': 200}, 'LAN_3': {'count': 1, 'speed': 25}, 'LAN_4': {'count': 1, 'speed': 25}, 'LAN_5': {'count': 1, 'speed': 1}, 'LAN_6': {'count': 8, 'speed': 400}, 'type': 'GPU_nodes'}, {'count': 20, 'wattage': 1600, 'height': 2, 'weight': 30, 'LAN_1': {'count': 1, 'speed': 400}, 'LAN_2': {'count': 1, 'speed': 200}, 'LAN_3': {'count': 1, 'speed': 25}, 'LAN_4': {'count': 1, 'speed': 25}, 'LAN_5': {'count': 1, 'speed': 1}, 'LAN_6': {'count': 0, 'speed': 400}, 'type': 'compute_nodes'}, {'count': 20, 'wattage': 1600, 'height': 2, 'weight': 30, 'LAN_1': {'count': 1, 'speed': 400}, 'LAN_2': {'count': 1, 'speed': 200}, 'LAN_3': {'count': 1, 'speed': 25}, 'LAN_4': {'count': 1, 'speed': 25}, 'LAN_5': {'count': 1, 'speed': 1}, 'LAN_6': {'count': 0, 'speed': 400}, 'type': 'compute_nodes'}, {'count': 20, 'wattage': 1600, 'height': 2, 'weight': 30, 'LAN_1': {'count': 1, 'speed': 400}, 'LAN_2': {'count': 1, 'speed': 200}, 'LAN_3': {'count': 1, 'speed': 25}, 'LAN_4': {'count': 1, 'speed': 25}, 'LAN_5': {'count': 1, 'speed': 1}, 'LAN_6': {'count': 0, 'speed': 400}, 'type': 'compute_nodes'}, {'count': 20, 'wattage': 1600, 'height': 2, 'weight': 30, 'LAN_1': {'count': 1, 'speed': 400}, 'LAN_2': {'count': 1, 'speed': 200}, 'LAN_3': {'count': 1, 'speed': 25}, 'LAN_4': {'count': 1, 'speed': 25}, 'LAN_5': {'count': 1, 'speed': 1}, 'LAN_6': {'count': 0, 'speed': 400}, 'type': 'compute_nodes'}]
# 
# sorted_servers = sort_device_before_placement(all_servers)
# print(sorted_servers)

# In[ ]:





# In[ ]:


def old_sort_device_before_placement(item):
    if 'LAN' in item['type']:
        return (0, item['type'])  # Prioritize LAN, then sort alphabetically by type
    else:
        return (1, item['type'])  # Other types come later, maintain original order



# In[ ]:


10//4


# In[ ]:





# In[ ]:


import numpy as np
from collections import Counter
import functools
import time
from functools import cache, lru_cache, wraps
global_rack_signature = dict()

def is_rack_stable(servers):
    """
    Checks if the rack configuration is likely stable based on the combined
    vertical center of gravity.
    """
    total_weight = rack_weight_kg + sum(s[1] for s in servers)
    if total_weight == 0:
        return True

    combined_vertical_cg = (rack_weight_kg * rack_cg_height_mm +
                             sum(s[1] * s[0] for s in servers)) / total_weight

    stability_threshold_fraction = 0.5  # Adjust as needed
    return combined_vertical_cg <= rack_height_mm * stability_threshold_fraction



import collections
from functools import wraps

def to_hashable(obj):
    """Convert objects to hashable forms while preserving structure."""
    if isinstance(obj, Counter):
        # For Counter, convert to tuple of items
        return ('__counter__', tuple(sorted(obj.items())))
    elif isinstance(obj, dict):
        # For dicts, convert to tuple of sorted items
        return ('__dict__', tuple(sorted((k, to_hashable(v)) for k, v in obj.items())))
    elif isinstance(obj, list):
        return ('__list__', tuple(to_hashable(item) for item in obj))
    elif isinstance(obj, tuple):
        return ('__tuple__', tuple(to_hashable(item) for item in obj)) 
    elif isinstance(obj, set):
        return ('__set__', frozenset(to_hashable(item) for item in obj))
    return obj

def from_hashable(obj):
    """Convert back from hashable form to original objects."""
    if isinstance(obj, tuple) and len(obj) == 2:
        type_tag, value = obj
        if type_tag == '__counter__':
            # Rebuild Counter from items
            return Counter(dict(value))
        elif type_tag == '__dict__':
            # Rebuild dict
            return {k: from_hashable(v) for k, v in value}
        elif type_tag == '__list__':
            # Rebuild list
            return [from_hashable(item) for item in value]
        elif type_tag == '__tuple__':
            # Keep as tuple but convert contents
            return tuple(from_hashable(item) for item in value)
        elif type_tag == '__set__':
            # Rebuild set
            return {from_hashable(item) for item in value}
    
    # If it's a regular tuple (not tagged), process its elements
    if isinstance(obj, tuple):
        return tuple(from_hashable(item) for item in obj)
    
    return obj

def hashable_cache(func):
    """
    Decorator that makes function arguments hashable for caching,
    then converts them back to their original types when calling the function.
    """
    @functools.lru_cache(maxsize=None)
    def cached_wrapper(*hashable_args, **hashable_kwargs):
        # Convert the hashable arguments back to their original types
        restored_args = tuple(from_hashable(arg) for arg in hashable_args)
        restored_kwargs = {k: from_hashable(v) for k, v in hashable_kwargs.items()}
        
        # Call the original function with restored arguments
        return func(*restored_args, **restored_kwargs)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert arguments to hashable versions
        hashable_args = tuple(to_hashable(arg) for arg in args)
        hashable_kwargs = {k: to_hashable(v) for k, v in kwargs.items()}
        
        # Call the cached version
        return cached_wrapper(*hashable_args, **hashable_kwargs)
    
    # Add cache control methods
    wrapper.cache_clear = cached_wrapper.cache_clear
    wrapper.cache_info = cached_wrapper.cache_info
    
    return wrapper

# Example usage - rename this to @hashable_args if that's what your code expects
def hashable_args(func):
    return hashable_cache(func)

# Test function to demonstrate usage
@functools.cache
def find_stable_positions_greedy_complex( servers_to_place_tuple,  prioritize_top=False):
    """
    A greedy approach to find stable server positions for various server types inside one rack
    """
    global global_rack_signature
    
    tohash = dict()
    #for key,value in enumerate(servers_to_place):
    #    tohash[str(key)+str(value)] = 1
    #signature = frozenset(tohash)
    servers_to_place = dict(servers_to_place_tuple)
    #signature = tuple(sorted(servers_to_place.items()))
    #if signature in global_rack_signature:
    #    return global_rack_signature[signature]

   
    

    
    all_servers = []
    for server_type, count in servers_to_place.items():
        specs = colors_info.get(server_type)
        if not specs:
            if 'LAN' in server_type:
                lan , sw_model = server_type.split('__')
                
                specs = [x for x in LANs[lan]['switch'] if x['model'] == sw_model][0]
                
            else:
                print(f"Warning: Specifications not found for server type '{server_type}'. Skipping.")
                continue
           
                
        specs['type'] = server_type
        for _ in range(count):
                all_servers.append(specs) # DEBUG_TAG: distribution_loop # append C
    

    if not all_servers:
        return {}, {}, {}  # Return empty dict for empty Counter
    all_servers = sort_device_before_placement(all_servers)
    
    lan_id = 0
    
    periority_lan = 'NA'
    workinglans = [x['type'] for x in all_servers if 'LAN' in x['type']]
    lan_loops = {}
    for lan in workinglans:
        loops =  len(all_servers) - 2
        lan_loops[lan] = {'loops':loops,'best_cable_lengths':float('inf'), 'best_position':0,'referrenced_lan':lan+'_10000000'}

    excluded_swap = []
    best_placement = ()
    best_cable_lengths = float('inf')
    best_position = 0
    if len(workinglans) == 0:
        workinglans = ['na']
    
    for lan in workinglans:
        if lan == 'na':
            loops = 0
        else:
            loops = lan_loops[lan]['loops']
        best_cable_lengths = float('inf')
        periority_lan = lan
        best_position = 0
        swap_pos = 0
        loops += 1
        original_allservers = list(all_servers)
        for _ in range(loops):
            if not prioritize_top:
                placed_servers_info = []
                occupied_u = [False] * rack_height_u
                for server in all_servers:
                    server_height_u = server['height']
                    server_weight_kg = server['weight']
                    server_cg_offset = (server_height_u * u_height_mm) / 2
                    placed = False
                    for i in range(rack_height_u):
                        if not occupied_u[i]:
                            start_u = i
                            server_base_height = start_u * u_height_mm
                            server_cg = server_base_height + server_cg_offset
                            can_place = True
                            for u_check in range(start_u, start_u + server_height_u):
                                if u_check >= rack_height_u or occupied_u[u_check]:
                                    can_place = False
                                    break
                            if can_place:
                                temp_positions = [(p['cg'], p['weight']) for p in placed_servers_info] + [(server_cg, server_weight_kg)]
                                if is_rack_stable(temp_positions):
                                    placed_servers_info.append({'cg': server_cg, 'weight': server_weight_kg, 'type': server['type'], 'height': server['height'], 'start_u': start_u})
                                    for u in range(start_u, start_u + server_height_u):
                                        if u < rack_height_u:
                                            occupied_u[u] = True
                                    placed = True
                                    break
                    if not placed:
                        return None, None, None

                final_placement = {}
                for i, server_info in enumerate(placed_servers_info):
                    final_placement[f"{server_info['type']}_{i+1}"] = server_info['start_u'] + 1
            else:
                placed_servers_info = []
                occupied_u = [False] * rack_height_u
                for server in all_servers:
                    server_height_u = server['height']
                    server_weight_kg = server['weight']
                    server_cg_offset = (server_height_u * u_height_mm) / 2
                    placed = False
                    for i in range(rack_height_u - 1, -1, -1):
                        if not occupied_u[i]:
                            start_u = i
                            server_base_height = start_u * u_height_mm
                            server_cg = server_base_height + server_cg_offset
                            can_place = True
                            for u_check in range(start_u, start_u + server_height_u):
                                if u_check >= rack_height_u or occupied_u[u_check]:
                                    can_place = False
                                    break
                            if can_place:
                                temp_positions = [(p['cg'], p['weight']) for p in placed_servers_info] + [(server_cg, server_weight_kg)]
                                if is_rack_stable(temp_positions):
                                    placed_servers_info.append({'cg': server_cg, 'weight': server_weight_kg, 'type': server['type'], 'height': server['height'], 'start_u': start_u}) # DEBUG_TAG: append B #append B 
                                    for u in range(start_u, start_u + server_height_u):
                                        if u < rack_height_u:
                                            occupied_u[u] = True
                                    placed = True
                                    break
                    if not placed:
                        return {}, {}, {}
                    
               
                final_placement = {}
                sorted_servers = sorted(placed_servers_info, key=lambda x: x['start_u'], reverse=True)
                occupied_map = [False] * rack_height_u
                placed_index = 1
                for server in sorted_servers:
                    start_u = server['start_u']
                    server_type = server['type']
                    server_height = server['height']
                    for u in range(start_u, -1, -1):
                        can_place_here = True
                        for check_u in range(u, u + server_height):
                            if check_u >= rack_height_u or occupied_map[check_u]:
                                can_place_here = False
                                break
                        if can_place_here:
                            final_placement[f"{server_type}_{placed_index}"] = u + 1
                            for occupy_u in range(u, u + server_height):
                                if occupy_u < rack_height_u:
                                    occupied_map[occupy_u] = True
                            placed_index += 1
                            break
                        else:
                            # Fallback to the initially found stable position
                            final_placement[f"{server_type}_{placed_index}"] = start_u + 1
                            for occupy_u in range(start_u, start_u + server_height):
                                if occupy_u < rack_height_u:
                                    occupied_map[occupy_u] = True
                            placed_index += 1
            
            devices, lans, rackswitches = calculate_cable_lengths(final_placement)
            
            if 'LAN' in periority_lan:
                for t_lan in lans:
                    if periority_lan in t_lan:
                        referrenced_lan = t_lan
                        break
                best_flag = 0
                check_preceding_cable_length = 0
                for old_lan in lan_loops:
                    for t_lan in lans:
                        if old_lan in t_lan:
                            old_current_referrenced_lan = t_lan
                            break
                    old_best_cable_lengths = lan_loops[old_lan]['best_cable_lengths']
                    check_preceding_cable_length =  lan_loops[old_lan]['best_cable_lengths'] - lans[old_current_referrenced_lan]['actual_cable_total_length']
                    if check_preceding_cable_length >  0:
                        old_preferred_lan = old_lan
                        best_flag = 1
                    if check_preceding_cable_length < 0:
                        break

                if best_flag and check_preceding_cable_length == 0 : 
                    best_flag = 0
                    for old_lan in lan_loops:
                        for t_lan in lans:
                            if old_lan in t_lan:
                                old_current_referrenced_lan = t_lan
                            break
                        old_referrenced_lan = lan_loops[old_lan]['referrenced_lan']
                        check_current_position = final_placement[old_current_referrenced_lan] - lan_loops[old_lan]['best_position']
                        if check_current_position < 0  :
                            best_flag = 0
                            break
                        if check_current_position > 0:
                            best_flag = 2
                        else:
                            if best_flag == 1:
                                best_flag = 0
                        
                if best_flag == 1:
                    best_placement = (final_placement.copy(), devices.copy(),lans.copy(), rackswitches.copy())
                    
                    for old_lan in lan_loops: 
                        for t_lan in lans:
                            if old_lan in t_lan:
                                old_current_referrenced_lan = t_lan
                            break
                        lan_loops[old_lan]['best_position'] = final_placement[old_current_referrenced_lan]
                        lan_loops[old_lan]['best_cable_lengths'] = lans[old_current_referrenced_lan]['actual_cable_total_length']
                        lan_loops[old_lan]['referrenced_lan'] = lan_loops[old_lan]['referrenced_lan']
                        #print('new placement',best_cable_lengths,'with',best_position,'for',referrenced_lan)
                #print('still',best_cable_lengths, 'best_position',best_position)
                all_servers[swap_pos], all_servers[swap_pos + 1] = all_servers[swap_pos+1], all_servers[swap_pos]
                swap_pos += 1
            else:
                #print('no lans there')
                best_placement = (final_placement.copy(), {},{},{})
                break
    if len(best_placement) < 4:
        print(all_servers)
        print('=============================')
        print(best_placement) 
    #global_rack_signature[signature] = best_placement
    return best_placement
    


# In[ ]:





# # --- Example Usage ---
# rack_height_u = 42
# rack_weight_kg = 114.55
# rack_width_mm = 600
# rack_depth_mm = 1200
# 
# all_racks_config = [
#     Counter({'LAN_1__IB+400':1,'compute_nodes': 5, 'GPU_nodes': 1, 'FrontEnd_nodes': 1}),
#     Counter({'FrontEnd_nodes': 12, 'GPU_nodes': 1}),
#     Counter({'compute_nodes': 5, 'GPU_nodes': 1, 'FrontEnd_nodes': 1}),
#     Counter({'FrontEnd_nodes': 15, 'compute_nodes': 5}),
#     Counter({'compute_nodes': 5, 'GPU_nodes': 1, 'FrontEnd_nodes': 1}),
#     Counter({'storage_nodes': 8}),
#     Counter({'storage_nodes': 15}),
#     Counter({'storage_nodes': 7, 'GPU_nodes': 1}),
#     Counter({'GPU_nodes': 1}),
#     Counter({'GPU_nodes': 1}),
#     Counter({'GPU_nodes': 1}),
#     Counter({'GPU_nodes': 1}),
#     Counter({'GPU_nodes': 1}),
#     Counter({'GPU_nodes': 1}),
#     Counter({'GPU_nodes': 1}),
#     Counter({'GPU_nodes': 1}),
#     Counter({'GPU_nodes': 1}),
#     Counter({'GPU_nodes': 1}),
#     Counter({'GPU_nodes': 1}),
#     Counter({'GPU_nodes': 1}),
#     Counter({'GPU_nodes': 1}),
#     Counter({'GPU_nodes': 1}),
#     Counter({'GPU_nodes': 1}),
#     Counter(),
#     Counter(),
# ]
# 
# 
# colors_info = {
#         'compute_nodes': {'count': 20, 'wattage': 1600, 'height': 2, 'weight':30, 'LAN_1':{'count':1,'speed':400},
#                                                                            'LAN_2':{'count':1,'speed':200},
#                                                                            'LAN_3':{'count':1,'speed':25},
#                                                                            'LAN_4':{'count':1, 'speed':25},
#                                                                            'LAN_5':{'count':1, 'speed':1},
#                                                                             'LAN_6':{'count':0, 'speed':400},
#                       },
#         'GPU_nodes': {'count': 20, 'wattage': 11000, 'height': 8, 'weight': 15, 'LAN_1':{'count':0,'speed':400},
#                                                                             'LAN_2':{'count':1,'speed':200},
#                                                                            'LAN_3':{'count':1, 'speed':25},
#                                                                            'LAN_4':{'count':1, 'speed':25},
#                                                                            'LAN_5':{'count':1, 'speed':1},
#                                                                             'LAN_6':{'count':8, 'speed':400},
#                       },
#         'storage_nodes': {'count': 30, 'wattage': 1250, 'height': 2, 'weight':20, 'LAN_1':{'count':0,'speed':400},
#                                                                             'LAN_2':{'count':3,'speed':200},
#                                                                            'LAN_3':{'count':1, 'speed':25},
#                                                                            'LAN_4':{'count':1, 'speed':25},
#                                                                            'LAN_5':{'count':1, 'speed':1},
#                                                                           'LAN_6':{'count':0, 'speed':400},
#                          },
#         'FrontEnd_nodes': {'count': 30, 'wattage': 750, 'height': 1, 'weight':9, 'LAN_1':{'count':0,'speed':400},
#                                                                             'LAN_2':{'count':0,'speed':200},
#                                                                             'LAN_3':{'count':1, 'speed':25},
#                                                                            'LAN_4':{'count':1, 'speed':25},
#                                                                            'LAN_5':{'count':1, 'speed':1},
#                                                                          'LAN_6':{'count':0, 'speed':400},
#                           },
#                    
#     }
# LANs = { 'LAN_1':{'type':'400gbsNDR','topology': 'halfports_spine_leaf',
#                   'switch':[{'model':'IB+400','ports':64,'speed':400, 'height':2,'wattage':2000, 'weight':20},
#                          {'model':'IB_800','ports':64,'speed':800, 'height':2,'wattage':2000, 'weight':20},
#                            ]},
#               'LAN_2':{'type':'400GbpsEth', 'topology': 'uplinks_spine_leaf',
# 'switch':[{'model':'Z_9xxx','ports':64,'speed':400, 'height':2,'wattage':2200, 'uplink_count':4, 'uplink_speed':800,'weight':20},
#                  {'model':'Z_6xxx','ports':64,'speed':200, 'height':1,'wattage':2000, 'uplink_count':4, 'uplink_speed':400,'weight':20},
#         ]},
#               'LAN_3':{'type':'25gbps','topology': 'uplinks_spine_leaf',
#         'switch':[{'model':'S_xx64','ports':64,'speed':25, 'height':2,'wattage':300, 'uplink_count':4, 'uplink_speed':100,'weight':2},
#                 {'model':'S_xx32','ports':32,'speed':25, 'height':1,'wattage':200,'uplink_count':4, 'uplink_speed':100,'weight':2},
#                  ]},
#               'LAN_4':{'type':'25gbps','topology': 'uplinks_spine_leaf',
#         'switch':[{'model':'S_xx64','ports':64,'speed':25, 'height':2,'wattage':300,'uplink_count':4, 'uplink_speed':100,'weight':2},
#                 {'model':'S_xx32','ports':32,'speed':25, 'height':1,'wattage':200,'uplink_count':2, 'uplink_speed':100,'weight':2},
#                  ]},
#               'LAN_5':{'type':'1gbps','topology': 'uplinks_spine_leaf',
#     'switch':[{'model':'SN_24','ports':24,'speed':1, 'height':1,'wattage':200, 'uplink_count':2, 'uplink_speed':25, 'weight':2},
#              ]},
#               'LAN_6':{'type':'400GbpsEth','topology':'rails',
#     'switch':[{'model':'Z_9xxx','ports':64,'speed':400, 'height':2,'wattage':2200, 'uplink_count':4, 'uplink_speed':800,'weight':20},
#              ]},
#            }
# Cables = {'IB+400':[{'model':'1.5m_400IB_copper','server_port_speed':400,'split':1,'length':1.5},
#                                            {'model':'3m_400IB_copper','server_port_speed':400,'split':1,'length':3},
#                                          {'model':'5m_400IB_fiber','server_port_speed':400,'split':1,'length':5},
#                                       { 'model':'7m_400IB_fiber','server_port_speed':400,'split':1,'length':7},
#                                       { 'model':'10m_400IB_fiber','server_port_speed':400,'split':1,'length':10},
#                                       { 'model':'20m_400IB_fiber','server_port_speed':400,'split':1,'length':20},
#                                       { 'model':'3m_400sIB_copper','server_port_speed':200,'split':2,'length':3},
#                                      {  'model':'5m_400sIB_fiber','server_port_speed':200,'split':2,'length':5},
#                                       { 'model':'7m_400sIB_fiber','server_port_speed':200,'split':2,'length':7},
#                                       { 'model':'10m_400sIB_fiber','server_port_speed':200,'split':2,'length':10},
#                                      {  'model':'20m_400sIB_fiber','server_port_speed':200,'split':2,'length':20},
#                             ],
#                             
#           'IB_800':[{'model':'1.5m_800IB_copper','server_port_speed':800,'split':1,'length':1.5},
#                                        {'model':'3m_800IB_copper','server_port_speed':800,'split':1,'length':3},
#                                       { 'model':'8m_800IB_fiber','server_port_speed':800,'split':1,'length':5},
#                                       { 'model':'7m_800IB_fiber','server_port_speed':800,'split':1,'length':7},
#                                       { 'model':'10m_800IB_fiber','server_port_speed':800,'split':1,'length':10},
#                                        {'model':'20m_800IB_fiber','server_port_speed':800,'split':1,'length':20},
#                                       { 'model':'3m_800sIB_copper','server_port_speed':400,'split':2,'length':3},
#                                      {  'model':'5m_800sIB_fiber','server_port_speed':400,'split':2,'length':5},
#                                       { 'model':'7m_800sIB_fiber','server_port_speed':400,'split':2,'length':7},
#                                       { 'model':'10m_800sIB_fiber','server_port_speed':400,'split':2,'length':10},
#                                       { 'model':'20m_800sIB_fiber','server_port_speed':400,'split':2,'length':20},
#                    ],
#           'S_xx64':[{'model':'1.5m_25_copper','server_port_speed':25,'split':1,'length':1.5},
#                                        {'model':'3m_25_copper','server_port_speed':25,'split':1,'length':3},
#                                        {'model':'5m_25_fiber','server_port_speed':25,'split':1,'length':5},
#                                      {  'model':'7m_25_fiber','server_port_speed':25,'split':1,'length':7},
#                                        {'model':'10m_25_fiber','server_port_speed':25,'split':1,'length':10},
#                                        {'model':'20m_25_fiber','server_port_speed':25,'split':1,'length':20},                                       
#                    ],
#           'S_xx32':[{'model':'1.5m_25_copper','server_port_speed':25,'split':1,'length':1.5},
#                                        {'model':'3m_25_copper','server_port_speed':25,'split':1,'length':3},
#                                        {'model':'5m_25_fiber','server_port_speed':25,'split':1,'length':5},
#                                        {'model':'7m_25_fiber','server_port_speed':25,'split':1,'length':7},
#                                        {'model':'10m_25_fiber','server_port_speed':25,'split':1,'length':10},
#                                        {'model':'20m_25_fiber','server_port_speed':25,'split':1,'length':20},                                       
#                    ],
#           'SN_24':[{'model':'1.5m_1_copper','server_port_speed':1,'split':1,'length':1.5},
#                                       { 'model':'3m_1_coppe','server_port_speed':1,'split':1,'length':3},
#                                        {'model':'5m_1_copper','server_port_speed':1,'split':1,'length':5},
#                                       { 'model':'7m_1_copper','server_port_speed':1,'split':1,'length':7},
#                                       { 'model':'10m_1_copper','server_port_speed':1,'split':1,'length':10},
#                                       { 'model':'20m_1_copper','server_port_speed':1,'split':1,'length':20}, 
#                                       { 'model':'30m_1_copper','server_port_speed':1,'split':1,'length':30},
#                                        {'model':'40m_1_copper','server_port_speed':1,'split':1,'length':40},
#                   ],
#           'Z_9xxx':[{'model':'1.5m_400_copper_eth','server_port_speed':400,'split':1,'length':1.5},
#                                        {'model':'3m_400_copper_eth','server_port_speed':400,'split':1,'length':3},
#                                       { 'model':'5m_400_fiber_eth','server_port_speed':400,'split':1,'length':5},
#                                       { 'model':'7m_400_fiber_eth','server_port_speed':400,'split':1,'length':7},
#                                      {  'model':'10m_400_fiber_eth','server_port_speed':400,'split':1,'length':10},
#                                       { 'model':'20m_400_fiber_eth','server_port_speed':400,'split':1,'length':20},
#                                       { 'model':'3m_400s_copper_eth','server_port_speed':200,'split':2,'length':3},
#                                      {  'model':'5m_400s_fiber_eth','server_port_speed':200,'split':2,'length':5},
#                                      {  'model':'7m_400s_fiber_eth','server_port_speed':200,'split':2,'length':7},
#                                       { 'model':'10m_400s_fiber_eth','server_port_speed':200,'split':2,'length':10},
#                                       { 'model':'20m_400s_fiber_eth','server_port_speed':200,'split':2,'length':20},
#                    ],
#          }         
#             
#           
# 
# results = []
# 
# for rack_config in all_racks_config:
#     if rack_config:
#         #print(f"Processing rack with config: {rack_config}")
#         # Bottom-up placement
#         stable_placement_bottom, devices, lans = find_stable_positions_greedy_complex(
#             rack_height_u, rack_weight_kg, rack_width_mm, rack_depth_mm,
#             rack_config, LANs, Cables, colors_info, prioritize_top=True
#         )
#         print(f" Top-biased Placement: {stable_placement_bottom}")
# 
#         # Top-biased placement
#         stable_placement_top_biased, devices, lans = find_stable_positions_greedy_complex(
#             rack_height_u, rack_weight_kg, rack_width_mm, rack_depth_mm,
#             rack_config, LANs, Cables, colors_info, prioritize_top=False
#         )
#         print(f" Bottom-up Placement: {stable_placement_top_biased}")
#         print("-" * 30)
# 
# print("Processing complete.")

# In[ ]:





# In[12]:


def get_rack_layouts(distributions):
    
    distributions_info = dict()
    for i, rack_config in enumerate(distributions):
        if rack_config:
            switches = dict()
            # Top-biased placement
            rack_config_tuple = tuple(sorted(rack_config.items()))
            stable_placement , devices, lans, rackswitches = find_stable_positions_greedy_complex(
                rack_config_tuple,  prioritize_top=True
            )
            distributions_info[i] = dict({'rack_config':rack_config, 'stable_placement': stable_placement ,'lan_info':lans,
                                               'device_info':devices, 'switches':rackswitches})
           
            #print(f" Top-biased Placement: {stable_placement_bottom}")

            
             # Bottom-up placement
            #stable_placement_top_biased = find_stable_positions_greedy_complex(
            #    rack_config, prioritize_top=false
            #)
            #print(f"   Bottom-up Placement:{stable_placement_top_biased}")
            #print("-" * 30)
    
    #print("Processing rack layouts Complete.")
    return  distributions_info


# In[13]:


x= [1,2,4]

x[::-1]


# In[ ]:





# In[14]:


def otpimize_rack_layout(current_placement, device_info):

    """
    Place devices in the rack considering cable constraints and existing placements.
    
    Args:
        current_placement: Dictionary of device_id to position in rack
        device_info: Dictionary containing device connection information
        Cables: Global dictionary of cable specifications
        colors_info: Global dictionary of device specifications including height
        
    Returns:
        Tuple of (updated_current_placement, updated_device_info) with new positions
    """
    # Create copies of the input dictionaries to avoid modifying originals
    updated_placement = current_placement.copy()
    updated_device_info = {k: v.copy() for k, v in device_info.items()}
    
    # Create a set of occupied positions for quick lookup
    occupied_positions = set()
    for device, pos in updated_placement.items():
        # Get device type (before the first underscore or number)
        device_type = '_'.join(device.rsplit('_', 1)[:-1])
        if device_type in colors_info:
            height = colors_info[device_type]['height']
           
        else: 
            height =  LANs[device_type.split('__')[0]]['switch'][0]['height']
         # Mark all positions from pos to pos+height-1 as occupied
        occupied_positions.update(range(pos, pos + height))
    
    # Sort devices by their current position (lowest first) to prioritize moving them lower
    sorted_devices = sorted(
        [dev for dev in updated_device_info.items() if not dev[0].startswith('LAN_')],
        key=lambda x: x[1]['position']
    )
    
    for device_id, device_data in sorted_devices:
        current_pos = device_data['position']
        device_type = '_'.join(device.rsplit('_', 1)[:-1])
       
        if device_type not in colors_info:
            continue  # Skip unknown device types
            
        height = colors_info[device_type]['height']
        
        # Find all connected LAN devices and their constraints
        lan_constraints = []
        for lan_device, connection in device_data.items():
            if isinstance(connection, dict) and 'cable_type' in connection and lan_device in updated_placement:
                cable_model = connection['cable_type']['model']
                in_rack_stretch = connection['cable_type']['in_rack_stretch']
                lan_pos = updated_placement[lan_device]
                lan_device_type = lan_device.split('_')[0]
                lan_height = colors_info.get(lan_device_type, {}).get('height', 1)
                
                lan_constraints.append({
                    'lan_pos': lan_pos,
                    'lan_height': lan_height,
                    'max_distance': in_rack_stretch,
                    'cable_model': cable_model,
                    'lan_device': lan_device
                })
        
        # Try to find the best position for this device (try to go as low as possible)
        best_pos = None
        
        # Determine the lowest possible position we can try (considering height)
        min_possible_pos = 1
        max_possible_pos = current_pos  # We're only trying to move downward
        
        # Search from lowest possible position up to current position
        for pos in range(min_possible_pos, max_possible_pos + 1):
            # Check if this position and height would fit
            
            if any(p in occupied_positions for p in range(pos, pos + height)):
                continue
            
            # Check all LAN constraints
            valid = True
            for constraint in lan_constraints:
                distance = abs(pos - constraint['lan_pos'])
                if distance > constraint['max_distance']:
                    valid = False
                    break
            
            if valid:
                best_pos = pos
                break
                
        
        # If we found a valid position (lower than current), update everything
        if best_pos is not None and best_pos < current_pos:
            # Remove old position from occupied positions
            for i in range(height):
                if current_pos + i in occupied_positions:
                    occupied_positions.remove(current_pos + i)
            
            # Update the position in both dictionaries
            updated_placement[device_id] = best_pos
            updated_device_info[device_id]['position'] = best_pos
            
            # Add new position to occupied positions
            for i in range(height):
                occupied_positions.add(best_pos + i)
    
    return updated_placement, updated_device_info
    
    


# In[15]:


def arrange_racks_best_(distributions_info):
    rack_updates = {}
    for i in distributions_info:
        if 'rack_config' in distributions_info[i]:
            rack_updates[i] = {}
            #xx = {'compute_nodes_1': 29, 'compute_nodes_2': 27, 'compute_nodes_3': 25, 'LAN_2__Z_9xxx_4': 21, 'LAN_3__S_xx64_5': 19, 'compute_nodes_6': 17, 'compute_nodes_7': 15, 'compute_nodes_8': 13}    
            rack_updates[i]['palcement'], rack_updates[i]['laninfo'] = otpimize_rack_layout( distributions_info[i]['stable_placement'],distributions_info[i]['device_info'])
            #otpimize_rack_layout( xx,distributions_info[i]['device_info'])
    #hhhhhhh
    for i in rack_updates:
       distributions_info[i]['bottom_place_periority'], distributions_info[i]['bottom_laninfo_periority'] = rack_updates[i]['palcement'], rack_updates[i]['laninfo'] 
    return distributions_info


# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


from collections import Counter
import time
import dask
from dask.distributed import Client, LocalCluster


# Initialize Dask Client for local parallel computing
# n_workers sets the number of worker processes (defaults to number of cores)
# threads_per_worker sets threads per worker process
cluster = LocalCluster(n_workers=4, threads_per_worker=2)
client = Client(cluster)
print(f"Dask dashboard available at: {client.dashboard_link}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


# Function to process a batch of iterations
def process_batch( switch_scenarios, num_boxes_to_try):
    global batch_size
    results = []
    for _ in range(batch_size):
        batch_results = {}
        for sw_scenario in switch_scenarios:
            scenario_results = {}
            for num_boxes in num_boxes_to_try:
                # This will be executed in parallel
                scenario_results[num_boxes] = dask.delayed(process_combination)(sw_scenario, num_boxes)
            batch_results[sw_scenario] = scenario_results
        results.append(batch_results)
    
    # Compute all delayed objects at once
    computed_results = dask.compute(*[result for batch in results for sw in batch.values() for result in sw.values()])
    return computed_results


# In[ ]:





# In[21]:


import numpy as np
import os
import time
import multiprocessing as mp
from collections import Counter
from itertools import product, permutations
from math import ceil, floor
import logging
import random
import functools, itertools
from collections import deque
from copy import deepcopy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_box_wattage(box,color_metadata):
    total = 0
    color_metadata = dict(color_metadata_tuple)
    box = dict(box_tuple)
    for color, count in box.items():
        meta = color_metadata[color]
        total += count * meta[1]
    return total


def calculate_box_height(box, color_metadata):
    total = 0
    color_metadata = dict(color_metadata_tuple)
    box = dict(box_tuple)
    for color, count in box.items():
        meta = color_metadata[color]
        total += count * meta[2]
    return total

@functools.cache
def calculate_box_wattage_height(box_tuple, color_metadata_tuple):
    total_wattage= 0
    total_height = 0
    color_metadata = dict(color_metadata_tuple)
    box = dict(box_tuple)
    for color, count in box.items():
        meta = color_metadata[color]
        total_wattage += count * meta[1]
        total_height += count * meta[2]
    return total_wattage, total_height
    
@functools.cache 
def check_box_limits(box_tuple, color_metadata_tuple):
    color_metadata = dict(color_metadata_tuple)
    box = dict(box_tuple)
    wattage = height = 0
    for color, count in box.items():
        meta = color_metadata[color]
        wattage += count * meta[1]
        height += count * meta[2]
        # Early exit if limits exceeded
        if wattage > max_box_wattage or height > max_box_height:
            return False, wattage, height
    return True, wattage, height

def get_total_cable_lengths(distributions_info):
    global_minimum_length = 0
    global_actual_length = 0
    global_switches = dict()
    cables = dict()
    
    for rack_info in distributions_info:
        if 'lan_info' in distributions_info[rack_info]:
            for lan in distributions_info[rack_info]['lan_info']:
                global_minimum_length += distributions_info[rack_info]['lan_info'][lan]['minimum_cable_total_length']
                global_actual_length += distributions_info[rack_info]['lan_info'][lan]['actual_cable_total_length']
                for cable in distributions_info[rack_info]['lan_info'][lan]['cables']:
                    if cable not in cables:
                        cables[cable] = 0
                    cables[cable] += distributions_info[rack_info]['lan_info'][lan]['cables'][cable]
        if 'switches' in distributions_info[rack_info]:
            for sw_lan in distributions_info[rack_info]['switches']:
                if sw_lan not in global_switches:
                    global_switches[sw_lan] = 0
                global_switches[sw_lan] += distributions_info[rack_info]['switches'][sw_lan]
    
    distributions_info['cables'] = {'global_minimum_length':global_minimum_length, 'global_actual_length':global_actual_length,
                                   'cables':cables }
    distributions_info['switches'] = global_switches.copy()

    return distributions_info

def display_distribution_filled_only(distributions_info, color_metadata_tuple, total_balls, balls_placed_overall):
    print(f"\nValid Distribution Found (Iterative Greedy - Filled Racks Only):")
    current_total_balls = sum(balls_placed_overall.values())
    total_wattage = 0
    total_height = 0
    total_count = 0
    filled_boxes = []
    # Create a dictionary to store unique rack configurations
    unique_racks = {}
    rack_summary = []

    # First pass: Identify and group identical racks
    for i, info in enumerate(distributions_info):
        if 'rack_config' in distributions_info[info]:
            # Create a hashable representation of the rack configuration
            rack_config = distributions_info[info]['rack_config']
            # Convert the rack_config dictionary to a tuple of sorted items for hashability
            rack_tuple = tuple(sorted(rack_config.items()))

            # Store the rack details with its ID
            rack_details = {
                'rack_id': i+1,
                'config': rack_config,
                'stable_placement': distributions_info[info].get('stable_placement', []),
                'bottom_place_periority': distributions_info[info].get('bottom_place_periority', []),
                'lan_info': distributions_info[info].get('lan_info', {}),
                'device_info': distributions_info[info].get('device_info', {})
            }

            # If we've seen this configuration before, append to the list
            if rack_tuple in unique_racks:
                unique_racks[rack_tuple]['rack_ids'].append(i+1) # append D
                unique_racks[rack_tuple]['count'] += 1
            else:
                # First time seeing this configuration
                unique_racks[rack_tuple] = {
                    'rack_ids': [i+1],
                    'count': 1,
                    'details': rack_details
                }

        # Track the total count regardless of whether we're processing racks
        if 'stable_placement' in distributions_info[info]:
            total_count += len(distributions_info[info]['stable_placement'])

    # Second pass: Generate summary output
    filled_boxes = []
    rack_counts = 0
    for rack_tuple, data in unique_racks.items():
        rack_config = data['details']['config']
        rack_config_tuple = tuple(sorted(rack_config.items()))
        box_contents = ", ".join(f"{count} {color}" for color, count in rack_config.items())
        box_wattage, box_height = calculate_box_wattage_height(rack_config_tuple, color_metadata_tuple)
        total_wattage += box_wattage * data['count']  # Multiply by number of identical racks
        total_height += box_height * data['count']  # Multiply by number of identical racks
        ball_count = sum(rack_config.values())

        # Format rack IDs nicely
        if len(data['rack_ids']) > 1:
            rack_id_str = f"Racks {', '.join(map(str, data['rack_ids']))}"
        else:
            rack_id_str = f"Rack {data['rack_ids'][0]}"

        # Add number of identical racks if more than one
        count_str = f" ({data['count']} identical racks)" if data['count'] > 1 else ""
        rack_counts += data['count']

        filled_boxes.append(f"\x1b[1m -------------------------{rack_id_str} info: {count_str}------------------------------\x1b[0m")
        filled_boxes.append(f"\x1b[1;4m-->{rack_id_str}:\x1b[0m \n {box_contents} (Wattage: {box_wattage}, Height: {box_height}, count:{ball_count})")
        filled_boxes.append(f"-->\x1b[1;4mdevice placement: in the rack \x1b[0m \n {data['details']['stable_placement']}")
        filled_boxes.append(f"-->\x1b[1;4mbottom periority placement: in the rack \x1b[0m \n {data['details']['bottom_place_periority']}")
        filled_boxes.append(f"-->\x1b[1;4maggregated LAN info: \x1b[0m \n {data['details']['lan_info']}")
        #filled_boxes.append(f"-->\x1b[1;4mDevice detailed info: \x1b[0m \n {data['details']['device_info']}")   

 

    filled_boxes.append(f"\x1b[1;4m--------------------Total distribution info ({rack_counts}) racks, in {len(unique_racks)} groups------------------------------ \x1b[0m \n {distributions_info['cables']} \n {distributions_info['switches']}")
    if filled_boxes:
        for box_info in filled_boxes:
            print('rack_info', box_info)
        print(f"  Total Wattage: {total_wattage}, Total Height: {total_height}, Total Balls Distributed:{total_count} {current_total_balls}")
    else:
        print(f"  No racks were filled. Total Balls Distributed: {current_total_balls}")
    if current_total_balls != total_balls:
        print(f"  WARNING: Total balls distributed ({current_total_balls}) does not match the expected total ({total_balls})!")
    
@functools.cache
def is_switch_relevent(rack):
    alllans = set()
    allnodes = ""
    include_lan = set()
    execlude_lan = set()
    for device in rack:
        if 'LAN' in device:
            switch = device.split('__')[-1]
            lan = device.split('__')[0]
            alllans.add(lan)
    is_there_device = 0
    for device in rack:
        if 'LAN' not in device:
            for lan in alllans:
                if colors_info[device][lan]['count'] > 0:
                    include_lan.add(lan)
    execlude_lan = alllans - include_lan
    return list(execlude_lan)

def get_distribution_signature(distribution):
        #start_time = time.time()
        filled_boxes_signature = tuple(sorted(tuple(sorted(box.items())) for box in distribution if box))
        #end_time = time.time()
        #print(f"Update: the signing the distribution is calcualted in {end_time - start_time:.6f} seconds")
        return filled_boxes_signature

def format_time_difference_compact(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    parts = []
    if days > 0:
        parts.append(f"{int(days)}d")
    if hours > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0:
        parts.append(f"{int(minutes)}m")
    parts.append(f"{seconds:.3f}s")  # Show seconds always, with 3 decimal places for compactness

    return " ".join(parts) or "0.000s" 
    
def find_valid_distributions_iterative_greedy_adaptive( initial_num_boxes,  total_balls):
    global batch_size
    import cProfile

    profiler = cProfile.Profile()
    profiler.enable()
    colors = list(colors_info.keys())
    #create a dictionary same like the color_info which include the LANs various switches
    switch_info = dict()
    current_lans = []
    str_nodes= str(colors_info)
    current_lans = [lan for lan in LANs if lan in str_nodes]
    for lan in current_lans:
        switch_info[lan] = LANs[lan]['switch'].copy()
    colors_wattage = {color: colors_info[color]['wattage'] for color in colors}
    colors_wattage.update({lan+'__'+switch_info[lan][0]['model']: switch_info[lan][0]['wattage'] for lan in list(switch_info.keys())})
    
    colors_height = {color: colors_info[color]['height'] for color in colors}
    colors_height.update({lan+'__'+switch_info[lan][0]['model']: switch_info[lan][0]['height'] for lan in list(switch_info.keys())})
    
    colors_weight = {color: colors_info[color]['weight'] for color in colors}
    colors_weight.update({lan+'__'+switch_info[lan][0]['model']: switch_info[lan][0]['weight'] for lan in list(switch_info.keys())})

    color_metadata = {
    color: (
        color.split('__')[-1],  # Gets the part after last '__',
        colors_wattage[color],
        colors_height[color],
        colors_weight[color],
        )
        for color in set(colors_wattage.keys()).union(colors_height.keys())
    }
    color_metadata_tuple = tuple(sorted(color_metadata.items()))
    
    max_possible_boxes = total_balls
    valid_distributions = deque()
    seen_distributions = set()

    

    num_boxes_to_try = list(range(1, initial_num_boxes + max_possible_boxes + 1))
    #random.shuffle(num_boxes_to_try)
    lan_len = len(LANs)
    switch_scenarios = []
    switch_scenarios.append("switch_per_1_rack")
    #switch_scenarios.append("switch_per_2_Racks")
    #switch_scenarios.append("switch_per_3_Racks")
    #switch_scenarios.append("switch_per_4_Racks")
    #switch_scenarios.append("switch_per_5_Racks")
    #switch_scenarios.append("switch_per_6_Racks")
    swtich_index = 0
    start_time = time.time()
    current_switches = switch_info.copy()
    swcounter = 0 
    iter_counter = 1
    switch_to_add = dict()
    for lan in switch_info:
        switch_to_add[lan] = lan+ '__' + switch_info[lan][0]['model']
    best_minimum_length = float('inf')
    best_actual_length = float('inf')
    best_distributions = []
    best_min_devices = float('inf')
    possible_colors = [c for c in colors]
    rng = random.Random(43)
    best_LAN_check = []
    for _ in LANs:
        best_LAN_check.append(float('inf'))
    sc_counter = -1
    
    while iter_counter:
        batch_results = process_batch(switch_scenarios, num_boxes_to_try)
        iter_counter += 1
        sc_counter += batch_size
        
        
            # Potentially break here if you only need one solution
        
        if iter_counter > max_iter:
            profiler.disable()
            profiler.print_stats(sort='cumtime')  # Sort by cumulative time
            return seen_distributions, valid_distributions, iter_counter
        if iter_counter/500 == iter_counter //500:
            #print(f"passing the {iter_counter} of {max_iter}")
            end_time = time.time()
            print(f"Update: the seen/validated distribution No {len(seen_distributions)} is calcualted in {end_time - start_time:.6f} sec after passing {iter_counter} iterations and rack cash {len(global_rack_signature)}",end='\r', flush=True)
            start_time = time.time()
        iter_counter += 1
            #end_time = time.time()
            #print(f"Update: the seen/validated distribution No{len(seen_distributions)} is calcualted in {end_time - start_time:.6f} seconds")
            #start_time = time.time()
    profiler.disable()
    profiler.print_stats(sort='cumtime')  # Sort by cumulative time
    return seen_distributions, valid_distributions, iter_counter


# In[ ]:





# In[22]:


def process_combination(sw_scenario, num_boxes):  

    distributions = [Counter() for _ in range(num_boxes)]

    #if sw_scenario.split('_')[2] == '1':
    #    for box in distributions:
    #        for sw in switch_to_add:
    #            box[sw] = 1


    #else:
    #    break
    sw_scenario_divisor = int(sw_scenario.split('_')[2])
    stub_rack = distributions[0]
    balls_placed = {color: 0 for color in colors}
    box_index = 0
    while box_index < num_boxes and any(balls_placed[color] < colors_info[color]['count'] for color in colors):
        current_box = distributions[box_index]
        remaining_balls = {c: colors_info[c]['count'] - balls_placed[c] for c in colors if colors_info[c]['count'] - balls_placed[c]> 0}
        possible_colors = list(remaining_balls.keys())
        possible_len = len(possible_colors)
        if not possible_colors:
            box_index += 1
            sc_counter -= 1
            end_time = time.time()
            print(f"Update: failed combination {len(seen_distributions)} is calcualted in {end_time - start_time:.6f} sec after passing {iter_counter} iterations and rack cash {len(global_rack_signature)}",end='\r', flush=True)
            start_time = time.time()
            continue
        if sc_counter == 100:
            sc_counter = 0
            for i in range(possible_len-1, 0, -1):
                j = rng.randint(0, i)
                possible_colors[i], possible_colors[j] = possible_colors[j], possible_colors[i]

        scenarios = deque()

        for iddc in range(100-sc_counter,0,-1):

            #scenarios.append(("single_color", color))
            for color in possible_colors:
                scenarios.append(("partial_color_"+str(iddc), color)) # append E
            #scenarios.append(("mixed_fill_initial", color))
        if sc_counter > 10:
            sc_len = len(scenarios)  - 1
            for i in range(sc_len-1, 0, -1):
                j = rng.randint(0, i)
                scenarios[i], scenarios[j] = scenarios[j], scenarios[i]


        #random.shuffle(scenarios) # Try scenarios in a random order for each box


        applied_scenario = False
        for scenario_type, main_color in scenarios:
        #     print(scenario_type,main_color)
        #    continue
        #ssssls
        #while True:
            temp_box = current_box.copy()
            temp_box_tuple = tuple(sorted(temp_box.items()))
            temp_balls_placed = balls_placed.copy()
            temp_remaining_balls = remaining_balls.copy()
            main_color_info = colors_info[main_color]  # Cache dict lookup
            lan_prefixes_to_check = [lan for lan in main_color_info if 'LAN' in lan]



            #if scenario_type.startswith("partial_color"):
            if "partial_color" in scenario_type:

                percentage = float(int(scenario_type.split('_')[2])/100)
                add_amount = min(ceil(main_color_info['count'] * percentage), temp_remaining_balls[main_color])

                can_add = True
                cycles = add_amount
                recent_sw = []
                temp_box_keys = set(temp_box.keys())
                temp_box_tuple = tuple(sorted(temp_box.items()))
                while cycles:
                    if not recent_sw:
                        for lan in lan_prefixes_to_check:
                            # Check count first (cheaper than string operations)
                            if int(main_color_info[lan]['count']) <= 0:
                                continue

                            # Fast substring check using precomputed keys
                            lan_missing = True
                            for key in temp_box_keys:
                                if lan in key:
                                    lan_missing = False
                                    break

                            if lan_missing:
                                # Optimized switch addition logic
                                if not box_index % sw_scenario_divisor:
                                    switch_key = switch_to_add.get(lan)  # Direct dict access
                                    if switch_key:
                                        temp_box[switch_key] = 1
                                        temp_box_tuple = tuple(sorted(temp_box.items()))
                                        recent_sw.append(switch_key) # append F
                    box_limits, box_wattage, box_height = check_box_limits(temp_box_tuple, color_metadata_tuple)
                    if box_limits and \
                       box_wattage + colors_wattage[main_color] <= max_box_wattage and \
                       box_height + colors_height[main_color] <= max_box_height and \
                       temp_remaining_balls.get(main_color,0) > 0:
                            temp_box[main_color] += 1
                            temp_box_tuple = tuple(sorted(temp_box.items()))
                            temp_balls_placed[main_color] += 1
                            temp_remaining_balls[main_color] -= 1
                            cycles -= 1
                    else:

                        cycles = 0


                iteratecolor = 1
                removed_sw = []
                cycled = 1
                twice_state = 2
                ball_added = 1

                while twice_state:
                    twice_state -=1
                    others = (x for x in temp_remaining_balls.keys() if x != main_color)
                    if temp_remaining_balls.get(main_color, 0) > 0: 
                        others = itertools.chain(others, [main_color])

                    recent_sw = []
                    for other_color in others:
                        temp_box_keys = set(temp_box.keys())
                        other_color_info = colors_info[other_color]  # Cache dict lookup
                        other_lan_prefixes_to_check = [lan for lan in other_color_info if 'LAN' in lan]
                        if not recent_sw:
                            for lan in other_lan_prefixes_to_check:
                                # Check count first (cheaper than string operations)
                                if int(other_color_info[lan]['count']) <= 0:
                                    continue

                                # Fast substring check using precomputed keys
                                lan_missing = True
                                for key in temp_box_keys:
                                    if lan in key:
                                        lan_missing = False
                                        break

                                if lan_missing:
                                    # Optimized switch addition logic
                                    if not box_index % sw_scenario_divisor:
                                        switch_key = switch_to_add.get(lan)  # Direct dict access
                                        if switch_key:
                                            temp_box[switch_key] = 1
                                            temp_box_tuple = tuple(sorted(temp_box.items()))
                                            recent_sw.append(switch_key) # append G


                        #if temp_remaining_balls.get(other_color, 0) > 0:
                        #print('checking again',temp_box)
                        #print('temp_remaining_balls[other_color] > 0',temp_remaining_balls[other_color] > 0)
                        #print('box_limits',check_box_limits(temp_box, colors_wattage, colors_height))
                        #print('wattage',calculate_box_wattage(temp_box, color_metadata) , colors_wattage.get(other_color, 0), max_box_wattage)
                        #print('height',calculate_box_height(temp_box, colors_height) + colors_height.get(other_color, 0) <= max_box_height)
                        box_limits, box_wattage, box_height = check_box_limits(temp_box_tuple, color_metadata_tuple)
                        if temp_remaining_balls[other_color] > 0 and box_limits and \
                                box_wattage + colors_wattage.get(other_color, 0) <= max_box_wattage and \
                                box_height + colors_height.get(other_color, 0) <= max_box_height:
                            temp_box[other_color] = temp_box.get(other_color, 0) + 1
                            temp_box_tuple = tuple(sorted(temp_box.items()))
                            temp_balls_placed[other_color] = temp_balls_placed.get(other_color, 0) + 1
                            temp_remaining_balls[other_color] -= 1
                            iteratecolor = 1
                            ball_added = 1
                            #print(temp_box)

                        else:
                            while recent_sw:
                                can_pop = 1
                                sw_lan = recent_sw.pop()
                                for device in temp_box:
                                    if 'LAN' not in device:
                                        if int(colors_info[device][sw_lan.split('__')[0]]['count']) > 0:
                                            can_pop = 0
                                            break
                                if can_pop:
                                    temp_box.pop(sw_lan)
                                    temp_box_tuple = tuple(sorted(temp_box.items()))
                                    #twice_state = 3
                                    #print('popped')
                                    break

                            if twice_state >= 1 and ball_added:
                                twice_state = 3
                                ball_added = 0
                                #print('ball was added so ts = 3')

                            elif twice_state == 2:
                                twice_state = 0

                if (can_add or add_amount > 0) and temp_box != current_box:
                    distributions[box_index] = temp_box.copy()
                    balls_placed.update(temp_balls_placed)
                    applied_scenario = True
                    break


        advance_box = 0        
        if applied_scenario or not possible_colors:
            advance_box = 1
        elif not any(remaining_balls.values()):
            advance_box = 1
        if advance_box:
            execlude_switch = is_switch_relevent(temp_box_tuple)
            #print(f" before {temp_box}")
            if execlude_switch:
                advanece_box = 0
                for lan in execlude_switch:
                    temp_box = Counter({k: v for k, v in temp_box.items() if lan not in k})
                    temp_box_tuple = tuple(sorted(temp_box.items()))
                    #print(f" after {temp_box}")
                    distributions[box_index] = temp_box

            box_index += 1

    if sum(balls_placed.values()) == total_balls:
        remove_indices = []
        for i,rack in enumerate(distributions):
            remove_rack = 1
            for item in rack:
                if 'LAN' not in str(item):
                    remove_rack = 0
                    break
            if remove_rack:
                remove_indices.append(i) # DEBUG_TAG: append A # append A
        for i in reversed(remove_indices):  
            del distributions[i]

        signature = get_distribution_signature(distributions)
        if signature not in seen_distributions:

            seen_distributions.add(signature)
            distributions_info = get_rack_layouts(distributions)
            distributions_info = get_total_cable_lengths(distributions_info)
            total_devices = 0
            for rack in distributions:
                    total_devices  += sum(rack.values())
            LAN_check = [float('inf')] * len(LANs)
            for id in range(len(LANs)):
                for key in distributions_info['switches']:
                    if 'LAN_'+str(id) in key:
                        LAN_check[id] = distributions_info['switches'][key]
                        break
            actual_length_check = distributions_info['cables']['global_actual_length']
            total_length_check = sum(LAN_check)
            if (actual_length_check < best_actual_length) or \
            (actual_length_check == best_actual_length and LAN_check[0] < best_LAN_check[0]) or \
             (actual_length_check == best_actual_length and LAN_check[0] == best_LAN_check[0] and total_devices < best_min_devices) :
                best_min_devices = total_devices
                best_LAN_check = LAN_check.copy()
                best_distributions = distributions_info
                best_actual_length = distributions_info['cables']['global_actual_length']
                distributions_info = arrange_racks_best_(distributions_info)
                display_distribution_filled_only(distributions_info, color_metadata_tuple, total_balls, balls_placed)
                valid_distributions.append(deque(distributions)) # DEBUG_TAG: valid_distributions #append J
                return distributions


# In[ ]:





# In[23]:


def main_iterative_greedy_adaptive_with_height():
        
    
    num_devices = sum(info['count'] for info in colors_info.values())
    max_needed_wattage = sum(colors_info[color]['wattage'] * colors_info[color]['count'] for color in colors_info)
    # arrange the list of lANs that are found int eh color_info
    current_lans = []
    str_nodes= str(colors_info)
    for lan in LANs:
        if lan in str_nodes:
            current_lans.append(lan)
   
 
            
    #share this dictionary with the find_valid_distributions function
    initial_num_boxes = ceil(max_needed_wattage / max_box_wattage) + 2

    print(f"Distributing {num_devices} balls into initially {initial_num_boxes} boxes which consumes {max_needed_wattage} watts (Iterative Greedy Adaptive with Height):")

    start_time = time.time()
    

    seen_distributions, valid_distributions, iterations = find_valid_distributions_iterative_greedy_adaptive(
        initial_num_boxes, num_devices
    )

   
    
    
    end_time = time.time()
    elapsedtime = format_time_difference_compact(end_time-start_time)

    print(f"\nSummary: Found {len(seen_distributions)} unique distribution(s) with {len(valid_distributions)} times of optimum length after {iterations} iterations tries in {elapsedtime} seconds")


def global_main():
    global colors_info, LANs, Cables, max_box_wattage, max_box_height,  rack_height_u , rack_weight_kg, rack_width_mm, \
            rack_depth_mm, u_height_mm, rack_height_mm, rack_height_u, rack_cg_height_mm, unit_to_cm, u_height_mm, \
            device_to_rackside, racktop_to_ceiling, rack_to_rack, batch_size, max_iter 
    
    colors_info = {
        'compute_nodes': {'count':91, 'wattage': 2387, 'height': 2, 'weight':28.7, 'LAN_1':{'count':0,'speed':400},
                                                                           'LAN_2':{'count':1,'speed':200},
                                                                           'LAN_3':{'count':1,'speed':25},
                                                                           'LAN_4':{'count':0, 'speed':25},
                                                                           'LAN_5':{'count':0, 'speed':1},
                                                                            'LAN_6':{'count':0, 'speed':400},
                      },
        'GPU_nodes': {'count':33, 'wattage': 5894, 'height': 4, 'weight': 61.4, 'LAN_1':{'count':0,'speed':400},
                                                                            'LAN_2':{'count':2,'speed':400},
                                                                           'LAN_3':{'count':1, 'speed':25},
                                                                           'LAN_4':{'count':0, 'speed':25},
                                                                           'LAN_5':{'count':0, 'speed':1},
                                                                            'LAN_6':{'count':0, 'speed':400},
                      
                      },
        'Kubernetes': {'count':0, 'wattage': 471, 'height': 1, 'weight': 20, 'LAN_1':{'count':0,'speed':400},
                                                                            'LAN_2':{'count':1,'speed':200},
                                                                           'LAN_3':{'count':1, 'speed':25},
                                                                           'LAN_4':{'count':0, 'speed':25},
                                                                           'LAN_5':{'count':0, 'speed':1},
                                                                            'LAN_6':{'count':0, 'speed':400},
                      
                      },
        'login': {'count':0, 'wattage': 471, 'height': 1, 'weight': 20, 'LAN_1':{'count':0,'speed':400},
                                                                            'LAN_2':{'count':1,'speed':200},
                                                                           'LAN_3':{'count':1, 'speed':25},
                                                                           'LAN_4':{'count':0, 'speed':25},
                                                                           'LAN_5':{'count':0, 'speed':1},
                                                                            'LAN_6':{'count':0, 'speed':400},
                      
                      },
        'slurm': {'count':0, 'wattage': 501, 'height': 1, 'weight': 20, 'LAN_1':{'count':0,'speed':400},
                                                                            'LAN_2':{'count':1,'speed':200},
                                                                           'LAN_3':{'count':1, 'speed':25},
                                                                           'LAN_4':{'count':0, 'speed':25},
                                                                           'LAN_5':{'count':0, 'speed':1},
                                                                            'LAN_6':{'count':0, 'speed':400},
                      
                      },
        'ost_beegfs': {'count':0, 'wattage': 965, 'height': 2, 'weight': 36.1, 'LAN_1':{'count':0,'speed':400},
                                                                            'LAN_2':{'count':2,'speed':200},
                                                                           'LAN_3':{'count':1, 'speed':25},
                                                                           'LAN_4':{'count':0, 'speed':25},
                                                                           'LAN_5':{'count':0, 'speed':1},
                                                                            'LAN_6':{'count':0, 'speed':400},
                      
                      },
        'MDS_beegfs': {'count': 0, 'wattage': 887, 'height': 2, 'weight':36.1, 'LAN_1':{'count':0,'speed':400},
                                                                            'LAN_2':{'count':2,'speed':200},
                                                                           'LAN_3':{'count':1, 'speed':25},
                                                                           'LAN_4':{'count':0, 'speed':25},
                                                                           'LAN_5':{'count':0, 'speed':1},
                                                                          'LAN_6':{'count':0, 'speed':400},
                         },
        'mgmt': {'count':0, 'wattage': 646, 'height': 2, 'weight': 36.1, 'LAN_1':{'count':0,'speed':400},
                                                                            'LAN_2':{'count':1,'speed':200},
                                                                           'LAN_3':{'count':1, 'speed':25},
                                                                           'LAN_4':{'count':0, 'speed':25},
                                                                           'LAN_5':{'count':0, 'speed':1},
                                                                            'LAN_6':{'count':0, 'speed':400},
                      
                      },
        'NFS_beegfs': {'count': 0, 'wattage': 479, 'height': 2, 'weight':25.1, 'LAN_1':{'count':0,'speed':400},
                                                                            'LAN_2':{'count':2,'speed':200},
                                                                            'LAN_3':{'count':1, 'speed':25},
                                                                           'LAN_4':{'count':0, 'speed':25},
                                                                           'LAN_5':{'count':0, 'speed':1},
                                                                         'LAN_6':{'count':0, 'speed':400},
                          },
                   
    }
    LANs = { 'LAN_1':{'type':'400gbsNDR','topology': 'halfports_spine_leaf',
                  'switch':[{'model':'IB+400','ports':64,'speed':400, 'height':2,'wattage':2000, 'weight':20},
                         {'model':'IB_800','ports':64,'speed':800, 'height':2,'wattage':2000, 'weight':20},
                           ]},
              'LAN_2':{'type':'400GbpsEth', 'topology': 'uplinks_spine_leaf',
'switch':[{'model':'Z_9xxx','ports':64,'speed':400, 'height':2,'wattage':1304, 'uplink_count':4, 'uplink_speed':800,'weight':20},
                 {'model':'Z_6xxx','ports':64,'speed':200, 'height':1,'wattage':1304, 'uplink_count':4, 'uplink_speed':400,'weight':20},
        ]},
              'LAN_3':{'type':'25gbps','topology': 'uplinks_spine_leaf',
        'switch':[{'model':'S_xx64','ports':64,'speed':25, 'height':2,'wattage':300, 'uplink_count':4, 'uplink_speed':100,'weight':12},
                {'model':'S_xx32','ports':32,'speed':25, 'height':1,'wattage':200,'uplink_count':4, 'uplink_speed':100,'weight':12},
                 ]},
              'LAN_4':{'type':'25gbps','topology': 'uplinks_spine_leaf',
        'switch':[{'model':'S_xx64','ports':64,'speed':25, 'height':2,'wattage':200,'uplink_count':4, 'uplink_speed':100,'weight':12},
                {'model':'S_xx32','ports':32,'speed':25, 'height':1,'wattage':200,'uplink_count':2, 'uplink_speed':100,'weight':12},
                 ]},
              'LAN_5':{'type':'1gbps','topology': 'uplinks_spine_leaf',
    'switch':[{'model':'SN_24','ports':24,'speed':1, 'height':1,'wattage':200, 'uplink_count':2, 'uplink_speed':25, 'weight':6},
             ]},
              'LAN_6':{'type':'400GbpsEth','topology':'rails',
    'switch':[{'model':'Z_9xxx','ports':64,'speed':400, 'height':2,'wattage':2200, 'uplink_count':4, 'uplink_speed':800,'weight':20},
             ]},
           }
    Cables = {'IB+400':[{'model':'1.5m_400IB_copper','server_port_speed':400,'split':1,'length':1.5},
                                           {'model':'3m_400IB_copper','server_port_speed':400,'split':1,'length':3},
                                         {'model':'5m_400IB_fiber','server_port_speed':400,'split':1,'length':5},
                                      { 'model':'7m_400IB_fiber','server_port_speed':400,'split':1,'length':7},
                                      { 'model':'10m_400IB_fiber','server_port_speed':400,'split':1,'length':10},
                                      { 'model':'20m_400IB_fiber','server_port_speed':400,'split':1,'length':20},
                                      { 'model':'3m_400sIB_copper','server_port_speed':200,'split':2,'length':3},
                                     {  'model':'5m_400sIB_fiber','server_port_speed':200,'split':2,'length':5},
                                      { 'model':'7m_400sIB_fiber','server_port_speed':200,'split':2,'length':7},
                                      { 'model':'10m_400sIB_fiber','server_port_speed':200,'split':2,'length':10},
                                     {  'model':'20m_400sIB_fiber','server_port_speed':200,'split':2,'length':20},
                            ],
                            
          'IB_800':[{'model':'1.5m_800IB_copper','server_port_speed':800,'split':1,'length':1.5},
                                       {'model':'3m_800IB_copper','server_port_speed':800,'split':1,'length':3},
                                      { 'model':'8m_800IB_fiber','server_port_speed':800,'split':1,'length':5},
                                      { 'model':'7m_800IB_fiber','server_port_speed':800,'split':1,'length':7},
                                      { 'model':'10m_800IB_fiber','server_port_speed':800,'split':1,'length':10},
                                       {'model':'20m_800IB_fiber','server_port_speed':800,'split':1,'length':20},
                                      { 'model':'3m_800sIB_copper','server_port_speed':400,'split':2,'length':3},
                                     {  'model':'5m_800sIB_fiber','server_port_speed':400,'split':2,'length':5},
                                      { 'model':'7m_800sIB_fiber','server_port_speed':400,'split':2,'length':7},
                                      { 'model':'10m_800sIB_fiber','server_port_speed':400,'split':2,'length':10},
                                      { 'model':'20m_800sIB_fiber','server_port_speed':400,'split':2,'length':20},
                   ],
          'S_xx64':[{'model':'1.5m_25_copper','server_port_speed':25,'split':1,'length':1.5},
                                       {'model':'3m_25_copper','server_port_speed':25,'split':1,'length':3},
                                       {'model':'5m_25_fiber','server_port_speed':25,'split':1,'length':5},
                                     {  'model':'7m_25_fiber','server_port_speed':25,'split':1,'length':7},
                                       {'model':'10m_25_fiber','server_port_speed':25,'split':1,'length':10},
                                       {'model':'20m_25_fiber','server_port_speed':25,'split':1,'length':20},                                       
                   ],
          'S_xx32':[{'model':'1.5m_25_copper','server_port_speed':25,'split':1,'length':1.5},
                                       {'model':'3m_25_copper','server_port_speed':25,'split':1,'length':3},
                                       {'model':'5m_25_fiber','server_port_speed':25,'split':1,'length':5},
                                       {'model':'7m_25_fiber','server_port_speed':25,'split':1,'length':7},
                                       {'model':'10m_25_fiber','server_port_speed':25,'split':1,'length':10},
                                       {'model':'20m_25_fiber','server_port_speed':25,'split':1,'length':20},                                       
                   ],
          'SN_24':[{'model':'1.5m_1_copper','server_port_speed':1,'split':1,'length':1.5},
                                      { 'model':'3m_1_coppe','server_port_speed':1,'split':1,'length':3},
                                       {'model':'5m_1_copper','server_port_speed':1,'split':1,'length':5},
                                      { 'model':'7m_1_copper','server_port_speed':1,'split':1,'length':7},
                                      { 'model':'10m_1_copper','server_port_speed':1,'split':1,'length':10},
                                      { 'model':'20m_1_copper','server_port_speed':1,'split':1,'length':20}, 
                                      { 'model':'30m_1_copper','server_port_speed':1,'split':1,'length':30},
                                       {'model':'40m_1_copper','server_port_speed':1,'split':1,'length':40},
                  ],
          'Z_9xxx':[{'model':'1.5m_400_copper_eth','server_port_speed':400,'split':1,'length':1.5},
                                       {'model':'3m_400_copper_eth','server_port_speed':400,'split':1,'length':3},
                                      { 'model':'5m_400_fiber_eth','server_port_speed':400,'split':1,'length':5},
                                      { 'model':'7m_400_fiber_eth','server_port_speed':400,'split':1,'length':7},
                                     {  'model':'10m_400_fiber_eth','server_port_speed':400,'split':1,'length':10},
                                      { 'model':'20m_400_fiber_eth','server_port_speed':400,'split':1,'length':20},
                                        {'model':'1.5m_400s_copper_eth','server_port_speed':200,'split':2,'length':1.5},
                                      { 'model':'3m_400s_copper_eth','server_port_speed':200,'split':2,'length':3},
                                     {  'model':'5m_400s_fiber_eth','server_port_speed':200,'split':2,'length':5},
                                     {  'model':'7m_400s_fiber_eth','server_port_speed':200,'split':2,'length':7},
                                      { 'model':'10m_400s_fiber_eth','server_port_speed':200,'split':2,'length':10},
                                      { 'model':'20m_400s_fiber_eth','server_port_speed':200,'split':2,'length':20},
                   ],
         }         
            
     
    max_box_wattage = 16000
    max_box_height = rack_height_u = 42
    rack_weight_kg = 114.55
    rack_width_mm = 750
    rack_depth_mm = 1200
    u_height_mm = 44.45
    rack_height_mm = rack_height_u * u_height_mm
    rack_cg_height_mm = rack_height_mm / 2
    global_rack_signature = dict()
    unit_to_cm = u_height_mm /10
    # the following are in units
    device_to_rackside = 6.75
    racktop_to_ceiling = 4.5 # assumed 20cm
    rack_to_rack = 4.5 # assumed from side to the adjacent side no spacing
    batch_size = 1
    max_iter = 5000000
    # adding to the Cables the maximum stretch
    for switch in Cables:
        for cable in Cables[switch]:
            cable['in_rack_stretch'] = floor((cable['length']*100/unit_to_cm) - (2* device_to_rackside))
            
            
if __name__ == "__main__":
    global_main()
    main_iterative_greedy_adaptive_with_height()
    print('fffffffffffffffffffffffffffffffffffffff')


# In[ ]:


check_box_limits.cache_info()


# In[ ]:


find_stable_positions_greedy_complex.cache_info()


# In[ ]:


calculate_box_wattage_height.cache_info()


# In[ ]:


is_switch_relevent.cache_info()


# In[ ]:


ceil(5/3)


# In[ ]:


Cablessd


# In[ ]:




