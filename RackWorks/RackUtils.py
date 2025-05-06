import functools


import json
from collections import Counter, deque
import redis

from Rack_global_vars import colors_info, LANs,Cables, max_box_wattage, max_box_height, rack_height_u, rack_weight_kg,rack_width_mm
from Rack_global_vars import rack_depth_mm, u_height_mm, rack_height_mm, rack_height_u, rack_cg_height_mm, unit_to_cm, u_height_mm
from Rack_global_vars import device_to_rackside, racktop_to_ceiling, rack_to_rack, batch_size, max_iter,  colors, switch_to_add
from Rack_global_vars import color_metadata_tuple, colors_weight, colors_height, colors_wattage, total_balls



@functools.cache
def get_redis_connection(ip="192.168.8.10"):
    
    """Initialize and return a Redis connection."""
    return redis.Redis(host=ip, port=6379, db=0)

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
                    # absolute cable  lengeth in meters:
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

def get_distribution_signature(distribution):
        #start_time = time.time()
        filled_boxes_signature = tuple(sorted(tuple(sorted(box.items())) for box in distribution if box))
        #end_time = time.time()
        #print(f"Update: the signing the distribution is calcualted in {end_time - start_time:.6f} seconds")
        return filled_boxes_signature

def make_hashable(obj):
    """Recursively convert objects to hashable types."""
    if isinstance(obj, (list, deque)):
        return tuple(make_hashable(x) for x in obj)
    elif isinstance(obj, dict):
        return tuple((k, make_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, set):
        return frozenset(make_hashable(x) for x in obj)
    return obj

def write_to_redis(key, value):
    """Store Python objects in Redis with complete type safety."""
    red = get_redis_connection()
    if isinstance(value, (list, dict, Counter, deque, set)):
        if isinstance(value, set):
            # Store the set data directly, don't try to make it hashable here
            # We'll just store the serializable version of each item
            serializable_set = [make_serializable(x) for x in value]
            type_name = "set"
            data = serializable_set
        elif isinstance(value, deque):
            data = list(value)
            type_name = "deque"
        elif isinstance(value, Counter):
            data = dict(value)
            type_name = "Counter"
        else:
            data = value
            type_name = type(value).__name__
        
        wrapped = {
            "__type__": type_name,
            "__data__": data
        }
        red.set(key, json.dumps(wrapped))
    else:
        red.set(key, value)

def make_serializable(obj):
    """Convert hashable types to JSON-serializable types."""
    if isinstance(obj, tuple):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, frozenset):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, deque)):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, set):
        return [make_serializable(x) for x in obj]
    return obj

def read_from_redis( key):
    """Retrieve objects from Redis with perfect type reconstruction."""
    red = get_redis_connection()
    value = red.get(key)
    if value is None:
        return None
    
    try:
        data = json.loads(value)
        if isinstance(data, dict) and "__type__" in data:
            if data["__type__"] == "set":
                # Convert each item to a hashable type before adding to set
                result_set = set()
                for item in data["__data__"]:
                    try:
                        hashable_item = make_hashable(item)
                        result_set.add(hashable_item)
                    except TypeError:
                        # If an item can't be made hashable, skip it
                        pass
                return result_set
            elif data["__type__"] == "deque":
                return deque(data["__data__"])
            elif data["__type__"] == "Counter":
                return Counter(data["__data__"])
            elif data["__type__"] in ("dict", "list"):
                return data["__data__"]
        return data
    except json.JSONDecodeError:
        return value


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

@functools.cache 
def check_box_limits(box_tuple, color_metadata_tuple, max_box_wattage, max_box_height):
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

import numpy as np
from collections import Counter
import functools
import time
from functools import cache, lru_cache, wraps


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



@functools.cache
def find_stable_positions_greedy_complex( servers_to_place_tuple,  prioritize_top=False):
    """
    A greedy approach to find stable server positions for various server types inside one rack
    """
    #global global_rack_signature

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
