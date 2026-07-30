"""Minimal truck--UAV integration example using final Xinjiekou data."""

from ground_network import get_launch_recovery_nodes, shortest_ground_path
from uav_path_planner import plan_uav_path


nodes = get_launch_recovery_nodes("xinjiekou")
start, goal = nodes.iloc[0], nodes.iloc[1]
ground = shortest_ground_path("xinjiekou", start.node_id, goal.node_id)
flight = plan_uav_path("xinjiekou", start.geometry, goal.geometry)

print(f"Ground: {ground['distance_m']:.1f} m, {ground['travel_time_s']:.1f} s")
print(f"UAV: {flight['path_mode']} at {flight['selected_height_m']} m, {flight['distance_m']}")
