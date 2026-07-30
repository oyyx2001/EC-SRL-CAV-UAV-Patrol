import itertools

from shapely.geometry import LineString

from ground_network import get_launch_recovery_nodes
from uav_path_planner import plan_uav_path
from uav_path_planner import planner


def _nodes():
    return get_launch_recovery_nodes("xinjiekou").iloc[:80]


def test_direct_path_interface():
    obstacles = planner._obstacles("xinjiekou", 120.0)
    for left, right in itertools.combinations(_nodes().itertuples(), 2):
        line = LineString([left.geometry, right.geometry])
        if planner._endpoint_is_safe(left.geometry, obstacles) and planner._endpoint_is_safe(right.geometry, obstacles) and planner._line_is_clear(line, obstacles):
            result = plan_uav_path("xinjiekou", left.geometry, right.geometry, candidate_heights=(120,))
            assert result["feasible"] and result["path_mode"] == "direct"
            return
    raise AssertionError("No direct 120 m test pair was found in final Xinjiekou launch nodes.")


def test_local_detour_interface():
    # This pair is a successful blocked-direct case from the extracted final-stage
    # Xinjiekou visibility-graph validation, avoiding a broad test-time search.
    nodes = get_launch_recovery_nodes("xinjiekou").set_index("node_id")
    start, goal = nodes.loc["ground_node_000449"].geometry, nodes.loc["ground_node_000644"].geometry
    result = plan_uav_path("xinjiekou", start, goal, candidate_heights=(20,), search_buffer_m=5.0)
    assert result["feasible"] and result["path_mode"] == "local_detour"


def test_road_following_fallback_geometry():
    nodes = _nodes()
    route = planner._road_fallback("xinjiekou", nodes.iloc[0].geometry, nodes.iloc[1].geometry, (120.0,))[0]
    assert route is not None
    assert route["geometry"].length > 0
    assert len(route["node_sequence"]) >= 2
