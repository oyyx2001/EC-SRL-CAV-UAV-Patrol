from ground_network import get_launch_recovery_nodes, load_ground_network, shortest_ground_path


def test_final_networks_load_without_fixed_node_count_logic():
    expected = {"xinjiekou": 333, "baijiahu": 129}
    for region, count in expected.items():
        network = load_ground_network(region)
        nodes = get_launch_recovery_nodes(region)
        assert len(nodes) == count
        assert len(network.intersection_nodes) + len(network.supplementary_nodes) == count
        assert network.compressed_edges.geometry.geom_type.isin(["LineString", "MultiLineString"]).all()


def test_shortest_path_keeps_real_road_geometry():
    nodes = get_launch_recovery_nodes("xinjiekou")
    route = shortest_ground_path("xinjiekou", nodes.iloc[0].node_id, nodes.iloc[1].node_id)
    assert route["distance_m"] > 0
    assert route["travel_time_s"] > 0
    assert len(route["node_sequence"]) >= 2
    assert route["geometry"].length > 0
