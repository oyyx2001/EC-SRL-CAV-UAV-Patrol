from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .road_network import RoadNetwork


@dataclass
class Vehicle:
    """CAV state with strict edge-by-edge movement on the directed road network.

    The earlier first-stage implementation used the directed shortest-path travel time
    but jumped directly from the current node to the target rendezvous node when the
    travel time elapsed. This version expands the shortest path into directed edges and
    advances the CAV along those edges. Therefore, every logged CAV movement satisfies
    ``network.G.has_edge(u, v)``.
    """

    id: str
    position: int
    target_node: Optional[int] = None
    at_intersection: bool = True

    # Total remaining time to the final target and remaining time on the current edge.
    remaining_time: float = 0.0
    edge_remaining_time: float = 0.0

    # Current shortest path being executed.
    path_nodes: List[int] = field(default_factory=list)
    path_index: int = 0

    # Node and edge logs. route_log stores node arrivals; route_edge_log stores actual
    # directed road edges traversed by the CAV.
    route_log: List[Tuple[float, int]] = field(default_factory=list)
    route_edge_log: List[Tuple[float, int, int]] = field(default_factory=list)

    wait_time: float = 0.0
    distance_time: float = 0.0
    last_arrival_time: Optional[float] = None
    schedule_version: int = 0

    def is_idle(self) -> bool:
        return self.target_node is None or self.remaining_time <= 1e-9

    def set_target(self, target: int, network: RoadNetwork, now: float) -> None:
        """Set a new CAV target and expand the directed shortest path to it."""
        target = int(target)
        self.schedule_version += 1
        self.last_arrival_time = None

        if target == self.position:
            self.target_node = None
            self.at_intersection = True
            self.remaining_time = 0.0
            self.edge_remaining_time = 0.0
            self.path_nodes = []
            self.path_index = 0
            return

        path = network.get_vehicle_path(self.position, target)
        if not path or len(path) < 2:
            raise ValueError(f"Vehicle target unreachable: {self.position}->{target}")

        path = [int(x) for x in path]
        for u, v in zip(path[:-1], path[1:]):
            if not network.G.has_edge(u, v):
                raise ValueError(f"Shortest path contains invalid directed edge: {u}->{v}")

        self.target_node = target
        self.at_intersection = True
        self.path_nodes = path
        self.path_index = 0
        self.remaining_time = sum(
            float(network.G[u][v]["vehicle_time_s"])
            for u, v in zip(path[:-1], path[1:])
        )
        self.edge_remaining_time = float(network.G[path[0]][path[1]]["vehicle_time_s"])

    def step(self, dt: float, now: float, network: RoadNetwork) -> bool:
        """Advance the CAV strictly along directed road-network edges.

        Parameters
        ----------
        dt:
            Simulation time increment.
        now:
            Start time of the current simulation increment.
        network:
            Directed road network containing edge travel times.

        Returns
        -------
        bool
            True if the CAV reaches its final target during this time increment.
        """
        self.last_arrival_time = None
        if self.target_node is None or not self.path_nodes:
            self.at_intersection = True
            return False

        remaining_dt = float(dt)
        elapsed = 0.0
        arrived_target = False

        while remaining_dt > 1e-9 and self.target_node is not None:
            if self.path_index >= len(self.path_nodes) - 1:
                # Defensive cleanup for a completed path.
                self.target_node = None
                self.remaining_time = 0.0
                self.edge_remaining_time = 0.0
                self.path_nodes = []
                self.path_index = 0
                self.at_intersection = True
                arrived_target = True
                self.last_arrival_time = now + elapsed
                break

            u = int(self.path_nodes[self.path_index])
            v = int(self.path_nodes[self.path_index + 1])
            if not network.G.has_edge(u, v):
                raise ValueError(f"CAV attempted invalid directed edge: {u}->{v}")

            if self.edge_remaining_time <= 1e-9:
                self.edge_remaining_time = float(network.G[u][v]["vehicle_time_s"])

            move = min(remaining_dt, self.edge_remaining_time)
            if move > 1e-9:
                self.at_intersection = False
            self.edge_remaining_time -= move
            self.remaining_time = max(0.0, self.remaining_time - move)
            self.distance_time += move
            remaining_dt -= move
            elapsed += move

            # Still on the same edge.
            if self.edge_remaining_time > 1e-9:
                break

            # The CAV has reached the next road-network node.
            arrival_time = now + elapsed
            self.position = v
            self.at_intersection = True
            self.route_edge_log.append((arrival_time, u, v))
            self.route_log.append((arrival_time, v))
            self.path_index += 1

            if self.position == self.target_node:
                self.target_node = None
                self.remaining_time = 0.0
                self.edge_remaining_time = 0.0
                self.path_nodes = []
                self.path_index = 0
                self.at_intersection = True
                arrived_target = True
                self.last_arrival_time = arrival_time
                break

            # Prepare the next directed edge on the path.
            next_u = int(self.path_nodes[self.path_index])
            next_v = int(self.path_nodes[self.path_index + 1])
            if not network.G.has_edge(next_u, next_v):
                raise ValueError(f"CAV attempted invalid directed edge: {next_u}->{next_v}")
            self.edge_remaining_time = float(network.G[next_u][next_v]["vehicle_time_s"])

        return arrived_target

    def add_wait(self, dt: float) -> None:
        self.wait_time += dt
