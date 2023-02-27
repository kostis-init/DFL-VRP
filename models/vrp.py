from dataclasses import dataclass

import numpy as np

from models.vrp_node import VRPNode
from models.vrp_vehicle import VRPVehicle


@dataclass
class VRP:
    name: str  # name of the problem instance
    vehicles: [VRPVehicle]  # vehicles
    depot: VRPNode  # the depot node
    customers: [VRPNode]  # list of customer nodes

    def __post_init__(self):
        self.nodes = [self.depot] + self.customers
        self.arcs = [(i, j, k) for i in self.nodes for j in self.nodes for k in self.vehicles]
        self.demands = {i: i.demand for i in self.nodes}
        self.service_times = {i: i.service_time for i in self.nodes}
        self.ready_times = {i: i.ready_time for i in self.nodes}
        self.due_dates = {i: i.due_date for i in self.nodes}
        self.vehicle_capacities = {i: i.capacity for i in self.vehicles}
        self.travel_times = {(i, j, k): i.service_time + np.hypot(i.x - j.x, i.y - j.y) / k.speed
                             for i in self.nodes for j in self.nodes for k in self.vehicles}

    def __str__(self):
        return f"VRP instance: {self.name}"

    def __repr__(self):
        return f"VRP instance: {self.name}"

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, index):
        return self.nodes[index]

    def __iter__(self):
        return iter(self.nodes)

    def __contains__(self, item: VRPNode):
        return item in self.nodes
