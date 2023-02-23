from dataclasses import dataclass

from models.vrp_node import VRPNode
from models.vrp_vehicle import VRPVehicle


@dataclass
class VRP:
    name: str  # name of the problem instance
    vehicles: [VRPVehicle]  # vehicles
    depot: VRPNode  # the depot node
    customers: [VRPNode]  # list of customer nodes

    def get_demands(self) -> dict[VRPNode, float]:
        return {node: node.demand for node in self.get_all_nodes()}

    def get_service_times(self) -> dict[VRPNode, float]:
        return {node: node.service_time for node in self.get_all_nodes()}

    def get_ready_times(self) -> dict[VRPNode, float]:
        return {node: node.ready_time for node in self.get_all_nodes()}

    def get_due_dates(self) -> dict[VRPNode, float]:
        return {node: node.due_date for node in self.get_all_nodes()}

    def get_vehicle_capacities(self) -> dict[VRPVehicle, float]:
        return {vehicle: vehicle.capacity for vehicle in self.vehicles}

    def get_all_nodes(self) -> [VRPNode]:
        return [self.depot] + self.customers

    def get_all_nodes_per_vehicle(self) -> [VRPNode, VRPVehicle]:
        return [(i, k) for i in self.get_all_nodes() for k in self.vehicles]

    def get_arcs(self) -> [tuple[VRPNode, VRPNode]]:
        return [(i, j) for i in self.get_all_nodes() for j in self.get_all_nodes()]

    def get_arcs_per_vehicle(self) -> [tuple[VRPNode, VRPNode, VRPVehicle]]:
        return [(i, j, k) for i in self.get_all_nodes() for j in self.get_all_nodes() for k in self.vehicles]

    def __str__(self):
        return f"VRP instance: {self.name}"

    def __repr__(self):
        return f"VRP instance: {self.name}"

    def __len__(self):
        return len(self.get_all_nodes())

    def __getitem__(self, index):
        return self.get_all_nodes()[index]

    def __iter__(self):
        return iter(self.get_all_nodes())

    def __contains__(self, item: VRPNode):
        return item in self.get_all_nodes()

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __le__(self, other):
        return self.name <= other.name

    def __gt__(self, other):
        return self.name > other.name

    def __ge__(self, other):
        return self.name >= other.name
