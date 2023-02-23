from dataclasses import dataclass

from models.vrp_node import VRPNode


@dataclass
class VRPEdge:
    i: VRPNode
    j: VRPNode
    cost: float
    travel_time: float


