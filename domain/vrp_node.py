from dataclasses import dataclass


@dataclass
class VRPNode:
    id: int  # index of node
    x: float  # x coordinate of node
    y: float  # y coordinate of node
    demand: float  # demand of node

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return str(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
