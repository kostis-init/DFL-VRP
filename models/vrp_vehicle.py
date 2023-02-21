from dataclasses import dataclass


@dataclass
class VRPVehicle:
    id: int
    capacity: float

    def __str__(self):
        return f"Vehicle {self.id}"

    def __repr__(self):
        return f"Vehicle {self.id}"

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return self.id < other.id

    def __le__(self, other):
        return self.id <= other.id

    def __gt__(self, other):
        return self.id > other.id

    def __ge__(self, other):
        return self.id >= other.id