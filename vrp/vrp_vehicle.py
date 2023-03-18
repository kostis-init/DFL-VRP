from dataclasses import dataclass


@dataclass
class VRPVehicle:
    id: int
    capacity: float = 1000000.0
    consumption: float = 0.1
    fuel_capacity: float = 100.0
    speed: float = 10.0

    def __str__(self):
        return f"Vehicle {self.id}"

    def __repr__(self):
        return f"Vehicle {self.id}"

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
