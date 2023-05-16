from enum import Enum


class SolverMode(Enum):
    """Enumeration of solver modes."""
    TRUE_COST = 1
    PRED_COST = 2
    NCE = 2
    SPO = 3
    DISTANCE = 4
    SPO_2 = 5

