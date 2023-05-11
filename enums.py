from enum import Enum


class SolverMode(Enum):
    """Enumeration of solver modes."""
    TRUE_COST = 1
    """Actual solver mode."""
    PRED_COST = 2
    """Predicted solver mode."""
    NCE = 2
    """NCE solver mode."""
    SPO = 3
    """SPO solver mode."""
    DISTANCE = 4
    """Distance solver mode."""
