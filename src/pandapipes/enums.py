import enum

class PhysDomain(str, enum.Enum):
    """Physical domain for simulation components."""
    HYD = "hydraulics"
    HEAT = "heat_transfer"

    def __str__(self):
        """str representation of PhysDomain member.

        Overridden to allow string operations on members

        Example:
        >> f"active_{PhysDomain.HEAT}"  # "active_heat_transfer"
        """
        return self.value

class SimMode(str, enum.Enum):
    """Modes for pipeflow simulation."""
    HYD = "hydraulics"
    HEAT = "heat"
    SEQ = "sequential"
    BIDIR = "bidirectional"
    ALL = "all"

    def __str__(self):
        """str representation of SimMode member.

        Overridden to allow string operations on members

        Example:
        >> f"used mode: {SimMode.HEAT}"  # "used mode: heat"
        """
        return self.value