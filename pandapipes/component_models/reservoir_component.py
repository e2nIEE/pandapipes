from numpy import dtype

from pandapipes.component_models.ext_grid_component import ExtGrid


try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class Reservoir(ExtGrid):
    """

    """

    @classmethod
    def table_name(cls):
        return "reservoir"

    @classmethod
    def get_component_input(cls):
        """

        :return:
        """

        return [("name", dtype(object)),
                ("junction", "u4"),
                ("h_m", "f8"),
                ("p_bar", "f8"),
                ("t_k", "f8"),
                ("in_service", "bool"),
                ('type', dtype(object))]
