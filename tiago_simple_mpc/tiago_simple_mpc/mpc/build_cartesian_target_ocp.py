from attr import dataclass
import numpy as np
import crocoddyl
import pinocchio as pin
import yaml

import os
from ament_index_python.packages import get_package_share_directory

from tiago_simple_mpc.ocp.ocp_builder import OCPBuilder
from tiago_simple_mpc.ocp.cost_manager import CostModelManager
from tiago_simple_mpc.mpc.mpc_builder import MPCOCP


@dataclass
class CartesianOCPConfig:
    """Configuration for Cartesian target OCP."""

    dt: float
    horizon_length: int
    max_iterations: int
    has_free_flyer: bool
    ee_tracking_weight: float
    state_reg_weight: float
    control_reg_weight: float
    terminal_weight_multiplier: float
    frame_name: str
    default_target_position: list
    default_target_quaternion: list

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "CartesianOCPConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to the YAML configuration file

        Returns:
            CartesianOCPConfig instance
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        config_data = data.get("cartesian_ocp", {})

        return cls(
            dt=config_data.get("dt", 0.01),
            horizon_length=config_data.get("horizon_length", 100),
            max_iterations=config_data.get("max_iterations", 20),
            has_free_flyer=config_data.get("has_free_flyer", False),
            ee_tracking_weight=config_data.get("ee_tracking_weight", 1000.0),
            state_reg_weight=config_data.get("state_reg_weight", 0.01),
            control_reg_weight=config_data.get("control_reg_weight", 0.001),
            terminal_weight_multiplier=config_data.get(
                "terminal_weight_multiplier", 10.0
            ),
            frame_name=config_data.get("frame_name", "gripper_grasping_frame"),
            default_target_position=config_data.get("default_target", {}).get(
                "position", None
            ),
            default_target_quaternion=config_data.get("default_target", {}).get(
                "quaternion", None
            ),
        )

    @classmethod
    def from_package(
        cls,
        package_name: str = "tiago_simple_mpc",
        config_filename: str = "cartesian_target_ocp_config.yaml",
    ) -> "CartesianOCPConfig":
        """Load configuration from ROS 2 package.

        Args:
            package_name: Name of the ROS 2 package
            config_filename: Name of the config file

        Returns:
            CartesianOCPConfig instance
        """
        pkg_share = get_package_share_directory(package_name)
        yaml_path = os.path.join(pkg_share, "config", config_filename)
        return cls.from_yaml(yaml_path)

    def get_default_target_pose(self) -> pin.SE3:
        """Convert default_target config to pin.SE3 pose."""
        if self.default_target_position is None:
            raise ValueError("No default_target defined in config!")

        pos = np.array(self.default_target_position)
        quat = pin.Quaternion(
            self.default_target_quaternion[3],  # w
            self.default_target_quaternion[0],  # x
            self.default_target_quaternion[1],  # y
            self.default_target_quaternion[2],  # z
        )
        return pin.SE3(quat.matrix(), pos)


def build_cartesian_target_ocp(
    x0: np.ndarray,
    model: pin.Model,
    config: CartesianOCPConfig,
    target_pose: pin.SE3 = None,
) -> MPCOCP:
    """Builds a Crocoddyl OCP for reaching a Cartesian target with the end-effector.

    Args:
        x0: Initial state
        model: Pinocchio model
        config: OCP configuration (contient frame_name et default_target)
        target_pose: Target pose (optionnel, utilise config.default_target si None)
    """

    # Utilise la target par défaut si non fournie
    if target_pose is None:
        if config.default_target_position is None:
            raise ValueError("No target_pose provided and no default_target in config!")

        # Convertir default_target en pin.SE3
        pos = np.array(config.default_target_position)
        quat = pin.Quaternion(
            config.default_target_quaternion[3],  # w
            config.default_target_quaternion[0],  # x
            config.default_target_quaternion[1],  # y
            config.default_target_quaternion[2],  # z
        )
        target_pose = pin.SE3(quat.matrix(), pos)

    # Utilise frame_name de la config
    frame_name = config.frame_name

    # Build OCP using OCPBuilder
    ocp_builder = OCPBuilder(
        initial_state=x0,
        rmodel=model,
        dt=config.dt,
        horizon_length=config.horizon_length,
        has_free_flyer=config.has_free_flyer,
        wheel_params=None,
    )

    # Create costs
    running_cost_manager = CostModelManager(ocp_builder.state, ocp_builder.actuation)

    # Cost 1: Reach the target
    running_cost_manager.add_frame_placement_cost(
        frame_name=frame_name, target_pose=target_pose, weight=config.ee_tracking_weight
    )

    # Cost 2: State regularization
    pkg_share = get_package_share_directory("tiago_simple_mpc")
    state_weights_config = os.path.join(
        pkg_share, "config", "regulation_state_weights.yaml"
    )
    running_cost_manager.add_weighted_regulation_state_cost(
        x_ref=x0,
        config_filepath=state_weights_config,
        weight=config.state_reg_weight,
    )

    # Cost 3: Control regularization
    control_weights_config = os.path.join(
        pkg_share, "config", "regulation_control_weights.yaml"
    )
    running_cost_manager.add_weighted_regulation_control_cost(
        config_filepath=control_weights_config,
        weight=config.control_reg_weight,
    )

    # Terminal cost
    terminal_cost_manager = CostModelManager(ocp_builder.state, ocp_builder.actuation)
    terminal_cost_manager.add_frame_placement_cost(
        frame_name=frame_name,
        target_pose=target_pose,
        weight=config.ee_tracking_weight * config.terminal_weight_multiplier,
    )

    # Finalize OCP
    problem = ocp_builder.build(
        running_cost_managers=[running_cost_manager] * config.horizon_length,
        terminal_cost_manager=terminal_cost_manager,
        integrator_type="euler",
    )
    solver = crocoddyl.SolverFDDP(problem)

    return MPCOCP(
        problem=problem,
        solver=solver,
        dt=config.dt,
        horizon_length=config.horizon_length,
    )
