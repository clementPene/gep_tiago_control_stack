"""
Visualization utilities for OCP trajectories using MeshCat.
"""

import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.transformations as tf
from rclpy.node import Node


class TrajectoryVisualizer:
    """
    Handles MeshCat visualization for robot trajectories.
    """

    def __init__(
        self,
        viz: MeshcatVisualizer,
        model: pin.Model,
        data: pin.Data,
        frame_id: int,
        target_position: np.ndarray,
        dt: float,
        logger: Node,
    ):
        """
        Args:
            viz: MeshCat visualizer instance
            model: Pinocchio model
            data: Pinocchio data
            frame_id: End-effector frame ID
            target_position: Target position (3D vector)
            dt: Time step between nodes
            logger: ROS2 logger for info/warnings
        """
        self.viz = viz
        self.model = model
        self.data = data
        self.frame_id = frame_id
        self.target_position = target_position
        self.dt = dt
        self.logger = logger

    def replay_trajectory(
        self, trajectory_q: list, fps: int = 30, slowdown: float = 1.0
    ):
        """
        Replay a trajectory in MeshCat with animated end-effector sphere.

        Args:
            trajectory_q: List of configurations [(nq,), ...]
            fps: Frames per second for animation
            slowdown: Factor to slow down (>1) or speed up (<1) the replay
        """
        if not trajectory_q or len(trajectory_q) == 0:
            self.logger.warn("No trajectory to replay!")
            return

        self.logger.info(
            f"Replaying trajectory:\n"
            f"Nodes: {len(trajectory_q)}\n"
            f"Duration: {len(trajectory_q) * self.dt:.2f}s\n"
            f"FPS: {fps}\n"
            f"Slowdown: {slowdown}x"
        )

        # Compute delay between frames
        delay = (1.0 / fps) * slowdown

        # Display initial state
        self.viz.display(trajectory_q[0])
        time.sleep(1.0)  # Pause at start

        # Replay each configuration
        for i, q in enumerate(trajectory_q):
            self.viz.display(q)

            # Update EE sphere position
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.frame_id)
            ee_pos = self.data.oMf[self.frame_id].translation.copy()

            self.viz.viewer["ee_sphere"].set_transform(tf.translation_matrix(ee_pos))

            # Print progress every 10 frames
            if i % 10 == 0:
                distance_to_target = np.linalg.norm(ee_pos - self.target_position)
                self.logger.info(
                    f"  Frame {i}/{len(trajectory_q)} | "
                    f"Distance to target: {distance_to_target:.4f}m"
                )

            time.sleep(delay)

        # Hold final state
        self.logger.info("Replay completed!")
        time.sleep(2.0)

    def display_results(self, xs_solution: list, us_solution: list, cost: float):
        """
        Display OCP solution results with statistics.

        Args:
            xs_solution: State trajectory [(2*nq,), ...]
            us_solution: Control trajectory [(nu,), ...]
            cost: Final OCP cost
        """
        self.logger.info("=" * 60)
        self.logger.info("OCP SOLUTION RESULTS")
        self.logger.info("=" * 60)

        # Final cost
        self.logger.info(f"Final cost: {cost:.6e}")

        # Check final end-effector position
        q_final = xs_solution[-1][: self.model.nq]
        pin.forwardKinematics(self.model, self.data, q_final)
        pin.updateFramePlacement(self.model, self.data, self.frame_id)
        ee_pos_final = self.data.oMf[self.frame_id].translation

        error = np.linalg.norm(ee_pos_final - self.target_position)

        self.logger.info(f"Final EE position: {ee_pos_final}")
        self.logger.info(f"Target position:   {self.target_position}")
        self.logger.info(f"Position error:    {error:.6f} m")

        # Control statistics
        u_norms = [np.linalg.norm(u) for u in us_solution]
        self.logger.info("Control effort:")
        self.logger.info(f"   - Max:  {np.max(u_norms):.3f}")
        self.logger.info(f"   - Mean: {np.mean(u_norms):.3f}")
        self.logger.info(f"   - Min:  {np.min(u_norms):.3f}")

        # Joint limits check
        q_limits_violated = False
        for i, q in enumerate(xs_solution):
            q_vec = q[: self.model.nq]

            # Check chaque coordonnée
            for j in range(self.model.nq):
                q_value = q_vec[j]
                lower = self.model.lowerPositionLimit[j]
                upper = self.model.upperPositionLimit[j]

                if q_value < lower:
                    self.logger.warn(
                        f"Step {i}: q[{j}] BELOW limit: "
                        f"{q_value:.4f} < {lower:.4f} (violation: {lower - q_value:.4f})"
                    )
                    q_limits_violated = True

                elif q_value > upper:
                    self.logger.warn(
                        f"Step {i}: q[{j}] ABOVE limit: "
                        f"{q_value:.4f} > {upper:.4f} (violation: {q_value - upper:.4f})"
                    )
                    q_limits_violated = True

            if q_limits_violated:
                break

        if not q_limits_violated:
            self.logger.info("All joint limits respected")

        self.logger.info("=" * 60)

    def replay_trajectory_loop(
        self,
        trajectory_q: list,
        fps: int = 30,
        slowdown: float = 1.0,
        pause_between_loops: float = 1.0,
    ):
        """
        Replay trajectory in infinite loop until interrupted.

        Args:
            trajectory_q: List of configurations [(nq,), ...]
            fps: Frames per second for animation
            slowdown: Factor to slow down (>1) or speed up (<1)
            pause_between_loops: Pause duration (seconds) between loops
        """
        if not trajectory_q or len(trajectory_q) == 0:
            self.logger.warn("No trajectory to replay!")
            return

        self.logger.info(
            f"Starting infinite replay loop:\n"
            f"  Nodes: {len(trajectory_q)}\n"
            f"  Duration: {len(trajectory_q) * self.dt:.2f}s\n"
            f"  FPS: {fps}\n"
            f"  Slowdown: {slowdown}x\n"
            f"  Press Ctrl+C to stop"
        )

        loop_count = 0
        try:
            while True:
                loop_count += 1
                self.logger.info(f"Loop {loop_count}")

                # Replay once
                self.replay_trajectory(
                    trajectory_q=trajectory_q, fps=fps, slowdown=slowdown
                )

                # Pause between loops
                time.sleep(pause_between_loops)

        except KeyboardInterrupt:
            self.logger.info(f"\nReplay stopped after {loop_count} loops")
