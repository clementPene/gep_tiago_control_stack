import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os


class MPCLogger:
    """Simple MPC plotter"""

    def __init__(self, log_dir="log_mpc"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Data storage
        self.timestamps = []
        self.ee_positions = []
        self.ee_targets = []
        self.joint_states = []
        self.joint_velocities = []
        self.controls = []
        self.costs = []
        self.solve_times = []
        self.converged_flags = []

    def log_step(
        self,
        timestamp: float,
        ee_position: np.ndarray,
        ee_target: np.ndarray,
        joint_state: np.ndarray,
        joint_velocity: np.ndarray,
        control: np.ndarray,
        cost: float,
        solve_time: float,
        converged: bool,
    ):
        """Log one MPC step"""
        self.timestamps.append(timestamp)
        self.ee_positions.append(ee_position.copy())
        self.ee_targets.append(ee_target.copy())
        self.joint_states.append(joint_state.copy())
        self.joint_velocities.append(joint_velocity.copy())
        self.controls.append(control.copy())
        self.costs.append(cost)
        self.solve_times.append(solve_time)
        self.converged_flags.append(converged)

    def plot_results(self, nq: int) -> str:
        """Generate plots from accumulated data"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert to numpy
        timestamps = np.array(self.timestamps)
        ee_positions = np.array(self.ee_positions)
        ee_targets = np.array(self.ee_targets)
        joint_states = np.array(self.joint_states)
        costs = np.array(self.costs)
        solve_times = np.array(self.solve_times) * 1000  # Convert to ms

        # Compute tracking error
        ee_errors = np.linalg.norm(ee_positions - ee_targets, axis=1) * 1000  # mm

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle("MPC Execution Results", fontsize=16, fontweight="bold")

        # 1. End-effector tracking
        axes[0].set_title("End-Effector Tracking", fontweight="bold")
        axes[0].plot(timestamps, ee_positions[:, 0], "r-", label="EE X", linewidth=2)
        axes[0].plot(timestamps, ee_targets[:, 0], "r--", label="Target X", alpha=0.7)
        axes[0].plot(timestamps, ee_positions[:, 1], "g-", label="EE Y", linewidth=2)
        axes[0].plot(timestamps, ee_targets[:, 1], "g--", label="Target Y", alpha=0.7)
        axes[0].plot(timestamps, ee_positions[:, 2], "b-", label="EE Z", linewidth=2)
        axes[0].plot(timestamps, ee_targets[:, 2], "b--", label="Target Z", alpha=0.7)
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Position (m)")
        axes[0].legend(ncol=3, loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Add error on right y-axis
        ax0_twin = axes[0].twinx()
        ax0_twin.plot(
            timestamps, ee_errors, "k:", linewidth=2, alpha=0.5, label="Error"
        )
        ax0_twin.set_ylabel("Tracking Error (mm)", color="k")
        ax0_twin.tick_params(axis="y", labelcolor="k")

        # Add stats text box
        avg_error = np.mean(ee_errors)
        max_error = np.max(ee_errors)
        final_error = ee_errors[-1]

        stats_text = (
            f"Mean error: {avg_error:.2f} mm\n"
            f"Max error: {max_error:.2f} mm\n"
            f"Final error: {final_error:.2f} mm"
        )

        axes[0].text(
            0.02,
            0.98,
            stats_text,
            transform=axes[0].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
            fontsize=9,
            fontfamily="monospace",
        )

        # 2. Joint positions
        axes[1].set_title("Joint Positions", fontweight="bold")
        for i in range(nq):
            axes[1].plot(
                timestamps, joint_states[:, i], label=f"q{i + 1}", linewidth=1.5
            )
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Position (rad)")
        axes[1].legend(ncol=min(nq, 4), loc="best")
        axes[1].grid(True, alpha=0.3)

        # 3. MPC Performance
        axes[2].set_title("MPC Performance", fontweight="bold")
        ax2 = axes[2]
        ax2_twin = ax2.twinx()

        # Cost on left y-axis
        ax2.plot(timestamps, costs, "b-", linewidth=2, label="Cost")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Cost", color="b")
        ax2.tick_params(axis="y", labelcolor="b")
        ax2.grid(True, alpha=0.3)

        # Solve time on right y-axis
        ax2_twin.plot(timestamps, solve_times, "r-", linewidth=2, label="Solve Time")
        ax2_twin.axhline(
            y=10,
            color="orange",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="Real-time limit (10ms)",
        )
        ax2_twin.set_ylabel("Solve Time (ms)", color="r")
        ax2_twin.tick_params(axis="y", labelcolor="r")

        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Add performance stats
        avg_solve_time = np.mean(solve_times)
        max_solve_time = np.max(solve_times)
        convergence_rate = np.mean(self.converged_flags) * 100
        realtime_rate = np.mean(solve_times < 10) * 100

        perf_text = (
            f"Avg solve: {avg_solve_time:.2f} ms\n"
            f"Max solve: {max_solve_time:.2f} ms\n"
            f"Converged: {convergence_rate:.1f}%\n"
            f"Real-time: {realtime_rate:.1f}%"
        )

        ax2.text(
            0.02,
            0.98,
            perf_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.9),
            fontsize=9,
            fontfamily="monospace",
        )

        plt.tight_layout()

        # Save
        filepath = os.path.join(self.log_dir, f"mpc_results_{timestamp}.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\n{'=' * 60}")
        print("📊 MPC Results Summary")
        print(f"{'=' * 60}")
        print(f"Duration: {timestamps[-1]:.2f} s")
        print(f"Iterations: {len(timestamps)}")
        print("\nTracking:")
        print(f"  - Mean error: {avg_error:.2f} mm")
        print(f"  - Max error: {max_error:.2f} mm")
        print(f"  - Final error: {final_error:.2f} mm")
        print("\nPerformance:")
        print(f"  - Avg solve time: {avg_solve_time:.2f} ms")
        print(f"  - Max solve time: {max_solve_time:.2f} ms")
        print(f"  - Convergence rate: {convergence_rate:.1f}%")
        print(f"  - Real-time rate: {realtime_rate:.1f}%")
        print(f"\nPlot saved: {filepath}")
        print(f"{'=' * 60}\n")

        return filepath
