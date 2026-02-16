import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os


class OCPLogger:
    """Simple OCP plotter"""

    def __init__(self, log_dir="log_ocp"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def plot_results(self, solver, nq, converged):
        """Generate plots directly from solver"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract data
        xs = solver.xs
        us = solver.us
        problem = solver.problem

        # Extract costs exactly like in your MPC project
        running_costs = []
        try:
            HORIZON_LENGTH = len(problem.runningModels)

            # Running costs
            for k in range(HORIZON_LENGTH):
                model_k = problem.runningModels[k]
                data_k = problem.runningDatas[k]
                x_k = solver.xs[k]
                u_k = solver.us[k]
                model_k.calc(data_k, x_k, u_k)
                running_costs.append(data_k.cost)

            # Terminal cost
            terminal_model = problem.terminalModel
            terminal_data = problem.terminalData
            x_T = solver.xs[-1]
            terminal_model.calc(terminal_data, x_T)
            terminal_cost = terminal_data.cost

            J_running = float(np.sum(running_costs))
            J_total_check = J_running + terminal_cost

            costs_available = True

        except Exception as e:
            print(f"Could not extract costs: {e}")
            costs_available = False
            J_running = 0.0
            terminal_cost = 0.0
            J_total_check = 0.0

        # Convert to numpy
        q_traj = np.array([x[:nq] for x in xs])
        u_traj = np.array([u for u in us])

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # 1. Joint positions
        axes[0].set_title("Joint Positions")
        for i in range(nq):
            axes[0].plot(q_traj[:, i], label=f"q{i + 1}")
        axes[0].set_xlabel("Time step")
        axes[0].set_ylabel("Position (rad)")
        axes[0].legend()
        axes[0].grid(True)

        # 2. Control effort
        axes[1].set_title("Control Effort")
        for i in range(u_traj.shape[1]):
            axes[1].plot(u_traj[:, i], label=f"u{i + 1}")
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Control (N·m)")
        axes[1].legend()
        axes[1].grid(True)

        # 3. Cost evolution over horizon
        if costs_available:
            # Combine running + terminal costs
            all_costs = running_costs + [terminal_cost]
            timesteps = np.arange(len(all_costs))

            axes[2].set_title(
                f"Cost per Timestep (Converged: {converged}, Total: {J_total_check:.2f})"
            )
            axes[2].plot(
                timesteps, all_costs, "o-", color="steelblue", linewidth=2, markersize=6
            )

            # Highlight terminal cost
            axes[2].axvline(
                len(running_costs) - 0.5,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Terminal boundary",
            )
            axes[2].plot(
                len(running_costs),
                terminal_cost,
                "o",
                color="orange",
                markersize=10,
                label=f"Terminal: {terminal_cost:.2f}",
            )

            axes[2].set_xlabel("Timestep")
            axes[2].set_ylabel("Cost")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            # Add text annotation for running sum
            axes[2].text(
                0.02,
                0.98,
                f"Running sum: {J_running:.2f}",
                transform=axes[2].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            axes[2].set_title("Cost per Timestep (No data)")
            axes[2].text(0.5, 0.5, "Could not extract costs", ha="center", va="center")

        plt.tight_layout()

        # Save
        filepath = os.path.join(self.log_dir, f"ocp_results_{timestamp}.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        return filepath
