# -*- coding: utf-8 -*-
"""
Desktop (Tkinter) UI for Lab3: Multivariate Linear Regression with Gradient
Descent variants (Batch / Stochastic / Mini-batch) plus Polynomial Regression.

Run with:
    python app.py
"""

import math
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

DEFAULT_SALARY_URL = (
    "https://raw.githubusercontent.com/ekaratnida/Applied-machine-learning/"
    "master/Week03-MLR/Position_Salaries.csv"
)

SCHEDULE_TEMPLATES = {
    "Constant": "def schedule(alpha, iter_count, max_iter):\n    return alpha",
    "Time-based decay": "def schedule(alpha, iter_count, max_iter):\n    # alpha / (1 + decay * iter_count)\n    decay = 0.01\n    return alpha / (1 + decay * iter_count)",
    "Step decay": "def schedule(alpha, iter_count, max_iter):\n    # Drop by 0.5 every 100 steps\n    drop = 0.5\n    epochs = 100\n    return alpha * (drop ** (iter_count // epochs))",
    "Exponential decay": "def schedule(alpha, iter_count, max_iter):\n    # alpha * k ^ iter_count\n    k = 0.995\n    return alpha * (k ** iter_count)",
    "Custom": "def schedule(alpha, iter_count, max_iter):\n    # TODO: write your own learning-rate schedule here.\n    # Signature: schedule(alpha, iter_count, max_iter) -> new learning rate\n    return alpha",
}

# --------------------------------------------------------------------------------------
# Core math helpers (adapted from lab3.py)
# --------------------------------------------------------------------------------------


def cost_function(theta, x, y, N):
    """Mean squared error cost."""
    y_hat = x.dot(theta)
    return (1 / N) * np.sum((y_hat - y) ** 2)


def _current_alpha(alpha, lr_schedule, iter_count, max_iter):
    if lr_schedule is not None:
        return lr_schedule(alpha, iter_count, max_iter)
    return alpha


def gradient_descent(alpha, x, y, ep=0.001, max_iter=1000, random_state=None, lr_schedule=None):
    """Batch Gradient Descent."""
    converged = False
    iter_count = 0
    N = x.shape[0]
    rng = np.random.default_rng(random_state)

    theta = rng.random((x.shape[1], 1))
    theta_history = [theta.flatten()]
    cost_history = []

    J = cost_function(theta, x, y, N)
    cost_history.append(J)

    while not converged:
        y_hat = x.dot(theta)
        diff = y_hat - y
        grad = x.T.dot(diff)

        cur_alpha = _current_alpha(alpha, lr_schedule, iter_count, max_iter)
        theta = theta - cur_alpha * (1 / N) * grad
        theta_history.append(theta.flatten())

        J2 = cost_function(theta, x, y, N)
        cost_history.append(J2)

        if abs(J - J2) <= ep:
            converged = True

        J = J2
        iter_count += 1

        if iter_count >= max_iter:
            converged = True

    return theta, np.array(theta_history), np.array(cost_history), iter_count, J


def sgd(alpha, x, y, ep=0.001, max_iter=1000, random_state=None, lr_schedule=None):
    """Stochastic Gradient Descent (one random sample per step)."""
    converged = False
    iter_count = 0
    N = x.shape[0]
    rng = np.random.default_rng(random_state)

    theta = rng.random((x.shape[1], 1))
    theta_history = [theta.flatten()]
    cost_history = []

    J = cost_function(theta, x, y, N)
    cost_history.append(J)

    shuffle = rng.permutation(N)
    xs = x[shuffle]
    ys = y[shuffle]

    r_index = 0
    while not converged:
        xr = xs[r_index].reshape(1, -1)
        y_hat = xr.dot(theta)

        diff = (y_hat - ys[r_index]).reshape(-1, 1)
        grad = xr.T.dot(diff)

        cur_alpha = _current_alpha(alpha, lr_schedule, iter_count, max_iter)
        theta = theta - cur_alpha * grad
        theta_history.append(theta.flatten())

        J2 = cost_function(theta, xs, ys, N)
        cost_history.append(J2)

        if abs(J - J2) <= ep:
            converged = True

        J = J2
        iter_count += 1

        if iter_count >= max_iter:
            converged = True

        r_index += 1
        if r_index >= N:
            r_index = 0
            shuffle = rng.permutation(N)
            xs = xs[shuffle]
            ys = ys[shuffle]

    return theta, np.array(theta_history), np.array(cost_history), iter_count, J


def mbgd(alpha, x, y, ep=0.001, max_iter=1000, batch_size=256, random_state=None, lr_schedule=None):
    """Mini-batch Gradient Descent."""
    converged = False
    iter_count = 0
    N = x.shape[0]
    rng = np.random.default_rng(random_state)

    theta = rng.random((x.shape[1], 1))
    theta_history = [theta.flatten()]
    cost_history = []

    J = cost_function(theta, x, y, N)
    cost_history.append(J)

    steps_per_epoch = max(1, N // batch_size)
    x_shuffled, y_shuffled = x, y

    while not converged:
        if iter_count % steps_per_epoch == 0:
            shuffle_indices = rng.permutation(N)
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]

        start_idx = (iter_count * batch_size) % N
        end_idx = min(start_idx + batch_size, N)

        x_batch = x_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        N_batch = x_batch.shape[0]

        y_hat_batch = x_batch.dot(theta)
        diff_batch = y_hat_batch - y_batch
        grad = x_batch.T.dot(diff_batch)

        cur_alpha = _current_alpha(alpha, lr_schedule, iter_count, max_iter)
        theta = theta - cur_alpha * (1 / N_batch) * grad
        theta_history.append(theta.flatten())

        J2 = cost_function(theta, x, y, N)
        cost_history.append(J2)

        if abs(J - J2) <= ep:
            converged = True

        J = J2
        iter_count += 1

        if iter_count >= max_iter:
            converged = True

    return theta, np.array(theta_history), np.array(cost_history), iter_count, J


def generate_data(n_samples, n_features, noise, random_state):
    x, y = make_regression(
        n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state
    )
    x_b = np.c_[np.ones((x.shape[0], 1)), x]
    y = y.reshape(-1, 1)
    return x_b, y


# --------------------------------------------------------------------------------------
# Tkinter application
# --------------------------------------------------------------------------------------


class LabeledEntry(ttk.Frame):
    """A small labeled entry widget with a default value and getter."""

    def __init__(self, parent, label, default, width=12, **kwargs):
        super().__init__(parent)
        ttk.Label(self, text=label).pack(side=tk.LEFT, padx=(0, 6))
        self.var = tk.StringVar(value=str(default))
        entry = ttk.Entry(self, textvariable=self.var, width=width, **kwargs)
        entry.pack(side=tk.LEFT)

    def get_float(self):
        return float(self.var.get())

    def get_int(self):
        return int(float(self.var.get()))


class GDTab(ttk.Frame):
    """Gradient Descent comparison tab: BGD / SGD / MBGD."""

    def __init__(self, parent):
        super().__init__(parent)
        self.result_queue = queue.Queue()
        self.worker_thread = None
        self.stop_event = threading.Event()
        self._build_ui()

    # ---- UI construction -------------------------------------------------
    def _build_ui(self):
        self.controls_canvas = tk.Canvas(self, width=330)
        controls_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.controls_canvas.yview)
        controls = ttk.Frame(self.controls_canvas, padding=(10, 10))
        self.controls_window_id = self.controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        self.controls_canvas.pack(side=tk.LEFT, fill=tk.Y)
        controls_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.controls_canvas.bind("<Configure>", self._on_canvas_configure)
        self.controls_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        plots = ttk.Frame(self)
        plots.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Dataset params ---
        ttk.Label(controls, text="1) Synthetic dataset", font=("", 10, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        self.n_samples = LabeledEntry(controls, "n_samples:", 2000)
        self.n_samples.pack(anchor="w", pady=2)
        self.n_features = LabeledEntry(controls, "n_features:", 10)
        self.n_features.pack(anchor="w", pady=2)
        self.noise = LabeledEntry(controls, "noise:", 2.0)
        self.noise.pack(anchor="w", pady=2)
        self.data_seed = LabeledEntry(controls, "data random_state:", 123)
        self.data_seed.pack(anchor="w", pady=2)

        ttk.Separator(controls).pack(fill=tk.X, pady=8)

        # --- GD hyperparameters ---
        ttk.Label(controls, text="2) Gradient Descent hyperparameters", font=("", 10, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        self.eta = LabeledEntry(controls, "learning rate (eta):", 0.01)
        self.eta.pack(anchor="w", pady=2)
        self.epsilon = LabeledEntry(controls, "tolerance (epsilon):", 0.001)
        self.epsilon.pack(anchor="w", pady=2)
        self.miter = LabeledEntry(controls, "max_iter:", 500)
        self.miter.pack(anchor="w", pady=2)
        self.batch_size = LabeledEntry(controls, "MBGD batch_size:", 256)
        self.batch_size.pack(anchor="w", pady=2)
        self.gd_seed = LabeledEntry(controls, "GD random_state:", 42)
        self.gd_seed.pack(anchor="w", pady=2)

        ttk.Separator(controls).pack(fill=tk.X, pady=8)

        # --- Methods ---
        ttk.Label(controls, text="3) Methods to run", font=("", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.run_bgd_var = tk.BooleanVar(value=True)
        self.run_sgd_var = tk.BooleanVar(value=True)
        self.run_mbgd_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Batch Gradient Descent (BGD)", variable=self.run_bgd_var).pack(anchor="w")
        ttk.Checkbutton(controls, text="Stochastic Gradient Descent (SGD)", variable=self.run_sgd_var).pack(anchor="w")
        ttk.Checkbutton(controls, text="Mini-batch Gradient Descent (MBGD)", variable=self.run_mbgd_var).pack(anchor="w")

        ttk.Separator(controls).pack(fill=tk.X, pady=8)

        # --- Learning rate schedule (optional custom code) ---
        ttk.Label(controls, text="4) Learning-rate schedule (optional)", font=("", 10, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        self.use_schedule_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls, text="Use a learning-rate schedule",
            variable=self.use_schedule_var, command=self._update_schedule_state,
        ).pack(anchor="w")

        scheme_row = ttk.Frame(controls)
        scheme_row.pack(anchor="w", pady=2, fill=tk.X)
        ttk.Label(scheme_row, text="Preset:").pack(side=tk.LEFT)
        self.schedule_scheme_var = tk.StringVar(value="Constant")
        self.schedule_combo = ttk.Combobox(
            scheme_row,
            textvariable=self.schedule_scheme_var,
            values=list(SCHEDULE_TEMPLATES.keys()),
            state="readonly",
            width=14,
        )
        self.schedule_combo.pack(side=tk.LEFT, padx=(4, 0))
        self.schedule_combo.bind("<<ComboboxSelected>>", self._on_scheme_selected)

        self.schedule_text = tk.Text(
            controls, width=40, height=6, font=("Consolas", 9), state="disabled"
        )
        self.schedule_text.pack(anchor="w", pady=(4, 2), fill=tk.X)
        ttk.Label(
            controls, text="Signature: schedule(alpha, iter_count, max_iter) -> new learning rate",
            foreground="#666666",
        ).pack(anchor="w")
        self._load_scheme("Constant")

        ttk.Separator(controls).pack(fill=tk.X, pady=8)

        button_row = ttk.Frame(controls)
        button_row.pack(anchor="w", pady=4, fill=tk.X)
        self.run_button = ttk.Button(button_row, text="Run Comparison", command=self._on_run_clicked)
        self.run_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.stop_button = ttk.Button(
            button_row, text="Stop Training", command=self._on_stop_clicked, state="disabled"
        )
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, padx=(4, 0))

        self.reset_button = ttk.Button(controls, text="Reset to Defaults", command=self._on_reset_clicked)
        self.reset_button.pack(anchor="w", pady=(0, 4), fill=tk.X)

        self.progress = ttk.Progressbar(controls, mode="indeterminate", length=220)
        self.progress.pack(anchor="w", pady=4, fill=tk.X)

        self.status_label = ttk.Label(controls, text="Idle.")
        self.status_label.pack(anchor="w", pady=(0, 8))

        # --- Summary table ---
        ttk.Label(controls, text="Summary", font=("", 10, "bold")).pack(anchor="w", pady=(4, 4))
        self.summary_tree = ttk.Treeview(
            controls, columns=("iters", "cost", "time"), show="tree headings", height=4
        )
        self.summary_tree.heading("#0", text="Method")
        self.summary_tree.heading("iters", text="Iterations")
        self.summary_tree.heading("cost", text="Final Cost (J)")
        self.summary_tree.heading("time", text="Time (s)")
        self.summary_tree.column("#0", width=110, anchor="w")
        self.summary_tree.column("iters", width=80, anchor="center")
        self.summary_tree.column("cost", width=110, anchor="center")
        self.summary_tree.column("time", width=80, anchor="center")
        self.summary_tree.pack(anchor="w", fill=tk.X)

        # --- Plot area (notebook with two figures) ---
        plot_notebook = ttk.Notebook(plots)
        plot_notebook.pack(fill=tk.BOTH, expand=True)

        cost_frame = ttk.Frame(plot_notebook)
        pca_frame = ttk.Frame(plot_notebook)
        plot_notebook.add(cost_frame, text="Cost History")
        plot_notebook.add(pca_frame, text="PCA Optimization Paths")

        self.fig_cost = Figure(figsize=(7, 5.5), dpi=100)
        self.ax_cost = self.fig_cost.add_subplot(111)
        self.canvas_cost = FigureCanvasTkAgg(self.fig_cost, master=cost_frame)
        self.canvas_cost.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas_cost, cost_frame).pack(fill=tk.X)

        self.fig_pca = Figure(figsize=(7, 5.5), dpi=100)
        self.ax_pca = self.fig_pca.add_subplot(111)
        self.canvas_pca = FigureCanvasTkAgg(self.fig_pca, master=pca_frame)
        self.canvas_pca.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas_pca, pca_frame).pack(fill=tk.X)

        self._draw_placeholder(self.ax_cost, "Cost History will appear here")
        self._draw_placeholder(self.ax_pca, "PCA Optimization Paths will appear here")

    @staticmethod
    def _draw_placeholder(ax, text):
        ax.clear()
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])

    def _on_canvas_configure(self, event):
        self.controls_canvas.itemconfigure(self.controls_window_id, width=event.width - 8)
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _on_mousewheel(self, event):
        self.controls_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ---- Learning-rate schedule --------------------------------------
    def _load_scheme(self, scheme_name):
        self.schedule_text.config(state="normal")
        self.schedule_text.delete("1.0", tk.END)
        self.schedule_text.insert("1.0", SCHEDULE_TEMPLATES.get(scheme_name, SCHEDULE_TEMPLATES["Constant"]))
        self.schedule_text.config(state="disabled")

    def _on_scheme_selected(self, _event=None):
        self._load_scheme(self.schedule_scheme_var.get())

    def _update_schedule_state(self):
        enabled = self.use_schedule_var.get()
        state = "normal" if enabled else "disabled"
        self.schedule_combo.state(["!disabled"] if enabled else ["disabled"])
        self.schedule_text.config(state=state)

    def _get_schedule_func(self):
        """Return a compiled schedule function, or None if scheduling is disabled."""
        if not self.use_schedule_var.get():
            return None
        code = self.schedule_text.get("1.0", tk.END)
        try:
            namespace = {}
            exec(compile(code, "<schedule>", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid schedule code: {exc}") from exc
        schedule = namespace.get("schedule")
        if schedule is None:
            raise ValueError("The schedule code must define a function named 'schedule'.")
        return schedule

    # ---- Event handling ----------------------------------------------
    def _on_run_clicked(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress.")
            return

        if not (self.run_bgd_var.get() or self.run_sgd_var.get() or self.run_mbgd_var.get()):
            messagebox.showerror("No method selected", "Select at least one method to run.")
            return

        try:
            schedule_func = self._get_schedule_func()
        except ValueError as exc:
            messagebox.showerror("Invalid schedule code", str(exc))
            return

        try:
            params = dict(
                n_samples=self.n_samples.get_int(),
                n_features=self.n_features.get_int(),
                noise=self.noise.get_float(),
                data_seed=self.data_seed.get_int(),
                eta=self.eta.get_float(),
                epsilon=self.epsilon.get_float(),
                miter=self.miter.get_int(),
                batch_size=self.batch_size.get_int(),
                gd_seed=self.gd_seed.get_int(),
                run_bgd=self.run_bgd_var.get(),
                run_sgd=self.run_sgd_var.get(),
                run_mbgd=self.run_mbgd_var.get(),
                schedule=schedule_func,
            )
        except ValueError:
            messagebox.showerror("Invalid input", "Please check that all parameters are valid numbers.")
            return

        self.stop_event.clear()
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress.start(10)
        self.status_label.config(text="Running...")

        self.worker_thread = threading.Thread(target=self._run_worker, args=(params,), daemon=True)
        self.worker_thread.start()
        self.after(100, self._poll_worker)

    def _on_stop_clicked(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Stopping...")
            self.stop_event.set()

    def _on_reset_clicked(self):
        defaults = dict(
            n_samples="2000", n_features="10", noise="2.0", data_seed="123",
            eta="0.01", epsilon="0.001", miter="500", batch_size="256", gd_seed="42",
        )
        self.n_samples.var.set(defaults["n_samples"])
        self.n_features.var.set(defaults["n_features"])
        self.noise.var.set(defaults["noise"])
        self.data_seed.var.set(defaults["data_seed"])
        self.eta.var.set(defaults["eta"])
        self.epsilon.var.set(defaults["epsilon"])
        self.miter.var.set(defaults["miter"])
        self.batch_size.var.set(defaults["batch_size"])
        self.gd_seed.var.set(defaults["gd_seed"])

        self.run_bgd_var.set(True)
        self.run_sgd_var.set(True)
        self.run_mbgd_var.set(True)

        self.use_schedule_var.set(False)
        self.schedule_scheme_var.set("Constant")
        self._load_scheme("Constant")
        self._update_schedule_state()

        for row in self.summary_tree.get_children():
            self.summary_tree.delete(row)
        self.status_label.config(text="Idle.")

    def _run_worker(self, params):
        try:
            x_b, y = generate_data(
                params["n_samples"], params["n_features"], params["noise"], params["data_seed"]
            )

            results = {}
            schedule = params["schedule"]

            def stopped():
                return self.stop_event.is_set()

            if params["run_bgd"] and not stopped():
                t0 = time.perf_counter()
                theta, th_hist, c_hist, iters, final_j = gradient_descent(
                    params["eta"], x_b, y, ep=params["epsilon"], max_iter=params["miter"],
                    random_state=params["gd_seed"], lr_schedule=schedule,
                )
                if not stopped():
                    elapsed = time.perf_counter() - t0
                    results["BGD"] = dict(theta_history=th_hist, cost_history=c_hist,
                                           iters=iters, final_j=final_j, elapsed=elapsed, color="blue")

            if params["run_sgd"] and not stopped():
                t0 = time.perf_counter()
                theta, th_hist, c_hist, iters, final_j = sgd(
                    params["eta"], x_b, y, ep=params["epsilon"], max_iter=params["miter"],
                    random_state=params["gd_seed"], lr_schedule=schedule,
                )
                if not stopped():
                    elapsed = time.perf_counter() - t0
                    results["SGD"] = dict(theta_history=th_hist, cost_history=c_hist,
                                           iters=iters, final_j=final_j, elapsed=elapsed, color="red")

            if params["run_mbgd"] and not stopped():
                t0 = time.perf_counter()
                theta, th_hist, c_hist, iters, final_j = mbgd(
                    params["eta"], x_b, y, ep=params["epsilon"], max_iter=params["miter"],
                    batch_size=params["batch_size"], random_state=params["gd_seed"],
                    lr_schedule=schedule,
                )
                if not stopped():
                    elapsed = time.perf_counter() - t0
                    results["MBGD"] = dict(theta_history=th_hist, cost_history=c_hist,
                                            iters=iters, final_j=final_j, elapsed=elapsed, color="purple")

            if stopped():
                self.result_queue.put(("stopped", results))
            else:
                self.result_queue.put(("ok", results))
        except Exception as exc:  # noqa: BLE001
            self.result_queue.put(("error", str(exc)))

    def _poll_worker(self):
        try:
            status, payload = self.result_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_worker)
            return

        self.progress.stop()
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        if status == "error":
            self.status_label.config(text="Error.")
            messagebox.showerror("Computation error", payload)
            return

        if status == "stopped":
            self.status_label.config(text="Stopped.")
            if payload:
                self._render_results(payload)
            return

        self.status_label.config(text="Done.")
        self._render_results(payload)

    # ---- Rendering ------------------------------------------------------
    def _render_results(self, results):
        for row in self.summary_tree.get_children():
            self.summary_tree.delete(row)
        for name, r in results.items():
            self.summary_tree.insert(
                "", tk.END, text=name,
                values=(r["iters"], f"{r['final_j']:.4f}", f"{r['elapsed']:.3f}"),
            )

        # Cost history plot
        self.ax_cost.clear()
        for name, r in results.items():
            self.ax_cost.plot(range(len(r["cost_history"])), r["cost_history"], color=r["color"], label=name)
        self.ax_cost.set_xlabel("Iterations")
        self.ax_cost.set_ylabel("Cost (J)")
        self.ax_cost.set_title("Cost History Comparison")
        self.ax_cost.legend()
        self.ax_cost.grid(True)
        self.fig_cost.tight_layout()
        self.canvas_cost.draw()

        # PCA optimization path plot
        self.ax_pca.clear()
        if len(results) >= 2:
            all_paths = np.vstack([r["theta_history"] for r in results.values()])
            pca = PCA(n_components=2)
            pca.fit(all_paths)

            markers = {"BGD": "o", "SGD": "x", "MBGD": "s"}
            first_point = None
            for name, r in results.items():
                path_2d = pca.transform(r["theta_history"])
                self.ax_pca.plot(
                    path_2d[:, 0], path_2d[:, 1],
                    color=r["color"], marker=markers.get(name, "."),
                    linestyle="-", markersize=3, alpha=0.6, label=name,
                )
                if first_point is None:
                    first_point = path_2d[0]
            if first_point is not None:
                self.ax_pca.plot(first_point[0], first_point[1], "ko", markersize=10, label="Start")
            self.ax_pca.set_title("Optimization Paths in 2D PCA Space")
            self.ax_pca.set_xlabel("Principal Component 1")
            self.ax_pca.set_ylabel("Principal Component 2")
            self.ax_pca.legend()
            self.ax_pca.grid(True)
        else:
            self._draw_placeholder(self.ax_pca, "Run at least two methods to see PCA comparison")
        self.fig_pca.tight_layout()
        self.canvas_pca.draw()


class PolyTab(ttk.Frame):
    """Polynomial regression tab."""

    def __init__(self, parent):
        super().__init__(parent)
        self.df = None
        self.result_queue = queue.Queue()
        self.worker_thread = None
        self._build_ui()

    def _build_ui(self):
        controls = ttk.Frame(self)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        plots = ttk.Frame(self)
        plots.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(controls, text="Dataset source", font=("", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.source_var = tk.StringVar(value="default")
        ttk.Radiobutton(
            controls, text="Default (Position_Salaries.csv)", value="default", variable=self.source_var
        ).pack(anchor="w")
        ttk.Radiobutton(controls, text="Local CSV file...", value="file", variable=self.source_var).pack(anchor="w")

        self.file_path_var = tk.StringVar(value="")
        file_row = ttk.Frame(controls)
        file_row.pack(anchor="w", pady=(2, 8), fill=tk.X)
        self.file_entry = ttk.Entry(file_row, textvariable=self.file_path_var, width=22, state="disabled")
        self.file_entry.pack(side=tk.LEFT, padx=(0, 4))
        self.browse_button = ttk.Button(file_row, text="Browse...", command=self._browse_file, state="disabled")
        self.browse_button.pack(side=tk.LEFT)

        self.source_var.trace_add("write", self._on_source_changed)

        ttk.Separator(controls).pack(fill=tk.X, pady=8)

        ttk.Label(controls, text="Model parameters", font=("", 10, "bold")).pack(anchor="w", pady=(0, 4))
        degree_row = ttk.Frame(controls)
        degree_row.pack(anchor="w", pady=2, fill=tk.X)
        ttk.Label(degree_row, text="Polynomial degree:").pack(side=tk.LEFT)
        self.degree_var = tk.IntVar(value=3)
        ttk.Spinbox(degree_row, from_=1, to=10, textvariable=self.degree_var, width=5).pack(side=tk.LEFT, padx=(4, 0))

        self.predict_level = LabeledEntry(controls, "Predict at level:", 5.5)
        self.predict_level.pack(anchor="w", pady=(6, 2))

        ttk.Separator(controls).pack(fill=tk.X, pady=8)

        self.run_button = ttk.Button(controls, text="Load Data & Fit Models", command=self._on_run_clicked)
        self.run_button.pack(anchor="w", pady=4, fill=tk.X)

        self.progress = ttk.Progressbar(controls, mode="indeterminate", length=220)
        self.progress.pack(anchor="w", pady=4, fill=tk.X)

        self.status_label = ttk.Label(controls, text="Idle.")
        self.status_label.pack(anchor="w", pady=(0, 8))

        ttk.Separator(controls).pack(fill=tk.X, pady=8)
        ttk.Label(controls, text="Predictions", font=("", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.lin_pred_label = ttk.Label(controls, text="Linear: -")
        self.lin_pred_label.pack(anchor="w")
        self.poly_pred_label = ttk.Label(controls, text="Polynomial: -")
        self.poly_pred_label.pack(anchor="w")

        # Plot area
        plot_notebook = ttk.Notebook(plots)
        plot_notebook.pack(fill=tk.BOTH, expand=True)

        data_frame = ttk.Frame(plot_notebook)
        lin_frame = ttk.Frame(plot_notebook)
        poly_frame = ttk.Frame(plot_notebook)
        plot_notebook.add(data_frame, text="Data Preview")
        plot_notebook.add(lin_frame, text="Linear Fit")
        plot_notebook.add(poly_frame, text="Polynomial Fit")

        # Data preview as a table
        self.data_tree = ttk.Treeview(data_frame, show="headings")
        self.data_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.fig_lin = Figure(figsize=(6, 5), dpi=100)
        self.ax_lin = self.fig_lin.add_subplot(111)
        self.canvas_lin = FigureCanvasTkAgg(self.fig_lin, master=lin_frame)
        self.canvas_lin.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas_lin, lin_frame).pack(fill=tk.X)

        self.fig_poly = Figure(figsize=(6, 5), dpi=100)
        self.ax_poly = self.fig_poly.add_subplot(111)
        self.canvas_poly = FigureCanvasTkAgg(self.fig_poly, master=poly_frame)
        self.canvas_poly.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas_poly, poly_frame).pack(fill=tk.X)

        self._draw_placeholder(self.ax_lin, "Linear fit will appear here")
        self._draw_placeholder(self.ax_poly, "Polynomial fit will appear here")

    @staticmethod
    def _draw_placeholder(ax, text):
        ax.clear()
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])

    def _on_source_changed(self, *_args):
        if self.source_var.get() == "file":
            self.file_entry.config(state="normal")
            self.browse_button.config(state="normal")
        else:
            self.file_entry.config(state="disabled")
            self.browse_button.config(state="disabled")

    def _browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.file_path_var.set(path)

    def _on_run_clicked(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress.")
            return

        if self.source_var.get() == "file" and not self.file_path_var.get():
            messagebox.showerror("No file selected", "Please browse and select a CSV file.")
            return

        try:
            degree = int(self.degree_var.get())
            predict_level = self.predict_level.get_float()
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid input", "Please check degree and predict level values.")
            return

        source = self.file_path_var.get() if self.source_var.get() == "file" else DEFAULT_SALARY_URL

        self.run_button.config(state=tk.DISABLED)
        self.progress.start(10)
        self.status_label.config(text="Loading & fitting...")

        self.worker_thread = threading.Thread(
            target=self._run_worker, args=(source, degree, predict_level), daemon=True
        )
        self.worker_thread.start()
        self.after(100, self._poll_worker)

    def _run_worker(self, source, degree, predict_level):
        try:
            df = pd.read_csv(source)
            X = df.iloc[:, 1:2].values.astype(float)
            y = df.iloc[:, 2].values.astype(float)

            lin_reg = LinearRegression()
            lin_reg.fit(X, y)

            poly_reg = PolynomialFeatures(degree=degree)
            X_poly = poly_reg.fit_transform(X)
            pol_reg = LinearRegression()
            pol_reg.fit(X_poly, y)

            X_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

            lin_pred = lin_reg.predict([[predict_level]])[0]
            poly_pred = pol_reg.predict(poly_reg.transform([[predict_level]]))[0]

            payload = dict(
                df=df, X=X, y=y, X_grid=X_grid,
                lin_reg=lin_reg, pol_reg=pol_reg, poly_reg=poly_reg,
                degree=degree, predict_level=predict_level,
                lin_pred=lin_pred, poly_pred=poly_pred,
            )
            self.result_queue.put(("ok", payload))
        except Exception as exc:  # noqa: BLE001
            self.result_queue.put(("error", str(exc)))

    def _poll_worker(self):
        try:
            status, payload = self.result_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_worker)
            return

        self.progress.stop()
        self.run_button.config(state=tk.NORMAL)

        if status == "error":
            self.status_label.config(text="Error.")
            messagebox.showerror("Computation error", payload)
            return

        self.status_label.config(text="Done.")
        self._render_results(payload)

    def _render_results(self, payload):
        df = payload["df"]
        X = payload["X"]
        y = payload["y"]
        X_grid = payload["X_grid"]
        lin_reg = payload["lin_reg"]
        pol_reg = payload["pol_reg"]
        poly_reg = payload["poly_reg"]
        degree = payload["degree"]
        predict_level = payload["predict_level"]

        # Data preview table
        self.data_tree.delete(*self.data_tree.get_children())
        self.data_tree["columns"] = list(df.columns)
        for col in df.columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100, anchor="center")
        for _, row in df.iterrows():
            self.data_tree.insert("", tk.END, values=list(row))

        # Linear fit plot
        self.ax_lin.clear()
        self.ax_lin.scatter(X, y, color="red", label="Actual")
        self.ax_lin.plot(X_grid, lin_reg.predict(X_grid), color="blue", label="Linear fit")
        self.ax_lin.set_title("Truth or Bluff (Linear Regression)")
        self.ax_lin.set_xlabel("Position level")
        self.ax_lin.set_ylabel("Salary")
        self.ax_lin.legend()
        self.fig_lin.tight_layout()
        self.canvas_lin.draw()

        # Polynomial fit plot
        self.ax_poly.clear()
        self.ax_poly.scatter(X, y, color="red", label="Actual")
        self.ax_poly.plot(X_grid, lin_reg.predict(X_grid), color="green", linestyle="--", label="Linear fit")
        self.ax_poly.plot(
            X_grid, pol_reg.predict(poly_reg.transform(X_grid)), color="blue",
            label=f"Polynomial (deg={degree})"
        )
        self.ax_poly.set_title("Truth or Bluff (Polynomial Regression)")
        self.ax_poly.set_xlabel("Position level")
        self.ax_poly.set_ylabel("Salary")
        self.ax_poly.legend()
        self.fig_poly.tight_layout()
        self.canvas_poly.draw()

        # Predictions
        self.lin_pred_label.config(text=f"Linear @ {predict_level}: {payload['lin_pred']:.2f}")
        self.poly_pred_label.config(text=f"Polynomial (deg {degree}) @ {predict_level}: {payload['poly_pred']:.2f}")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lab3 - Regression & Gradient Descent Explorer")
        self.geometry("1200x750")

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        gd_tab = GDTab(notebook)
        poly_tab = PolyTab(notebook)

        notebook.add(gd_tab, text="Gradient Descent Comparison")
        notebook.add(poly_tab, text="Polynomial Regression")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
