import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox

class ODESolver:
    """Solver for ordinary differential equations with constant coefficients."""
    
    def __init__(self, m=1.0, c=0.5, k=1.0, forcing_func=None):
        self.m = m
        self.c = c
        self.k = k
        self.forcing_func = forcing_func if forcing_func else lambda t: 0
        self.omega_n = np.sqrt(k/m)
        self.zeta = c / (2 * np.sqrt(m*k))
        
    def system(self, state, t):
        y, dydt = state
        d2ydt2 = (self.forcing_func(t) - self.c*dydt - self.k*y) / self.m
        return [dydt, d2ydt2]
    
    def solve(self, y0, v0, t_span, num_points=1000):
        t = np.linspace(t_span[0], t_span[1], num_points)
        initial_state = [y0, v0]
        solution = odeint(self.system, initial_state, t)
        y = solution[:, 0]
        dydt = solution[:, 1]
        return t, y, dydt
    
    def get_damping_type(self):
        if self.zeta < 1:
            return f"Underdamped (ζ = {self.zeta:.3f})"
        elif abs(self.zeta - 1) < 0.001:
            return f"Critically damped (ζ = {self.zeta:.3f})"
        else:
            return f"Overdamped (ζ = {self.zeta:.3f})"


class ODEGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ODE Solver - Harmonic Oscillations")
        self.root.geometry("1400x800")
        
        # Create main frames
        self.control_frame = ttk.Frame(root, padding="10")
        self.control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.plot_frame = ttk.Frame(root, padding="10")
        self.plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        
        self.setup_controls()
        self.setup_plots()
        
        # Initial solve
        self.solve_and_plot()
    
    def setup_controls(self):
        # Title
        title = ttk.Label(self.control_frame, text="ODE Solver Control Panel", 
                         font=('Arial', 14, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Equation display
        eq_label = ttk.Label(self.control_frame, text="m·y'' + c·y' + k·y = F(t)", 
                            font=('Arial', 11))
        eq_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # System parameters
        ttk.Label(self.control_frame, text="System Parameters", 
                 font=('Arial', 11, 'bold')).grid(row=2, column=0, columnspan=2, pady=(15,5))
        
        # Mass
        ttk.Label(self.control_frame, text="Mass (m):").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.mass_var = tk.DoubleVar(value=1.0)
        self.mass_slider = ttk.Scale(self.control_frame, from_=0.1, to=5.0, 
                                     variable=self.mass_var, orient=tk.HORIZONTAL, length=200)
        self.mass_slider.grid(row=3, column=1, pady=3)
        self.mass_label = ttk.Label(self.control_frame, text="1.00")
        self.mass_label.grid(row=3, column=2, padx=5)
        
        # Damping
        ttk.Label(self.control_frame, text="Damping (c):").grid(row=4, column=0, sticky=tk.W, pady=3)
        self.damping_var = tk.DoubleVar(value=0.5)
        self.damping_slider = ttk.Scale(self.control_frame, from_=0.0, to=10.0, 
                                       variable=self.damping_var, orient=tk.HORIZONTAL, length=200)
        self.damping_slider.grid(row=4, column=1, pady=3)
        self.damping_label = ttk.Label(self.control_frame, text="0.50")
        self.damping_label.grid(row=4, column=2, padx=5)
        
        # Stiffness
        ttk.Label(self.control_frame, text="Stiffness (k):").grid(row=5, column=0, sticky=tk.W, pady=3)
        self.stiffness_var = tk.DoubleVar(value=4.0)
        self.stiffness_slider = ttk.Scale(self.control_frame, from_=0.1, to=10.0, 
                                         variable=self.stiffness_var, orient=tk.HORIZONTAL, length=200)
        self.stiffness_slider.grid(row=5, column=1, pady=3)
        self.stiffness_label = ttk.Label(self.control_frame, text="4.00")
        self.stiffness_label.grid(row=5, column=2, padx=5)
        
        # Initial conditions
        ttk.Label(self.control_frame, text="Initial Conditions", 
                 font=('Arial', 11, 'bold')).grid(row=6, column=0, columnspan=2, pady=(15,5))
        
        # Initial position
        ttk.Label(self.control_frame, text="Position y(0):").grid(row=7, column=0, sticky=tk.W, pady=3)
        self.y0_var = tk.DoubleVar(value=1.0)
        self.y0_slider = ttk.Scale(self.control_frame, from_=-5.0, to=5.0, 
                                   variable=self.y0_var, orient=tk.HORIZONTAL, length=200)
        self.y0_slider.grid(row=7, column=1, pady=3)
        self.y0_label = ttk.Label(self.control_frame, text="1.00")
        self.y0_label.grid(row=7, column=2, padx=5)
        
        # Initial velocity
        ttk.Label(self.control_frame, text="Velocity y'(0):").grid(row=8, column=0, sticky=tk.W, pady=3)
        self.v0_var = tk.DoubleVar(value=0.0)
        self.v0_slider = ttk.Scale(self.control_frame, from_=-5.0, to=5.0, 
                                   variable=self.v0_var, orient=tk.HORIZONTAL, length=200)
        self.v0_slider.grid(row=8, column=1, pady=3)
        self.v0_label = ttk.Label(self.control_frame, text="0.00")
        self.v0_label.grid(row=8, column=2, padx=5)
        
        # Time span
        ttk.Label(self.control_frame, text="Time Span (T):").grid(row=9, column=0, sticky=tk.W, pady=3)
        self.time_var = tk.DoubleVar(value=20.0)
        self.time_slider = ttk.Scale(self.control_frame, from_=5.0, to=100.0, 
                                     variable=self.time_var, orient=tk.HORIZONTAL, length=200)
        self.time_slider.grid(row=9, column=1, pady=3)
        self.time_label = ttk.Label(self.control_frame, text="20.0")
        self.time_label.grid(row=9, column=2, padx=5)
        
        # Forcing function
        ttk.Label(self.control_frame, text="Forcing Function", 
                 font=('Arial', 11, 'bold')).grid(row=10, column=0, columnspan=2, pady=(15,5))
        
        # Forcing type
        ttk.Label(self.control_frame, text="Type:").grid(row=11, column=0, sticky=tk.W, pady=3)
        self.forcing_type = tk.StringVar(value="None")
        forcing_combo = ttk.Combobox(self.control_frame, textvariable=self.forcing_type, 
                                     values=["None", "Sine", "Cosine", "Square", "Triangle"],
                                     state="readonly", width=18)
        forcing_combo.grid(row=11, column=1, pady=3)
        
        # Forcing amplitude
        ttk.Label(self.control_frame, text="Amplitude (F₀):").grid(row=12, column=0, sticky=tk.W, pady=3)
        self.amp_var = tk.DoubleVar(value=2.0)
        self.amp_slider = ttk.Scale(self.control_frame, from_=0.0, to=10.0, 
                                    variable=self.amp_var, orient=tk.HORIZONTAL, length=200)
        self.amp_slider.grid(row=12, column=1, pady=3)
        self.amp_label = ttk.Label(self.control_frame, text="2.00")
        self.amp_label.grid(row=12, column=2, padx=5)
        
        # Forcing frequency
        ttk.Label(self.control_frame, text="Frequency (ω):").grid(row=13, column=0, sticky=tk.W, pady=3)
        self.freq_var = tk.DoubleVar(value=1.5)
        self.freq_slider = ttk.Scale(self.control_frame, from_=0.1, to=5.0, 
                                     variable=self.freq_var, orient=tk.HORIZONTAL, length=200)
        self.freq_slider.grid(row=13, column=1, pady=3)
        self.freq_label = ttk.Label(self.control_frame, text="1.50")
        self.freq_label.grid(row=13, column=2, padx=5)
        
        # System info
        ttk.Label(self.control_frame, text="System Information", 
                 font=('Arial', 11, 'bold')).grid(row=14, column=0, columnspan=2, pady=(15,5))
        
        self.info_text = tk.Text(self.control_frame, height=6, width=35, state='disabled')
        self.info_text.grid(row=15, column=0, columnspan=3, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.control_frame)
        button_frame.grid(row=16, column=0, columnspan=3, pady=15)
        
        ttk.Button(button_frame, text="Update Plot", command=self.solve_and_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset_values).pack(side=tk.LEFT, padx=5)
        
        # Auto-update checkbox
        self.auto_update = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="Auto-update on slider change", 
                       variable=self.auto_update).grid(row=17, column=0, columnspan=3, pady=5)
        
        # Bind slider updates
        self.mass_var.trace_add('write', self.update_labels)
        self.damping_var.trace_add('write', self.update_labels)
        self.stiffness_var.trace_add('write', self.update_labels)
        self.y0_var.trace_add('write', self.update_labels)
        self.v0_var.trace_add('write', self.update_labels)
        self.time_var.trace_add('write', self.update_labels)
        self.amp_var.trace_add('write', self.update_labels)
        self.freq_var.trace_add('write', self.update_labels)
    
    def setup_plots(self):
        # Create figure with subplots
        self.fig = Figure(figsize=(10, 8))
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def get_forcing_function(self):
        f_type = self.forcing_type.get()
        amp = self.amp_var.get()
        freq = self.freq_var.get()
        
        if f_type == "None":
            return lambda t: 0
        elif f_type == "Sine":
            return lambda t: amp * np.sin(freq * t)
        elif f_type == "Cosine":
            return lambda t: amp * np.cos(freq * t)
        elif f_type == "Square":
            return lambda t: amp * np.sign(np.sin(freq * t))
        elif f_type == "Triangle":
            return lambda t: amp * (2/np.pi) * np.arcsin(np.sin(freq * t))
    
    def solve_and_plot(self):
        try:
            # Get parameters
            m = self.mass_var.get()
            c = self.damping_var.get()
            k = self.stiffness_var.get()
            y0 = self.y0_var.get()
            v0 = self.v0_var.get()
            t_end = self.time_var.get()
            
            # Create solver
            forcing = self.get_forcing_function()
            solver = ODESolver(m=m, c=c, k=k, forcing_func=forcing)
            
            # Solve
            t, y, dydt = solver.solve(y0, v0, (0, t_end))
            
            # Clear plots
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            
            # Plot position
            self.ax1.plot(t, y, 'b-', linewidth=2)
            self.ax1.set_ylabel('Position y(t)', fontsize=10)
            self.ax1.set_title('Position vs Time', fontsize=11, fontweight='bold')
            self.ax1.grid(True, alpha=0.3)
            
            # Plot velocity
            self.ax2.plot(t, dydt, 'r-', linewidth=2)
            self.ax2.set_ylabel("Velocity y'(t)", fontsize=10)
            self.ax2.set_title('Velocity vs Time', fontsize=11, fontweight='bold')
            self.ax2.grid(True, alpha=0.3)
            
            # Plot phase portrait
            self.ax3.plot(y, dydt, 'g-', linewidth=2)
            self.ax3.plot(y[0], dydt[0], 'ko', markersize=8, label='Start')
            self.ax3.plot(y[-1], dydt[-1], 'ro', markersize=8, label='End')
            self.ax3.set_xlabel('Position y(t)', fontsize=10)
            self.ax3.set_ylabel("Velocity y'(t)", fontsize=10)
            self.ax3.set_title('Phase Portrait', fontsize=11, fontweight='bold')
            self.ax3.grid(True, alpha=0.3)
            self.ax3.legend()
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Update info
            info = f"Natural Frequency: ωₙ = {solver.omega_n:.3f} rad/s\n"
            info += f"Damping Ratio: ζ = {solver.zeta:.3f}\n"
            info += f"Damping Type: {solver.get_damping_type()}\n"
            if self.forcing_type.get() != "None":
                info += f"\nForcing: {self.forcing_type.get()}\n"
                info += f"Amplitude: {self.amp_var.get():.2f}\n"
                info += f"Frequency: {self.freq_var.get():.2f} rad/s"
            
            self.info_text.config(state='normal')
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
            self.info_text.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"Error solving ODE: {str(e)}")
    
    def update_labels(self, *args):
        self.mass_label.config(text=f"{self.mass_var.get():.2f}")
        self.damping_label.config(text=f"{self.damping_var.get():.2f}")
        self.stiffness_label.config(text=f"{self.stiffness_var.get():.2f}")
        self.y0_label.config(text=f"{self.y0_var.get():.2f}")
        self.v0_label.config(text=f"{self.v0_var.get():.2f}")
        self.time_label.config(text=f"{self.time_var.get():.1f}")
        self.amp_label.config(text=f"{self.amp_var.get():.2f}")
        self.freq_label.config(text=f"{self.freq_var.get():.2f}")
        
        if self.auto_update.get():
            self.solve_and_plot()
    
    def reset_values(self):
        self.mass_var.set(1.0)
        self.damping_var.set(0.5)
        self.stiffness_var.set(4.0)
        self.y0_var.set(1.0)
        self.v0_var.set(0.0)
        self.time_var.set(20.0)
        self.forcing_type.set("None")
        self.amp_var.set(2.0)
        self.freq_var.set(1.5)
        self.solve_and_plot()


if __name__ == "__main__":
    root = tk.Tk()
    app = ODEGUIApp(root)
    root.mainloop()
