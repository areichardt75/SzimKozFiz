import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox

class CoupledOscillatorSolver:
    """Solver for coupled mass-spring system with 4 masses in a row."""
    
    def __init__(self, masses, spring_constants, damping_coeffs, forcing_funcs=None):
        """
        Initialize solver for coupled oscillators.
        
        Parameters:
        -----------
        masses : list of 4 floats
            Masses [m1, m2, m3, m4]
        spring_constants : list of 5 floats
            Spring constants [k0, k1, k2, k3, k4]
            k0: left wall to m1
            k1: m1 to m2
            k2: m2 to m3
            k3: m3 to m4
            k4: m4 to right wall
        damping_coeffs : list of 4 floats
            Damping coefficients [c1, c2, c3, c4]
        forcing_funcs : list of 4 callables or None
            External forces on each mass
        """
        self.masses = np.array(masses)
        self.k = np.array(spring_constants)
        self.c = np.array(damping_coeffs)
        
        if forcing_funcs is None:
            self.F = [lambda t: 0 for _ in range(4)]
        else:
            self.F = forcing_funcs
    
    def system(self, state, t):
        """
        System of ODEs for 4 coupled masses.
        state = [x1, x2, x3, x4, v1, v2, v3, v4]
        """
        # Positions and velocities
        x = state[:4]
        v = state[4:]
        
        # Calculate accelerations using Newton's 2nd law
        # For mass 1: F = -k0*x1 - k1*(x1-x2) - c1*v1 + F1(t)
        a1 = (-self.k[0]*x[0] - self.k[1]*(x[0]-x[1]) - self.c[0]*v[0] + self.F[0](t)) / self.masses[0]
        
        # For mass 2: F = -k1*(x2-x1) - k2*(x2-x3) - c2*v2 + F2(t)
        a2 = (-self.k[1]*(x[1]-x[0]) - self.k[2]*(x[1]-x[2]) - self.c[1]*v[1] + self.F[1](t)) / self.masses[1]
        
        # For mass 3: F = -k2*(x3-x2) - k3*(x3-x4) - c3*v3 + F3(t)
        a3 = (-self.k[2]*(x[2]-x[1]) - self.k[3]*(x[2]-x[3]) - self.c[2]*v[2] + self.F[2](t)) / self.masses[2]
        
        # For mass 4: F = -k3*(x4-x3) - k4*x4 - c4*v4 + F4(t)
        a4 = (-self.k[3]*(x[3]-x[2]) - self.k[4]*x[3] - self.c[3]*v[3] + self.F[3](t)) / self.masses[3]
        
        # Return derivatives [v1, v2, v3, v4, a1, a2, a3, a4]
        return [v[0], v[1], v[2], v[3], a1, a2, a3, a4]
    
    def solve(self, initial_positions, initial_velocities, t_span, num_points=1000):
        """
        Solve the coupled system.
        
        Parameters:
        -----------
        initial_positions : list of 4 floats
            Initial positions [x1(0), x2(0), x3(0), x4(0)]
        initial_velocities : list of 4 floats
            Initial velocities [v1(0), v2(0), v3(0), v4(0)]
        t_span : tuple
            Time span (t_start, t_end)
        num_points : int
            Number of time points
            
        Returns:
        --------
        t : array
            Time values
        positions : array (num_points, 4)
            Position values for all masses
        velocities : array (num_points, 4)
            Velocity values for all masses
        """
        t = np.linspace(t_span[0], t_span[1], num_points)
        initial_state = list(initial_positions) + list(initial_velocities)
        solution = odeint(self.system, initial_state, t)
        
        positions = solution[:, :4]
        velocities = solution[:, 4:]
        
        return t, positions, velocities
    
    def calculate_normal_modes(self):
        """Calculate normal mode frequencies (simplified for equal masses and springs)."""
        # This is a simplified calculation assuming equal masses and internal springs
        if len(set(self.masses)) == 1 and len(set(self.k[1:4])) == 1:
            m = self.masses[0]
            k = self.k[1]
            omega0 = np.sqrt(k/m)
            # Normal mode frequencies for 4 masses with fixed ends
            modes = [omega0 * np.sqrt(2 - 2*np.cos(n*np.pi/5)) for n in range(1, 5)]
            return modes
        return None


class CoupledOscillatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Coupled Oscillator System - 4 Masses Connected by Springs")
        self.root.geometry("1600x900")
        
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
        row = 0
        
        # Title
        title = ttk.Label(self.control_frame, text="4 Coupled Masses - Spring System", 
                         font=('Arial', 14, 'bold'))
        title.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        # System description
        desc = ttk.Label(self.control_frame, 
                        text="Wall—k₀—[m₁]—k₁—[m₂]—k₂—[m₃]—k₃—[m₄]—k₄—Wall",
                        font=('Courier', 10))
        desc.grid(row=row, column=0, columnspan=3, pady=5)
        row += 1
        
        # Create notebook for organized parameters
        notebook = ttk.Notebook(self.control_frame)
        notebook.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Masses tab
        masses_frame = ttk.Frame(notebook, padding="10")
        notebook.add(masses_frame, text="Masses")
        
        self.mass_vars = []
        self.mass_labels = []
        for i in range(4):
            ttk.Label(masses_frame, text=f"Mass m{i+1}:").grid(row=i, column=0, sticky=tk.W, pady=3)
            var = tk.DoubleVar(value=1.0)
            slider = ttk.Scale(masses_frame, from_=0.1, to=5.0, variable=var, 
                             orient=tk.HORIZONTAL, length=150)
            slider.grid(row=i, column=1, pady=3, padx=5)
            label = ttk.Label(masses_frame, text="1.00")
            label.grid(row=i, column=2, padx=5)
            self.mass_vars.append(var)
            self.mass_labels.append(label)
            var.trace_add('write', self.update_labels)
        
        # Springs tab
        springs_frame = ttk.Frame(notebook, padding="10")
        notebook.add(springs_frame, text="Springs")
        
        self.spring_vars = []
        self.spring_labels = []
        spring_names = ["k₀ (Wall-m₁)", "k₁ (m₁-m₂)", "k₂ (m₂-m₃)", 
                       "k₃ (m₃-m₄)", "k₄ (m₄-Wall)"]
        for i in range(5):
            ttk.Label(springs_frame, text=spring_names[i] + ":").grid(row=i, column=0, sticky=tk.W, pady=3)
            var = tk.DoubleVar(value=2.0)
            slider = ttk.Scale(springs_frame, from_=0.1, to=10.0, variable=var, 
                             orient=tk.HORIZONTAL, length=150)
            slider.grid(row=i, column=1, pady=3, padx=5)
            label = ttk.Label(springs_frame, text="2.00")
            label.grid(row=i, column=2, padx=5)
            self.spring_vars.append(var)
            self.spring_labels.append(label)
            var.trace_add('write', self.update_labels)
        
        # Damping tab
        damping_frame = ttk.Frame(notebook, padding="10")
        notebook.add(damping_frame, text="Damping")
        
        self.damping_vars = []
        self.damping_labels = []
        for i in range(4):
            ttk.Label(damping_frame, text=f"Damping c{i+1}:").grid(row=i, column=0, sticky=tk.W, pady=3)
            var = tk.DoubleVar(value=0.1)
            slider = ttk.Scale(damping_frame, from_=0.0, to=2.0, variable=var, 
                             orient=tk.HORIZONTAL, length=150)
            slider.grid(row=i, column=1, pady=3, padx=5)
            label = ttk.Label(damping_frame, text="0.10")
            label.grid(row=i, column=2, padx=5)
            self.damping_vars.append(var)
            self.damping_labels.append(label)
            var.trace_add('write', self.update_labels)
        
        # Initial Conditions tab
        ic_frame = ttk.Frame(notebook, padding="10")
        notebook.add(ic_frame, text="Initial Conditions")
        
        ttk.Label(ic_frame, text="Initial Positions", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=3, pady=5)
        
        self.pos_vars = []
        self.pos_labels = []
        for i in range(4):
            ttk.Label(ic_frame, text=f"x{i+1}(0):").grid(row=i+1, column=0, sticky=tk.W, pady=3)
            var = tk.DoubleVar(value=0.0 if i != 0 else 1.0)
            slider = ttk.Scale(ic_frame, from_=-3.0, to=3.0, variable=var, 
                             orient=tk.HORIZONTAL, length=150)
            slider.grid(row=i+1, column=1, pady=3, padx=5)
            label = ttk.Label(ic_frame, text="1.00" if i == 0 else "0.00")
            label.grid(row=i+1, column=2, padx=5)
            self.pos_vars.append(var)
            self.pos_labels.append(label)
            var.trace_add('write', self.update_labels)
        
        ttk.Label(ic_frame, text="Initial Velocities", 
                 font=('Arial', 10, 'bold')).grid(row=5, column=0, columnspan=3, pady=(15,5))
        
        self.vel_vars = []
        self.vel_labels = []
        for i in range(4):
            ttk.Label(ic_frame, text=f"v{i+1}(0):").grid(row=i+6, column=0, sticky=tk.W, pady=3)
            var = tk.DoubleVar(value=0.0)
            slider = ttk.Scale(ic_frame, from_=-3.0, to=3.0, variable=var, 
                             orient=tk.HORIZONTAL, length=150)
            slider.grid(row=i+6, column=1, pady=3, padx=5)
            label = ttk.Label(ic_frame, text="0.00")
            label.grid(row=i+6, column=2, padx=5)
            self.vel_vars.append(var)
            self.vel_labels.append(label)
            var.trace_add('write', self.update_labels)
        
        # Forcing tab
        forcing_frame = ttk.Frame(notebook, padding="10")
        notebook.add(forcing_frame, text="Forcing")
        
        self.forcing_enabled = []
        self.forcing_amp_vars = []
        self.forcing_freq_vars = []
        
        for i in range(4):
            ttk.Label(forcing_frame, text=f"Mass {i+1}:", 
                     font=('Arial', 9, 'bold')).grid(row=i*3, column=0, columnspan=3, pady=(10,5), sticky=tk.W)
            
            enabled = tk.BooleanVar(value=False)
            ttk.Checkbutton(forcing_frame, text="Enable Force", 
                          variable=enabled).grid(row=i*3+1, column=0, columnspan=3, sticky=tk.W)
            self.forcing_enabled.append(enabled)
            
            ttk.Label(forcing_frame, text="Amp:").grid(row=i*3+2, column=0, sticky=tk.W, pady=3)
            amp_var = tk.DoubleVar(value=1.0)
            ttk.Scale(forcing_frame, from_=0.0, to=5.0, variable=amp_var, 
                     orient=tk.HORIZONTAL, length=100).grid(row=i*3+2, column=1, pady=3)
            self.forcing_amp_vars.append(amp_var)
            
            ttk.Label(forcing_frame, text="Freq:").grid(row=i*3+2, column=2, sticky=tk.W, pady=3, padx=(10,0))
            freq_var = tk.DoubleVar(value=1.0)
            ttk.Scale(forcing_frame, from_=0.1, to=5.0, variable=freq_var, 
                     orient=tk.HORIZONTAL, length=100).grid(row=i*3+2, column=3, pady=3)
            self.forcing_freq_vars.append(freq_var)
        
        # Time span
        ttk.Label(self.control_frame, text="Time Span (T):", 
                 font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=(10,3))
        row += 1
        
        self.time_var = tk.DoubleVar(value=30.0)
        self.time_slider = ttk.Scale(self.control_frame, from_=10.0, to=100.0, 
                                     variable=self.time_var, orient=tk.HORIZONTAL, length=200)
        self.time_slider.grid(row=row, column=0, columnspan=2, pady=3)
        self.time_label = ttk.Label(self.control_frame, text="30.0")
        self.time_label.grid(row=row, column=2, padx=5)
        self.time_var.trace_add('write', self.update_labels)
        row += 1
        
        # Buttons
        button_frame = ttk.Frame(self.control_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=15)
        row += 1
        
        ttk.Button(button_frame, text="Update Plot", 
                  command=self.solve_and_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", 
                  command=self.reset_values).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Preset: Wave", 
                  command=self.preset_wave).pack(side=tk.LEFT, padx=5)
        
        # Auto-update
        self.auto_update = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="Auto-update", 
                       variable=self.auto_update).grid(row=row, column=0, columnspan=3, pady=5)
    
    def setup_plots(self):
        self.fig = Figure(figsize=(11, 9))
        
        # Create 5 subplots
        self.ax1 = self.fig.add_subplot(321)  # Mass positions vs time
        self.ax2 = self.fig.add_subplot(322)  # Animation snapshot
        self.ax3 = self.fig.add_subplot(323)  # Phase space mass 1
        self.ax4 = self.fig.add_subplot(324)  # Phase space mass 2
        self.ax5 = self.fig.add_subplot(325)  # Phase space mass 3
        self.ax6 = self.fig.add_subplot(326)  # Phase space mass 4
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def get_forcing_functions(self):
        forcing_funcs = []
        for i in range(4):
            if self.forcing_enabled[i].get():
                amp = self.forcing_amp_vars[i].get()
                freq = self.forcing_freq_vars[i].get()
                forcing_funcs.append(lambda t, a=amp, f=freq: a * np.sin(f * t))
            else:
                forcing_funcs.append(lambda t: 0)
        return forcing_funcs
    
    def solve_and_plot(self):
        try:
            # Get parameters
            masses = [var.get() for var in self.mass_vars]
            springs = [var.get() for var in self.spring_vars]
            damping = [var.get() for var in self.damping_vars]
            init_pos = [var.get() for var in self.pos_vars]
            init_vel = [var.get() for var in self.vel_vars]
            t_end = self.time_var.get()
            
            forcing_funcs = self.get_forcing_functions()
            
            # Create solver
            solver = CoupledOscillatorSolver(masses, springs, damping, forcing_funcs)
            
            # Solve
            t, positions, velocities = solver.solve(init_pos, init_vel, (0, t_end))
            
            # Clear all plots
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
                ax.clear()
            
            # Plot 1: All positions vs time
            colors = ['b', 'r', 'g', 'm']
            for i in range(4):
                self.ax1.plot(t, positions[:, i], colors[i], linewidth=2, label=f'Mass {i+1}')
            self.ax1.set_xlabel('Time (s)', fontsize=9)
            self.ax1.set_ylabel('Position', fontsize=9)
            self.ax1.set_title('Positions vs Time', fontsize=10, fontweight='bold')
            self.ax1.legend(fontsize=8)
            self.ax1.grid(True, alpha=0.3)
            
            # Plot 2: System visualization (snapshot at final time)
            self.ax2.set_xlim(-1, 5)
            self.ax2.set_ylim(-2, 2)
            
            # Draw walls
            self.ax2.axvline(x=0, color='k', linewidth=3)
            self.ax2.axvline(x=4, color='k', linewidth=3)
            
            # Draw masses at final positions
            final_pos = positions[-1, :]
            x_positions = [1, 2, 3, 4]
            for i, (xp, fp) in enumerate(zip(x_positions, final_pos)):
                # Draw spring from previous position
                if i == 0:
                    self.ax2.plot([0, xp+fp], [0, 0], 'k-', linewidth=1)
                else:
                    self.ax2.plot([x_positions[i-1]+final_pos[i-1], xp+fp], [0, 0], 'k-', linewidth=1)
                
                # Draw mass
                self.ax2.plot(xp+fp, 0, 'o', color=colors[i], markersize=20, label=f'm{i+1}')
            
            # Draw final spring to wall
            self.ax2.plot([x_positions[-1]+final_pos[-1], 4], [0, 0], 'k-', linewidth=1)
            
            self.ax2.set_xlabel('Position', fontsize=9)
            self.ax2.set_title('System Configuration (Final)', fontsize=10, fontweight='bold')
            self.ax2.legend(fontsize=8)
            self.ax2.grid(True, alpha=0.3)
            
            # Plots 3-6: Phase portraits
            phase_axes = [self.ax3, self.ax4, self.ax5, self.ax6]
            for i, ax in enumerate(phase_axes):
                ax.plot(positions[:, i], velocities[:, i], colors[i], linewidth=2)
                ax.plot(positions[0, i], velocities[0, i], 'ko', markersize=6)
                ax.plot(positions[-1, i], velocities[-1, i], 'ro', markersize=6)
                ax.set_xlabel(f'Position x{i+1}', fontsize=9)
                ax.set_ylabel(f'Velocity v{i+1}', fontsize=9)
                ax.set_title(f'Phase Portrait - Mass {i+1}', fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error solving system: {str(e)}")
    
    def update_labels(self, *args):
        for i, (var, label) in enumerate(zip(self.mass_vars, self.mass_labels)):
            label.config(text=f"{var.get():.2f}")
        
        for i, (var, label) in enumerate(zip(self.spring_vars, self.spring_labels)):
            label.config(text=f"{var.get():.2f}")
        
        for i, (var, label) in enumerate(zip(self.damping_vars, self.damping_labels)):
            label.config(text=f"{var.get():.2f}")
        
        for i, (var, label) in enumerate(zip(self.pos_vars, self.pos_labels)):
            label.config(text=f"{var.get():.2f}")
        
        for i, (var, label) in enumerate(zip(self.vel_vars, self.vel_labels)):
            label.config(text=f"{var.get():.2f}")
        
        self.time_label.config(text=f"{self.time_var.get():.1f}")
        
        if self.auto_update.get():
            self.solve_and_plot()
    
    def reset_values(self):
        for var in self.mass_vars:
            var.set(1.0)
        for var in self.spring_vars:
            var.set(2.0)
        for var in self.damping_vars:
            var.set(0.1)
        for i, var in enumerate(self.pos_vars):
            var.set(1.0 if i == 0 else 0.0)
        for var in self.vel_vars:
            var.set(0.0)
        for var in self.forcing_enabled:
            var.set(False)
        self.time_var.set(30.0)
        self.solve_and_plot()
    
    def preset_wave(self):
        """Preset for wave-like initial condition"""
        positions = [1.0, 0.5, -0.5, -1.0]
        for i, var in enumerate(self.pos_vars):
            var.set(positions[i])
        for var in self.vel_vars:
            var.set(0.0)
        self.solve_and_plot()


if __name__ == "__main__":
    root = tk.Tk()
    app = CoupledOscillatorGUI(root)
    root.mainloop()
