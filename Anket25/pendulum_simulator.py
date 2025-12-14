# Simple Pendulum Simulator with GUI
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk

class PendulumSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Matematikai Inga Szimulátor")
        self.root.geometry("1200x800")
        
        # Fizikai paraméterek
        self.g = 9.81  # gravitációs gyorsulás (m/s²)
        self.L = 1.0   # inga hossza (m)
        self.theta0 = 30  # kezdeti szög (fok)
        self.omega0 = 0   # kezdeti szögsebesség (rad/s)
        self.t_max = 10   # szimuláció ideje (s)
        
        self.is_animating = False
        self.animation = None
        
        self.setup_gui()
        
    def setup_gui(self):
        # Bal oldali vezérlőpanel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Cím
        title = ttk.Label(control_frame, text="Paraméterek", font=('Arial', 14, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Kezdeti szög
        ttk.Label(control_frame, text="Kezdeti szög (fok):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.theta0_var = tk.DoubleVar(value=self.theta0)
        theta0_scale = ttk.Scale(control_frame, from_=0, to=180, variable=self.theta0_var, 
                                 orient=tk.HORIZONTAL, length=200)
        theta0_scale.grid(row=1, column=1, pady=5)
        self.theta0_label = ttk.Label(control_frame, text=f"{self.theta0:.1f}°")
        self.theta0_label.grid(row=1, column=2, padx=5)
        
        # Kezdeti szögsebesség
        ttk.Label(control_frame, text="Kezdeti szögsebesség (rad/s):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.omega0_var = tk.DoubleVar(value=self.omega0)
        omega0_scale = ttk.Scale(control_frame, from_=-10, to=10, variable=self.omega0_var,
                                 orient=tk.HORIZONTAL, length=200)
        omega0_scale.grid(row=2, column=1, pady=5)
        self.omega0_label = ttk.Label(control_frame, text=f"{self.omega0:.1f}")
        self.omega0_label.grid(row=2, column=2, padx=5)
        
        # Inga hossza
        ttk.Label(control_frame, text="Inga hossza (m):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.L_var = tk.DoubleVar(value=self.L)
        L_scale = ttk.Scale(control_frame, from_=0.5, to=3.0, variable=self.L_var,
                           orient=tk.HORIZONTAL, length=200)
        L_scale.grid(row=3, column=1, pady=5)
        self.L_label = ttk.Label(control_frame, text=f"{self.L:.2f} m")
        self.L_label.grid(row=3, column=2, padx=5)
        
        # Szimuláció ideje
        ttk.Label(control_frame, text="Szimuláció ideje (s):").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.t_max_var = tk.DoubleVar(value=self.t_max)
        t_max_scale = ttk.Scale(control_frame, from_=5, to=30, variable=self.t_max_var,
                                orient=tk.HORIZONTAL, length=200)
        t_max_scale.grid(row=4, column=1, pady=5)
        self.t_max_label = ttk.Label(control_frame, text=f"{self.t_max:.1f} s")
        self.t_max_label.grid(row=4, column=2, padx=5)
        
        # Gombok
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=20)
        
        self.simulate_btn = ttk.Button(button_frame, text="Szimuláció futtatása", 
                                       command=self.run_simulation)
        self.simulate_btn.grid(row=0, column=0, padx=5)
        
        self.animate_btn = ttk.Button(button_frame, text="Animáció indítása",
                                      command=self.toggle_animation)
        self.animate_btn.grid(row=0, column=1, padx=5)
        
        # Információs szöveg
        info_text = """
Matematikai inga:

Nemlineáris egyenlet:
θ'' + (g/L)·sin(θ) = 0

Lineáris közelítés:
θ'' + (g/L)·θ = 0

A lineáris közelítés kis
szögek esetén pontos
(< 15-20 fok).
        """
        info_label = ttk.Label(control_frame, text=info_text, justify=tk.LEFT,
                              background='#f0f0f0', relief=tk.SUNKEN, padding=10)
        info_label.grid(row=6, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        # Jobb oldali grafikon terület
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Grid súlyok beállítása
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Értékek frissítése
        theta0_scale.configure(command=lambda v: self.update_label(self.theta0_label, float(v), "°"))
        omega0_scale.configure(command=lambda v: self.update_label(self.omega0_label, float(v), ""))
        L_scale.configure(command=lambda v: self.update_label(self.L_label, float(v), " m"))
        t_max_scale.configure(command=lambda v: self.update_label(self.t_max_label, float(v), " s"))
        
    def update_label(self, label, value, unit):
        label.config(text=f"{value:.1f}{unit}")
        
    def nonlinear_pendulum(self, y, t):
        """Nemlineáris inga differenciálegyenlete"""
        theta, omega = y
        dydt = [omega, -(self.g / self.L) * np.sin(theta)]
        return dydt
    
    def linear_pendulum(self, y, t):
        """Lineáris inga differenciálegyenlete"""
        theta, omega = y
        dydt = [omega, -(self.g / self.L) * theta]
        return dydt
    
    def run_simulation(self):
        # Paraméterek frissítése
        self.theta0 = self.theta0_var.get()
        self.omega0 = self.omega0_var.get()
        self.L = self.L_var.get()
        self.t_max = self.t_max_var.get()
        
        # Idővektor
        self.t = np.linspace(0, self.t_max, 1000)
        
        # Kezdeti feltételek (radiánban)
        y0 = [np.radians(self.theta0), self.omega0]
        
        # Numerikus megoldás
        self.sol_nonlinear = odeint(self.nonlinear_pendulum, y0, self.t)
        self.sol_linear = odeint(self.linear_pendulum, y0, self.t)
        
        # Szögek fokra konvertálása
        self.theta_nonlinear = np.degrees(self.sol_nonlinear[:, 0])
        self.theta_linear = np.degrees(self.sol_linear[:, 0])
        
        # Grafikonok rajzolása
        self.plot_results()
        
    def plot_results(self):
        # Régi figure törlése
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        # Új figure létrehozása
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        
        # 1. Szög vs idő
        ax1 = self.fig.add_subplot(3, 2, 1)
        ax1.plot(self.t, self.theta_nonlinear, 'b-', label='Nemlineáris', linewidth=2)
        ax1.plot(self.t, self.theta_linear, 'r--', label='Lineáris', linewidth=2)
        ax1.set_xlabel('Idő (s)')
        ax1.set_ylabel('Szög (fok)')
        ax1.set_title('Szög időfüggése')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Szögsebesség vs idő
        ax2 = self.fig.add_subplot(3, 2, 2)
        omega_nonlinear = np.degrees(self.sol_nonlinear[:, 1])
        omega_linear = np.degrees(self.sol_linear[:, 1])
        ax2.plot(self.t, omega_nonlinear, 'b-', label='Nemlineáris', linewidth=2)
        ax2.plot(self.t, omega_linear, 'r--', label='Lineáris', linewidth=2)
        ax2.set_xlabel('Idő (s)')
        ax2.set_ylabel('Szögsebesség (fok/s)')
        ax2.set_title('Szögsebesség időfüggése')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Fázistér
        ax3 = self.fig.add_subplot(3, 2, 3)
        ax3.plot(self.theta_nonlinear, omega_nonlinear, 'b-', label='Nemlineáris', linewidth=2)
        ax3.plot(self.theta_linear, omega_linear, 'r--', label='Lineáris', linewidth=2)
        ax3.set_xlabel('Szög (fok)')
        ax3.set_ylabel('Szögsebesség (fok/s)')
        ax3.set_title('Fázistér diagram')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Eltérés
        ax4 = self.fig.add_subplot(3, 2, 4)
        diff = np.abs(self.theta_nonlinear - self.theta_linear)
        ax4.plot(self.t, diff, 'g-', linewidth=2)
        ax4.set_xlabel('Idő (s)')
        ax4.set_ylabel('Abszolút eltérés (fok)')
        ax4.set_title('Eltérés a két megoldás között')
        ax4.grid(True, alpha=0.3)
        
        # 5-6. Inga vizualizáció (kezdeti és jelenlegi állapot)
        self.ax5 = self.fig.add_subplot(3, 2, 5)
        self.ax5.set_xlim(-self.L*1.2, self.L*1.2)
        self.ax5.set_ylim(-self.L*1.2, self.L*0.3)
        self.ax5.set_aspect('equal')
        self.ax5.set_title('Nemlineáris inga')
        self.ax5.grid(True, alpha=0.3)
        
        self.ax6 = self.fig.add_subplot(3, 2, 6)
        self.ax6.set_xlim(-self.L*1.2, self.L*1.2)
        self.ax6.set_ylim(-self.L*1.2, self.L*0.3)
        self.ax6.set_aspect('equal')
        self.ax6.set_title('Lineáris inga')
        self.ax6.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        
        # Canvas létrehozása
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def draw_pendulum(self, ax, theta_deg):
        """Egy adott szögnél rajzolja meg az ingát"""
        ax.clear()
        ax.set_xlim(-self.L*1.2, self.L*1.2)
        ax.set_ylim(-self.L*1.2, self.L*0.3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        theta_rad = np.radians(theta_deg)
        x = self.L * np.sin(theta_rad)
        y = -self.L * np.cos(theta_rad)
        
        # Felfüggesztési pont
        ax.plot(0, 0, 'ko', markersize=10)
        
        # Kötél
        ax.plot([0, x], [0, y], 'k-', linewidth=2)
        
        # Tömeg
        ax.plot(x, y, 'ro', markersize=20)
        
        # Szög jelölése
        angle_arc = np.linspace(0, theta_rad, 50)
        arc_r = 0.2
        ax.plot(arc_r * np.sin(angle_arc), -arc_r * np.cos(angle_arc), 'b--', linewidth=1)
        
        ax.text(0.1, -0.15, f'θ = {theta_deg:.1f}°', fontsize=10)
        
    def toggle_animation(self):
        if not hasattr(self, 'sol_nonlinear'):
            self.run_simulation()
            
        if self.is_animating:
            if self.animation:
                self.animation.event_source.stop()
            self.is_animating = False
            self.animate_btn.config(text="Animáció indítása")
        else:
            self.is_animating = True
            self.animate_btn.config(text="Animáció leállítása")
            self.start_animation()
            
    def start_animation(self):
        self.frame_idx = 0
        
        def animate(frame):
            if not self.is_animating:
                return
                
            idx = frame % len(self.t)
            
            # Nemlineáris inga
            self.draw_pendulum(self.ax5, self.theta_nonlinear[idx])
            self.ax5.set_title(f'Nemlineáris inga (t={self.t[idx]:.2f}s)')
            
            # Lineáris inga
            self.draw_pendulum(self.ax6, self.theta_linear[idx])
            self.ax6.set_title(f'Lineáris inga (t={self.t[idx]:.2f}s)')
            
            self.canvas.draw()
            
        self.animation = FuncAnimation(self.fig, animate, frames=len(self.t),
                                      interval=20, repeat=True, blit=False)
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = PendulumSimulator(root)
    root.mainloop()

if __name__ == "__main__":
    main()