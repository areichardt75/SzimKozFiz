import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from mpl_toolkits.mplot3d import Axes3D

class ParticleEMSimulation:
    def __init__(self, master):
        self.master = master
        self.master.title("Részecskék elektromágneses térben")
        self.master.geometry("1400x800")
        
        # Paraméterek
        self.num_particles = 50
        self.cylinder_radius = 1.0
        self.cylinder_height = 3.0
        self.E_field = 0.5
        self.B_field = 0.3
        self.charge = 1.0
        self.mass = 1.0
        self.max_initial_speed = 0.2
        self.dt = 0.01
        
        self.particles = []
        self.time = 0
        self.is_running = False
        
        # GUI létrehozása
        self.create_gui()
        self.initialize_particles()
        
    def create_gui(self):
        # Fő keret
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Bal oldal: vezérlők
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Cím
        title = ttk.Label(control_frame, text="Elektromágneses tér szimuláció", 
                         font=("Arial", 14, "bold"))
        title.pack(pady=10)
        
        # Gombok
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start", 
                                       command=self.toggle_simulation, width=10)
        self.start_button.grid(row=0, column=0, padx=5)
        
        reset_button = ttk.Button(button_frame, text="Reset", 
                                 command=self.reset_simulation, width=10)
        reset_button.grid(row=0, column=1, padx=5)
        
        # Paraméter vezérlők
        params_frame = ttk.LabelFrame(control_frame, text="Paraméterek", padding=10)
        params_frame.pack(pady=10, fill=tk.X)
        
        # Részecskék száma
        self.create_slider(params_frame, "Részecskék száma:", 
                          self.num_particles, 10, 200, 10, 0,
                          lambda v: setattr(self, 'num_particles', int(v)))
        
        # E tér
        self.create_slider(params_frame, "E tér erőssége:", 
                          self.E_field, 0, 2, 0.1, 1,
                          lambda v: setattr(self, 'E_field', float(v)))
        
        # B tér
        self.create_slider(params_frame, "B tér erőssége:", 
                          self.B_field, 0, 2, 0.1, 2,
                          lambda v: setattr(self, 'B_field', float(v)))
        
        # Kezdősebesség
        self.create_slider(params_frame, "Max kezdősebesség:", 
                          self.max_initial_speed, 0.05, 0.5, 0.05, 3,
                          lambda v: setattr(self, 'max_initial_speed', float(v)))
        
        # Töltés/tömeg
        self.create_slider(params_frame, "Töltés/tömeg:", 
                          self.charge/self.mass, 0.1, 5, 0.1, 4,
                          lambda v: setattr(self, 'charge', float(v)))
        
        # Info
        info_frame = ttk.LabelFrame(control_frame, text="Fizikai modell", padding=10)
        info_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        info_text = """
Henger tengelye: z-irány

E tér (kék):
  Radiális az x-y síkban
  E = E₀(x/r, y/r, 0)

B tér (piros):
  Körkörös az x-y síkban
  B = B₀(-y/r, x/r, 0)

Lorentz-erő:
  F = q(E + v × B)

Mindkét tér a henger teljes 
térfogatában jelen van.
        """
        
        info_label = ttk.Label(info_frame, text=info_text, 
                              justify=tk.LEFT, font=("Courier", 9))
        info_label.pack()
        
        # Státusz
        self.status_label = ttk.Label(control_frame, text="Idő: 0.00 | Aktív: 0/0", 
                                     font=("Arial", 10))
        self.status_label.pack(pady=10)
        
        # Jobb oldal: grafikon
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Matplotlib figura
        self.fig = plt.Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Animáció
        self.anim = None
        
    def create_slider(self, parent, label, initial, min_val, max_val, resolution, row, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)
        
        var = tk.DoubleVar(value=initial)
        slider = ttk.Scale(parent, from_=min_val, to=max_val, 
                          orient=tk.HORIZONTAL, variable=var, 
                          command=lambda v: command(float(v)))
        slider.grid(row=row, column=1, sticky=tk.EW, pady=5, padx=5)
        
        value_label = ttk.Label(parent, text=f"{initial:.2f}")
        value_label.grid(row=row, column=2, sticky=tk.W, pady=5)
        
        def update_label(v):
            value_label.config(text=f"{float(v):.2f}")
            command(float(v))
        
        slider.config(command=update_label)
        
        parent.grid_columnconfigure(1, weight=1)
        
    def initialize_particles(self):
        """Részecskék inicializálása"""
        self.particles = []
        launch_radius = self.cylinder_radius * 0.8
        
        for _ in range(self.num_particles):
            angle = np.random.random() * 2 * np.pi
            r = np.sqrt(np.random.random()) * launch_radius
            
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = 0.0
            
            speed = np.random.random() * self.max_initial_speed
            v_angle = np.random.random() * 2 * np.pi
            v_theta = np.random.random() * np.pi / 4  # max 45 fok
            
            vx = speed * np.sin(v_theta) * np.cos(v_angle)
            vy = speed * np.sin(v_theta) * np.sin(v_angle)
            vz = speed * np.cos(v_theta)
            
            self.particles.append({
                'pos': np.array([x, y, z]),
                'vel': np.array([vx, vy, vz]),
                'trail': [np.array([x, y, z])],
                'active': True
            })
        
        self.time = 0
        self.draw_plot()
        
    def calculate_forces(self, particle):
        """Lorentz-erő számítása"""
        pos = particle['pos']
        vel = particle['vel']
        
        r = np.sqrt(pos[0]**2 + pos[1]**2)
        
        if r < 0.001:
            return np.array([0.0, 0.0, 0.0])
        
        # Elektromos tér: radiális az x-y síkban
        E = self.E_field * np.array([pos[0]/r, pos[1]/r, 0.0])
        
        # Mágneses tér: körkörös az x-y síkban
        B = self.B_field * np.array([-pos[1]/r, pos[0]/r, 0.0])
        
        # v × B keresztszorzat
        v_cross_B = np.cross(vel, B)
        
        # Lorentz-erő: F = q(E + v × B)
        F = self.charge * (E + v_cross_B)
        
        return F
        
    def update_particles(self):
        """Részecskék mozgásának frissítése"""
        for particle in self.particles:
            if not particle['active']:
                continue
            
            # Erő számítása
            F = self.calculate_forces(particle)
            
            # Gyorsulás
            a = F / self.mass
            
            # Euler integráció
            particle['vel'] += a * self.dt
            particle['pos'] += particle['vel'] * self.dt
            
            # Határellenőrzés
            r = np.sqrt(particle['pos'][0]**2 + particle['pos'][1]**2)
            z = particle['pos'][2]
            
            if r >= self.cylinder_radius or z < 0 or z > self.cylinder_height:
                particle['active'] = False
            
            # Nyomvonal
            if particle['active']:
                particle['trail'].append(particle['pos'].copy())
                if len(particle['trail']) > 100:
                    particle['trail'].pop(0)
        
        self.time += self.dt
        
    def draw_plot(self):
        """3D ábra rajzolása"""
        self.ax.clear()
        
        # Henger rajzolása
        theta = np.linspace(0, 2*np.pi, 50)
        z_line = np.linspace(0, self.cylinder_height, 50)
        
        # Alsó kör
        x_circle = self.cylinder_radius * np.cos(theta)
        y_circle = self.cylinder_radius * np.sin(theta)
        z_bottom = np.zeros_like(theta)
        self.ax.plot(x_circle, y_circle, z_bottom, 'gray', alpha=0.5, linewidth=2)
        
        # Felső kör
        z_top = np.ones_like(theta) * self.cylinder_height
        self.ax.plot(x_circle, y_circle, z_top, 'gray', alpha=0.5, linewidth=2)
        
        # Oldalsó vonalak
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            x = self.cylinder_radius * np.cos(angle)
            y = self.cylinder_radius * np.sin(angle)
            self.ax.plot([x, x], [y, y], [0, self.cylinder_height], 
                        'gray', alpha=0.3, linewidth=1)
        
        # Elektromos tér nyilak (radiális)
        for z_level in [self.cylinder_height * 0.25, self.cylinder_height * 0.75]:
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                r1 = self.cylinder_radius * 0.3
                r2 = self.cylinder_radius * 0.65
                x1, y1 = r1 * np.cos(angle), r1 * np.sin(angle)
                x2, y2 = r2 * np.cos(angle), r2 * np.sin(angle)
                
                self.ax.quiver(x1, y1, z_level, x2-x1, y2-y1, 0,
                              color='blue', arrow_length_ratio=0.3, linewidth=2, alpha=0.7)
        
        # Mágneses tér nyilak (körkörös)
        for z_level in [self.cylinder_height * 0.25, self.cylinder_height * 0.75]:
            angles = np.linspace(0, 2*np.pi, 9, endpoint=False)
            r = self.cylinder_radius * 0.5
            for i, angle in enumerate(angles):
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                # Tangenciális irány
                dx = -r * np.sin(angle) * 0.2
                dy = r * np.cos(angle) * 0.2
                
                self.ax.quiver(x, y, z_level, dx, dy, 0,
                              color='red', arrow_length_ratio=0.3, linewidth=2, alpha=0.7)
        
        # Részecskék és nyomvonalak
        active_count = 0
        for particle in self.particles:
            if len(particle['trail']) > 1:
                trail = np.array(particle['trail'])
                color = 'green' if particle['active'] else 'gray'
                alpha = 0.6 if particle['active'] else 0.3
                self.ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], 
                           color=color, alpha=alpha, linewidth=1)
            
            # Részecske
            pos = particle['pos']
            color = 'lime' if particle['active'] else 'darkgray'
            self.ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                          color=color, s=30, alpha=0.8)
            
            if particle['active']:
                active_count += 1
        
        # Tengelyek beállítása
        self.ax.set_xlim([-self.cylinder_radius*1.2, self.cylinder_radius*1.2])
        self.ax.set_ylim([-self.cylinder_radius*1.2, self.cylinder_radius*1.2])
        self.ax.set_zlim([0, self.cylinder_height])
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Részecskék mozgása elektromágneses térben')
        
        # Státusz frissítése
        self.status_label.config(
            text=f"Idő: {self.time:.2f} | Aktív: {active_count}/{len(self.particles)}"
        )
        
        self.canvas.draw()
        
    def animate(self, frame):
        """Animációs frame"""
        if self.is_running:
            for _ in range(5):  # 5 lépés frame-enként
                self.update_particles()
            self.draw_plot()
        return []
        
    def toggle_simulation(self):
        """Szimuláció indítása/leállítása"""
        self.is_running = not self.is_running
        
        if self.is_running:
            self.start_button.config(text="Szünet")
            if self.anim is None:
                self.anim = FuncAnimation(self.fig, self.animate, 
                                         interval=50, blit=False)
        else:
            self.start_button.config(text="Start")
            
    def reset_simulation(self):
        """Szimuláció újraindítása"""
        self.is_running = False
        self.start_button.config(text="Start")
        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
        self.initialize_particles()

def main():
    root = tk.Tk()
    app = ParticleEMSimulation(root)
    root.mainloop()

if __name__ == "__main__":
    main()
