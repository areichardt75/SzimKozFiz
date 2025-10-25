import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class ODESolver:
    """
    Solver for ordinary differential equations with constant coefficients.
    Particularly useful for damped and forced harmonic oscillations.
    """
    
    def __init__(self, m=1.0, c=0.5, k=1.0, forcing_func=None):
        """
        Initialize the ODE solver for equations of the form:
        m*y'' + c*y' + k*y = F(t)
        
        Parameters:
        -----------
        m : float
            Mass coefficient (or coefficient of y'')
        c : float
            Damping coefficient (or coefficient of y')
        k : float
            Stiffness coefficient (or coefficient of y)
        forcing_func : callable
            External forcing function F(t). If None, system is unforced.
        """
        self.m = m
        self.c = c
        self.k = k
        self.forcing_func = forcing_func if forcing_func else lambda t: 0
        
        # Calculate natural frequency and damping ratio
        self.omega_n = np.sqrt(k/m)  # Natural frequency
        self.zeta = c / (2 * np.sqrt(m*k))  # Damping ratio
        
    def system(self, state, t):
        """
        Convert 2nd order ODE to system of 1st order ODEs.
        state = [y, y']
        """
        y, dydt = state
        d2ydt2 = (self.forcing_func(t) - self.c*dydt - self.k*y) / self.m
        return [dydt, d2ydt2]
    
    def solve(self, y0, v0, t_span, num_points=1000):
        """
        Solve the ODE with given initial conditions.
        
        Parameters:
        -----------
        y0 : float
            Initial position
        v0 : float
            Initial velocity
        t_span : tuple
            Time span (t_start, t_end)
        num_points : int
            Number of time points
            
        Returns:
        --------
        t : array
            Time values
        y : array
            Position values
        dydt : array
            Velocity values
        """
        t = np.linspace(t_span[0], t_span[1], num_points)
        initial_state = [y0, v0]
        solution = odeint(self.system, initial_state, t)
        
        y = solution[:, 0]
        dydt = solution[:, 1]
        
        return t, y, dydt
    
    def plot_solution(self, t, y, dydt, title="ODE Solution"):
        """Plot the solution."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Position vs time
        ax1.plot(t, y, 'b-', linewidth=2)
        ax1.set_xlabel('Time (t)')
        ax1.set_ylabel('Position y(t)')
        ax1.set_title(f'{title} - Position')
        ax1.grid(True, alpha=0.3)
        
        # Velocity vs time
        ax2.plot(t, dydt, 'r-', linewidth=2)
        ax2.set_xlabel('Time (t)')
        ax2.set_ylabel('Velocity y\'(t)')
        ax2.set_title(f'{title} - Velocity')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_phase_portrait(self, t, y, dydt, title="Phase Portrait"):
        """Plot phase portrait (velocity vs position)."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(y, dydt, 'b-', linewidth=2)
        ax.plot(y[0], dydt[0], 'go', markersize=10, label='Start')
        ax.plot(y[-1], dydt[-1], 'ro', markersize=10, label='End')
        ax.set_xlabel('Position y(t)')
        ax.set_ylabel('Velocity y\'(t)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return fig
    
    def get_damping_type(self):
        """Determine the type of damping."""
        if self.zeta < 1:
            return f"Underdamped (ζ = {self.zeta:.3f})"
        elif self.zeta == 1:
            return f"Critically damped (ζ = {self.zeta:.3f})"
        else:
            return f"Overdamped (ζ = {self.zeta:.3f})"


# ============ EXAMPLES ============

def example_1_free_underdamped():
    """Example 1: Free underdamped oscillation"""
    print("Example 1: Free Underdamped Oscillation")
    print("-" * 50)
    
    solver = ODESolver(m=1.0, c=0.2, k=4.0)
    print(f"Natural frequency: ω_n = {solver.omega_n:.3f} rad/s")
    print(f"Damping type: {solver.get_damping_type()}")
    
    t, y, dydt = solver.solve(y0=1.0, v0=0.0, t_span=(0, 20))
    solver.plot_solution(t, y, dydt, "Free Underdamped Oscillation")
    solver.plot_phase_portrait(t, y, dydt, "Phase Portrait - Underdamped")
    print()


def example_2_forced_oscillation():
    """Example 2: Forced harmonic oscillation"""
    print("Example 2: Forced Harmonic Oscillation")
    print("-" * 50)
    
    # Forcing frequency
    omega_f = 1.5
    F0 = 2.0
    forcing = lambda t: F0 * np.sin(omega_f * t)
    
    solver = ODESolver(m=1.0, c=0.3, k=4.0, forcing_func=forcing)
    print(f"Natural frequency: ω_n = {solver.omega_n:.3f} rad/s")
    print(f"Forcing frequency: ω_f = {omega_f:.3f} rad/s")
    print(f"Damping type: {solver.get_damping_type()}")
    
    t, y, dydt = solver.solve(y0=0.0, v0=0.0, t_span=(0, 30))
    solver.plot_solution(t, y, dydt, "Forced Harmonic Oscillation")
    solver.plot_phase_portrait(t, y, dydt, "Phase Portrait - Forced")
    print()


def example_3_resonance():
    """Example 3: Resonance (forcing near natural frequency)"""
    print("Example 3: Resonance")
    print("-" * 50)
    
    m, c, k = 1.0, 0.1, 4.0
    omega_n = np.sqrt(k/m)
    omega_f = omega_n * 0.95  # Near resonance
    F0 = 1.0
    forcing = lambda t: F0 * np.sin(omega_f * t)
    
    solver = ODESolver(m=m, c=c, k=k, forcing_func=forcing)
    print(f"Natural frequency: ω_n = {solver.omega_n:.3f} rad/s")
    print(f"Forcing frequency: ω_f = {omega_f:.3f} rad/s")
    print(f"Damping type: {solver.get_damping_type()}")
    
    t, y, dydt = solver.solve(y0=0.0, v0=0.0, t_span=(0, 50))
    solver.plot_solution(t, y, dydt, "Resonance - Large Amplitude")
    print()


def example_4_comparison():
    """Example 4: Compare different damping scenarios"""
    print("Example 4: Damping Comparison")
    print("-" * 50)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Underdamped
    solver1 = ODESolver(m=1.0, c=0.5, k=4.0)
    t1, y1, _ = solver1.solve(y0=1.0, v0=0.0, t_span=(0, 15))
    axes[0].plot(t1, y1, 'b-', linewidth=2)
    axes[0].set_title(f'Underdamped: {solver1.get_damping_type()}')
    axes[0].set_ylabel('Position y(t)')
    axes[0].grid(True, alpha=0.3)
    
    # Critically damped
    c_critical = 2 * np.sqrt(1.0 * 4.0)
    solver2 = ODESolver(m=1.0, c=c_critical, k=4.0)
    t2, y2, _ = solver2.solve(y0=1.0, v0=0.0, t_span=(0, 15))
    axes[1].plot(t2, y2, 'g-', linewidth=2)
    axes[1].set_title(f'Critically Damped: {solver2.get_damping_type()}')
    axes[1].set_ylabel('Position y(t)')
    axes[1].grid(True, alpha=0.3)
    
    # Overdamped
    solver3 = ODESolver(m=1.0, c=6.0, k=4.0)
    t3, y3, _ = solver3.solve(y0=1.0, v0=0.0, t_span=(0, 15))
    axes[2].plot(t3, y3, 'r-', linewidth=2)
    axes[2].set_title(f'Overdamped: {solver3.get_damping_type()}')
    axes[2].set_ylabel('Position y(t)')
    axes[2].set_xlabel('Time (t)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("Comparison plot created showing all three damping types\n")


# Run examples
if __name__ == "__main__":
    example_1_free_underdamped()
    example_2_forced_oscillation()
    example_3_resonance()
    example_4_comparison()
    plt.show()
