import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown
import ipywidgets as widgets

# -----------------------------
# Constants
# -----------------------------
mec2 = 0.511  # MeV
threshold = 2 * mec2  # 1.022 MeV

# -----------------------------
# Approximate probability model
# -----------------------------
def pair_production_probability(E_gamma, Z=82):
    if E_gamma <= threshold:
        return 0.0

    x = (E_gamma - threshold) / threshold
    prob = (1 - np.exp(-2.5 * x)) * (Z / 82.0) ** 2
    return min(prob, 1.0)

# -----------------------------
# Single event simulation
# -----------------------------
def simulate_event(E_gamma, Z=82):
    prob = pair_production_probability(E_gamma, Z)
    r = np.random.rand()

    if r > prob:
        return {
            "pair_created": False,
            "E_gamma": E_gamma,
            "probability": prob
        }

    kinetic_available = E_gamma - threshold
    share = np.random.rand()

    T_electron = share * kinetic_available
    T_positron = (1 - share) * kinetic_available

    return {
        "pair_created": True,
        "E_gamma": E_gamma,
        "probability": prob,
        "T_electron": T_electron,
        "T_positron": T_positron,
        "E_electron_total": T_electron + mec2,
        "E_positron_total": T_positron + mec2
    }

# -----------------------------
# Many events simulation
# -----------------------------
def run_simulation(num_events=1000, Z=82, Emin=0.2, Emax=10.0):
    E_values = np.random.uniform(Emin, Emax, num_events)
    created = []
    electron_kinetic = []
    positron_kinetic = []

    for E in E_values:
        result = simulate_event(E, Z)
        created.append(1 if result["pair_created"] else 0)

        if result["pair_created"]:
            electron_kinetic.append(result["T_electron"])
            positron_kinetic.append(result["T_positron"])

    return E_values, np.array(created), np.array(electron_kinetic), np.array(positron_kinetic)

# -----------------------------
# Interactive visualization
# -----------------------------
def interactive_plot(E_gamma=1.0, Z=82, num_events=1000):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: probability curve
    E_plot = np.linspace(0.0, 10.0, 400)
    P_plot = [pair_production_probability(E, Z) for E in E_plot]

    axes[0].plot(E_plot, P_plot, label="Approximate Probability")
    axes[0].axvline(threshold, linestyle='--', label=f"Threshold = {threshold:.3f} MeV")
    axes[0].axvline(E_gamma, linestyle=':', label=f"Selected E = {E_gamma:.2f} MeV")
    axes[0].set_xlabel("Photon Energy (MeV)")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Pair Production Probability vs Photon Energy")
    axes[0].grid(True)
    axes[0].legend()

    # Plot 2: Monte Carlo rate
    E_values, created, e_kin, p_kin = run_simulation(num_events=num_events, Z=Z)

    bins = np.linspace(0.2, 10.0, 25)
    digitized = np.digitize(E_values, bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    rates = []

    for i in range(1, len(bins)):
        mask = digitized == i
        if np.sum(mask) > 0:
            rates.append(np.mean(created[mask]))
        else:
            rates.append(0)

    axes[1].plot(bin_centers, rates, marker='o')
    axes[1].axvline(threshold, linestyle='--', label=f"Threshold = {threshold:.3f} MeV")
    axes[1].axvline(E_gamma, linestyle=':', label=f"Selected E = {E_gamma:.2f} MeV")
    axes[1].set_xlabel("Photon Energy (MeV)")
    axes[1].set_ylabel("Simulated Creation Rate")
    axes[1].set_title("Monte Carlo Simulation")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Show single event result
    result = simulate_event(E_gamma, Z)

    print("=" * 50)
    print(f"Photon energy = {E_gamma:.3f} MeV")
    print(f"Threshold     = {threshold:.3f} MeV")
    print(f"Nuclear Z     = {Z}")
    print(f"Probability   = {result['probability']:.4f}")
    print("-" * 50)

    if result["pair_created"]:
        print("Result: Pair produced")
        print(f"Electron kinetic energy   = {result['T_electron']:.4f} MeV")
        print(f"Positron kinetic energy   = {result['T_positron']:.4f} MeV")
        print(f"Electron total energy     = {result['E_electron_total']:.4f} MeV")
        print(f"Positron total energy     = {result['E_positron_total']:.4f} MeV")
    else:
        print("Result: No pair production in this event")

# -----------------------------
# Widget UI
# -----------------------------
def launch_simulation():
    interact(
        interactive_plot,
        E_gamma=FloatSlider(
            value=1.0,
            min=0.0,
            max=10.0,
            step=0.05,
            description='Eγ (MeV)'
        ),
        Z=IntSlider(
            value=82,
            min=1,
            max=92,
            step=1,
            description='Z'
        ),
        num_events=Dropdown(
            options=[500, 1000, 3000, 5000],
            value=1000,
            description='Events'
        )
    )

if __name__ == "__main__":
    launch_simulation()
