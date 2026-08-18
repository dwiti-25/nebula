from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

circuit = Circuit("Nebula PySpice Test")

circuit.V("input", "in", circuit.gnd, 1@u_V)
circuit.R(1, "in", "out", 1@u_kΩ)
circuit.C(1, "out", circuit.gnd, 1@u_uF)

simulator = circuit.simulator(
    temperature=25,
    nominal_temperature=25
)

analysis = simulator.transient(
    step_time=1@u_us,
    end_time=10@u_ms
)

print("Simulation successful!")
print(f"Final V(out) = {float(analysis.out[-1]):.6f} V")
