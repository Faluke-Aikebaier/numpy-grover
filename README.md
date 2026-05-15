# numpy-grover
Quantum-inspired global minimiser in pure NumPy. Implements the Grover/Durr-Hoyer algorithm with hierarchical zoom-and-refine for high accuracy, and a hybrid adaptive minimiser for D=2–50+ that routes each dimension to Grover (multimodal), Brent (smooth), or DE/CMA-ES (global basin) automatically.
