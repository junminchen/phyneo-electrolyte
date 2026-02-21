#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

dimer = 'test_data.xvg'
data = np.loadtxt(dimer)
x = data[:, 0]
y1 = data[:, 2]

fig, ax = plt.subplots(figsize=(4,3))

plt.scatter(x, y1, s=5, edgecolor='none')

ax.axline((0, 0), slope=1, linewidth=1.0, color="k", alpha=0.4)

plt.ylabel("PhyNEO Force Field (kJ/mol)")
plt.xlabel("Reference (kJ/mol)")
plt.tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)

plt.savefig(f"validation.png", dpi=300, bbox_inches='tight')