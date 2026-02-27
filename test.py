

"""
Music 422 

Plotting tool for BS.1116 Test Results

 MUSIC 422 listening tests (HW6)
-----------------------------------------------------------------------
 © 2009-26 Marina Bosi -- All rights reserved
-----------------------------------------------------------------------
"""
import matplotlib.pyplot as plt
from sdgplot import plotSDG


dirs = [
    'castanet/',
    'glockenspiel/',
    'harpsichord/',
    'spgm/',
]
# Plot 128kbps
fig128, ax128 = plotSDG(dirs, '128kbps')
plt.show()
# Plot 192kbps
fig192, ax192 = plotSDG(dirs, '192kbps')
plt.show()
