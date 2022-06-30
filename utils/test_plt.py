import numpy as np
import matplotlib.pyplot as plt

width = 512
height = 512
rows = 20
cols = 2
axes = []
fig = plt.figure(figsize=(30, 30))
# fig = plt.figure()
for a in range(rows * cols):
    b = np.random.randint(7, size=(height, width))
    axes.append(fig.add_subplot(rows, cols, a + 1))
    subplot_title = ("Subplot" + str(a))
    axes[-1].set_title(subplot_title)
    plt.imshow(b)
fig.tight_layout()
plt.show()
