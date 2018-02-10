import matplotlib
matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt


image = mpimg.imread('../test_images/test1.jpg')
image_flipped = np.flip(image, axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
fig.tight_layout()
ax1.imshow(image)
ax2.imshow(image_flipped)
plt.show()