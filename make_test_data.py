import numpy as np
import scipy.io as sio

np.random.seed(1234)
fake_volume = np.random.randn(400, 100)
fake_template = np.random.randn(400, 53)
np.save("fake_volume", fake_volume)
np.save("fake_template", fake_template)
sio.savemat("fake_volume.mat", {"fake_volume": fake_volume})
sio.savemat("fake_template.mat", {"fake_template": fake_template})

