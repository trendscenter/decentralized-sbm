import gigica
import numpy as np
import scipy.io as sio

# import matlab.engine
test_volume = np.load("fake_volume.npy")
test_template = np.load("fake_template.npy")
# eng = matlab.engine.start_matlab()
# ica_ICOutMax, ica_TCMax = eng.icatb_gigicar(test_volume, test_template)
py_ICOutMax, py_TCMax = gigica.gigicar(test_volume.T, test_template.T)
icatb_ICOutMax = sio.loadmat("icatb_ICOutMax.mat")["ICOutMax"]
icatb_TCMax = sio.loadmat("icatb_TCMax.mat")["TCMax"]

print(py_ICOutMax.shape, icatb_ICOutMax.shape)

np.save("py_ICOutMax", py_ICOutMax)
np.save("py_TCMax", py_TCMax)
np.save("icatb_ICOutMax", icatb_ICOutMax)
np.save("icatb_TCMax", icatb_TCMax)
