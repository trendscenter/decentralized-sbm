import gigica
import numpy as np

# import matlab.engine
test_volume = np.load("fake_volume.npy")
test_template = np.load("fake_template.npy")
# eng = matlab.engine.start_matlab()
# ica_ICOutMax, ica_TCMax = eng.icatb_gigicar(test_volume, test_template)
py_ICOutMax, py_TCMax = gigica.gigicar(test_volume, test_template)

np.save("py_ICOutMax", py_ICOutMax)
np.save("py_TCMax", py_TCMax)
