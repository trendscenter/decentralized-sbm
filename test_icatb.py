import gigica
import numpy as np
import scipy.io as sio
import nibabel as nib

import matlab.engine
import matlab 

import matplotlib.pyplot as plt 

plt.close('all')


VOXEL = 100
TIME = 100
NCOMP = 53
REAL = True
SECOND = False
LOAD = False
np.random.seed(456)
if LOAD:
    test_volume = np.load("fake_volume.npy").T
    test_template = np.load("fake_template.npy").T
else:
    test_volume = np.random.randn(TIME, VOXEL)
    test_template = np.random.randn(NCOMP, VOXEL)
    np.save('test_volume', test_volume)
    np.save('test_template', test_template)
    sio.savemat('test_volume.mat', {'test_volume': test_volume})
    sio.savemat('test_template.mat', {'test_template': test_template})
if REAL:
    test_template = nib.load('NeuroMark_INTERP_53_63_46.nii').get_fdata()
    test_volume = nib.load('fmri_preprocessed_example.nii').get_fdata()
    test_template = np.reshape(test_template, (np.prod(test_template.shape[:3]), test_template.shape[-1])).T
    test_volume = np.reshape(test_volume, (np.prod(test_volume.shape[:3]), test_volume.shape[-1])).T
    test_mask = np.prod(test_volume > np.mean(test_volume, 1).reshape(test_volume.shape[0], 1), 0)
    test_template = test_template[:, test_mask==1]
    test_volume = test_volume[:, test_mask==1]
eng = matlab.engine.start_matlab()
matlab_test_volume = matlab.double(test_volume.tolist())
matlab_test_template = matlab.double(test_template.tolist())
icatb_output, icatb_output_2 = eng.icatb_gigicar(matlab_test_volume, matlab_test_template, nargout=2)
icatb_ICOutMax = np.asarray(icatb_output)
icatb_TCMax = np.asarray(icatb_output_2)

if SECOND:
    icatb_output, icatb_output_2 = eng.icatb_gigicar(matlab_test_volume, matlab_test_template, nargout=2)
    icatb_ICOutMax_2 = np.asarray(icatb_output)
    icatb_TCMax_2 = np.asarray(icatb_output_2)


py_ICOutMax, py_TCMax = gigica.gigicar(test_volume, test_template, matlab_engine=None)
#icatb_ICOutMax = sio.loadmat("icatb_ICOutMax.mat")["ICOutMax"]
#icatb_TCMax = sio.loadmat("icatb_TCMax.mat")["TCMax"]

print(py_ICOutMax.shape, icatb_ICOutMax.shape)

#np.save("py_ICOutMax", py_ICOutMax)
#np.save("py_TCMax", py_TCMax)
#np.save("icatb_ICOutMax", icatb_ICOutMax)
#np.save("icatb_TCMax", icatb_TCMax)

fig = plt.figure()
plt.imshow(py_ICOutMax - icatb_ICOutMax)
plt.colorbar()
if SECOND:
    fig = plt.figure()
    plt.imshow(icatb_ICOutMax - icatb_ICOutMax_2)
    plt.colorbar()
fig2 = plt.figure()
plt.imshow(py_TCMax - icatb_TCMax)
plt.colorbar()
if SECOND:
    fig2 = plt.figure()
    plt.imshow(icatb_TCMax - icatb_TCMax_2)
    plt.colorbar()
plt.show()