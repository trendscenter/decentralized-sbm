"""
pygigicar
"""
import numpy as np
import numpy.matlib
import scipy as sp
import scipy.linalg as la
import matlab
import scipy.io as sio

def nege(x):
    y = np.log(np.cosh(x))
    E1 = np.mean(y)
    E2 = 0.3745672075
    negentropy = np.power(E1 - E2, 2)
    return negentropy

def save_test_workspace(**kwargs):
    test_dict = {}
    for k,v in kwargs.items():
        if k == 'matlab_engine':
            continue
        test_dict['test_'+k] = v
    sio.savemat('test.mat', test_dict)
    

def gigicar(FmriMatr, ICRefMax, matlab_engine=None):
    n, m = FmriMatr.shape
    nmean = numpy.matlib.repmat(
        np.mean(FmriMatr, 1).reshape(n, 1), 1, m
    )  # Remove Column Means
    FmriMat = FmriMatr - numpy.matlib.repmat(
        np.mean(FmriMatr, 1).reshape(n, 1), 1, m
    )  # Remove Column Means
    CovFmri = (FmriMat.dot(FmriMat.T)) / m  # TODO: use np.cov

    if matlab_engine is not None:
        E, D = matlab_engine.eig(matlab.double(CovFmri.tolist()), nargout=2)
        E = np.asarray(E)
        D = np.asarray(D)
        dd = np.diag(D)
    else:
        dd, E = np.linalg.eig(CovFmri)
        D = dd * np.eye(E.shape[0], E.shape[1])
    EsICnum = ICRefMax.shape[0]
    index = np.argsort(np.diag(D))
    #index = index[::-1]
    eigenvalues = dd[index]

    cols = E.shape[1]
    #Esort = np.zeros_like(E)
    #dsort = np.zeros_like(eigenvalues)
    Esort = E[:,index[::-1]]
    dsort = dd[index[::-1]]
    #for i in range(cols):
    #   iii = cols - i - 1
        #Esort[:, i] = E[:, index[cols - i - 1]]
        #dsort[i] = eigenvalues[index[cols - i - 1]]
    
    thr = 0
    numpc = 0
    for i in range(cols):
        if dsort[i] > thr:
            numpc = numpc + 1
    Epart = Esort[:, :numpc]
    dpart = dsort[:numpc]
    Lambda_part = np.diag(dpart)
    WhitenMatrix = np.linalg.inv(la.sqrtm(Lambda_part)).dot(Epart.T)
    Y = WhitenMatrix.dot(FmriMat)

    if thr < 1e-10 and numpc < n:
        for i in range(Y.shape[0]):
            Y[i, :] = Y[i, :] / np.std(Y[i, :], ddof=1)

    Yinv = np.linalg.pinv(Y)
    ICRefMaxN = np.zeros((EsICnum, m))
    nmean = np.mean(ICRefMax, 1).reshape(ICRefMax.shape[0], 1)

    ICRefMaxC = ICRefMax - numpy.matlib.repmat(nmean, 1, m)
    for i in range(EsICnum):
        ICRefMaxN[i, :] = ICRefMaxC[i, :] / np.std(ICRefMaxC[i, :], ddof=1)

    NegeEva = np.zeros((EsICnum, 1))
    for i in range(EsICnum):
        NegeEva[i] = nege(ICRefMaxN[i, :])
    iternum = 100
    a = 0.5
    b = 1 - a
    EGv = 0.3745672075
    ErChuPai = 2 / np.pi
    ICOutMax = np.zeros((EsICnum, m))
    for ICnum in range(EsICnum):
        reference = ICRefMaxN[ICnum, :]
        wc = (reference.dot(Yinv)).T
        wc = wc/np.linalg.norm(wc, 2)
        y1 = wc.T.dot(Y)
        EyrInitial = (1 / m) * (y1.dot(reference.T))
        NegeInitial = nege(y1)
        c = (np.tan((EyrInitial * np.pi) / 2)) / NegeInitial
        IniObjValue = a * ErChuPai * np.arctan(c * NegeInitial) + b * EyrInitial

        itertime = 1
        Nemda = 1
        for i in range(iternum):
            Cosy1 = np.cosh(y1)
            logCosy1 = np.log(Cosy1)
            EGy1 = np.mean(logCosy1)
            Negama = EGy1 - EGv
            EYgy = (1 / m) * Y.dot((np.tanh(y1)).T)
            Jy1 = (EGy1 - EGv) ** 2
            KwDaoshu = ErChuPai * c * (1 / (1 + (c * Jy1) ** 2))
            Simgrad = (1 / m) * Y.dot(reference.T)
            g = a * KwDaoshu * 2 * Negama * EYgy + b * Simgrad
            d = g / (g.T.dot(g)) ** 0.5
            wx = wc + Nemda * d
            wx = wx / np.linalg.norm(wx, 2)
            y3 = wx.T.dot(Y)
            PreObjValue = a * ErChuPai * np.arctan(c * (nege(y3))) + b * (
                1 / m
            ) * y3.dot(reference.T)
            ObjValueChange = PreObjValue - IniObjValue
            ftol = 0.02
            dg = g.T.dot(d)
            ArmiCondiThr = Nemda * ftol * dg
            if ObjValueChange < ArmiCondiThr:
                Nemda = Nemda / 2
                continue
            if (wc - wx).T.dot(wc - wx) < 1e-5:
                break
            elif itertime == iternum:
                break
            IniObjValue = PreObjValue
            y1 = y3
            wc = wx
            itertime = itertime + 1
        Source = wx.T.dot(Y)
        ICOutMax[ICnum, :] = Source
    # end
    TCMax = (1 / m) * FmriMatr.dot(ICOutMax.T)
    return ICOutMax, TCMax


if __name__ == "__main__":
    test_volume = np.load("fake_volume.npy")
    test_template = np.load("fake_template.npy")
    # eng = matlab.engine.start_matlab()
    # ica_ICOutMax, ica_TCMax = eng.icatb_gigicar(test_volume, test_template)
    py_ICOutMax, py_TCMax = gigicar(test_volume.T, test_template.T)

    np.save("py_ICOutMax", py_ICOutMax)
    np.save("py_TCMax", py_TCMax)
