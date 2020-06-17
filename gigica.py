"""
pygigicar
"""
import numpy as np
import scipy as sp

def nege(x):
    y = np.log(np.cosh(x))
    E1 = np.mean(y)
    E2=0.3745672075
    negentropy=(E1- E2)^2
    return negentropy

def gigicar(FmriMatr, ICRefMax):
    n, m = FmriMatr.shape
    FmriMatr -= np.mean(FmriMatr, 1)  # Remove Column Means
    CovFmri = (FmriMat.dot(FmriMat.T)) / m  # TODO: use np.cov
    d, E = np.eig(CovFmri)  # D is eigenvalues
    D = d*np.ones_like(E)
    EsICnum = ICRefMax.shape[0]
    index = np.argsort(d)
    eigenvalues = d[index]
    cols = E.shape[1]
    Esort = np.zeros_like(E)
    dsort = np.zeros_like(eigenvalues)
    for i in range(cols):
        Esort[:, i] = E[:, index[cols-i]]
        dsort[i] = eigenvalues[index[cols-i]]
    
    thr = 0
    numpc = 0
    for i in range(cols):
        if dsort[i] > thr:
            numpc = numpc+1
    Epart = Esort[:, :numpc]
    dpart = dsort[:numpc]
    Lambda_part=np.diag(dpart)
    WhitenMatrix=np.linalg.inv(sp.linalg.sqrtm(Lambda_part))*Epart.T
    Y = WhitenMatrix*FmriMat

    if thr < 1e-10 and numpc < n:
        for i in range(Y.shape[0]):
            Y[i, :] = Y[i, :]/np.std(Y[i, :])

    Yinv = np.linalg.pinv(Y)
    ICRefMaxN = np.zeros((EsICnum, m))
    ICRefMaxC = ICRefMax - np.mean(ICRefMax, 1)
    for i in range(EsICnum):
        ICRefMaxN[i, :] = ICRefMaxC[i, :]/np.std(ICRefMaxC[i, :])
    
    NegeEva = np.zeros((EsICnum, 1))
    for i in range(EsICnum):
        NegeEva[i] = nege(ICRefMaxN[i ,:])
    iternum=100
    a=0.5
    b=1-a
    EGv=0.3745672075
    ErChuPai=2/np.pi
    ICOutMax=np.zeros((EsICnum, m))
    for ICnum in range(EsICnum):
        reference = ICRefMaxN[ICnum, :]
        wc = (reference*Yinv).T
        y1 = wc.T*Y
        EyrInitial = (1/m)*(y1)*reference.T
        NegeInitial = nege(y1)
        c = (np.tan((EyrInitial*pi)/2))/NegeInitial
        IniObjValue = a*ErChuPai*np.atan(c*NegeInitial)+b*EyrInitial

        itertime=1
        Nemda=1
        for i in range(iternum):
            Cosy1=np.cosh(y1)
            logCosy1=np.log(Cosy1)
            EGy1=np.mean(logCosy1)
            Negama=EGy1-EGv
            EYgy=(1/m)*Y*(np.tanh(y1)).T
            Jy1=(EGy1-EGv)**2
            KwDaoshu = ErChuPai*c*(1/(1+(c*Jy1)**2))
            Simgrad=(1/m)*Y*reference.T
            g=a*KwDaoshu*2*Negama*EYgy+b*Simgrad
            d=g/(g.T*g)**0.5
            wx=wc+Nemda*d
            wx=wx/np.linalg.norm(wx)
            y3=wx.T*Y
            PreObjValue = a*ErChuPai*np.atan(c*nege(y3))+b*(1/m)*y3*reference.T
            ObjValueChange=PreObjValue-IniObjValue
            ftol=0.02
            dg=g.T*d
            ArmiCondiThr=Nemda*ftol*dg
            if ObjValueChange < ArmiCondiThr:
                Nemda=Nemda/2
                continue
            if (wc-wx).T*(wc-wx) < 1e-5:
                break
            elif itertime==iternum
                break
            IniObjValue=PreObjValue
            y1=y3
            wc=wx
            itertime=itertime+1
        Source=wx.T*Y
        ICOutMax[ICnum, :] = Source
    # end
    TCMax=(1/m)*FmriMatr*ICOutMax.T



