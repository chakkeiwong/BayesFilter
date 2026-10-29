"""Independent diagnostic adapter to Octave's system LAPACK, never a TF dependency.

LP64 Fortran ABI, Linux x86_64; caller verifies the loaded library is the one
linked by Octave. Only the small saved-matrix reference check may use this.
"""
import ctypes as ct
import numpy as np


class SystemLapack:
    def __init__(self, path):
        self.library=ct.CDLL(str(path))

    def _call(self, name, *args):
        packed=[]
        for value in args:
            if isinstance(value,np.ndarray):
                assert value.flags.f_contiguous and value.dtype in (np.float64,np.int32)
                packed.append(value.ctypes.data_as(ct.c_void_p))
            elif isinstance(value,(ct.c_int,ct.c_double)):
                packed.append(ct.byref(value))
            else:
                packed.append(value)
        function=getattr(self.library,name+'_');function.restype=None
        function(*packed)

    def solve(self, matrix, rhs):
        n=matrix.shape[0];ni=ct.c_int(n);one=ct.c_int(1);info=ct.c_int()
        norm=ct.c_double(float(np.linalg.norm(matrix,1)))
        factor=np.array(matrix,dtype=np.float64,order='F')
        self._call('dpotrf',b'L',ni,factor,ni,info,ct.c_size_t(1))
        assert info.value==0
        rcond=ct.c_double();work=np.zeros(4*n);iwork=np.zeros(n,dtype=np.int32)
        self._call('dpocon',b'L',ni,factor,ni,norm,rcond,work,iwork,info,ct.c_size_t(1))
        assert info.value==0
        chol_rcond=rcond.value;lu_rcond=None;rank=None
        answer=np.array(rhs,dtype=np.float64,order='F')
        if chol_rcond+1.0!=1.0:
            self._call('dpotrs',b'L',ni,one,factor,ni,answer,ni,info,ct.c_size_t(1))
            branch='Positive Definite'
        else:
            factor=np.array(matrix,dtype=np.float64,order='F');pivot=np.zeros(n,dtype=np.int32)
            self._call('dgetrf',ni,ni,factor,ni,pivot,info)
            factor_info=info.value;assert factor_info>=0
            self._call('dgecon',b'1',ni,factor,ni,norm,rcond,work,iwork,info,ct.c_size_t(1))
            assert info.value==0;lu_rcond=rcond.value
            if factor_info==0 and lu_rcond+1.0!=1.0:
                self._call('dgetrs',b'N',ni,one,factor,ni,pivot,answer,ni,info,ct.c_size_t(1))
                branch='Full'
            else:
                factor=np.array(matrix,dtype=np.float64,order='F')
                spectrum=np.zeros(n);cutoff=ct.c_double(-1.);rank_out=ct.c_int()
                # Workspace query includes a safely oversized integer workspace
                # for n=45; this is bounded diagnostic storage, not a solver cutoff.
                iw=np.zeros(100*n,dtype=np.int32);query=np.zeros(1);minus=ct.c_int(-1)
                self._call('dgelsd',ni,ni,one,factor,ni,answer,ni,spectrum,cutoff,rank_out,query,minus,iw,info)
                assert info.value==0
                length=ct.c_int(int(query[0]));workspace=np.zeros(length.value)
                self._call('dgelsd',ni,ni,one,factor,ni,answer,ni,spectrum,cutoff,rank_out,workspace,length,iw,info)
                rank=rank_out.value;branch='Singular'
        assert info.value==0
        return answer,dict(branch=branch,chol_rcond=chol_rcond,lu_rcond=lu_rcond,rank=rank)
