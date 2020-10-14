import sys
import tensorflow as tf
import numpy as np
import time
import os
import pandas as pd
import cupy as cp

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

num_bytes = 8
dtype = tf.float64
num_modes = 25

num_cols = int(sys.argv[1])
realpart = sys.argv[2]
imagpart = sys.argv[3]
outpath = sys.argv[4]
freq = sys.argv[5]

dataset = tf.data.FixedLengthRecordDataset(realpart,record_bytes=num_bytes * num_cols)
dataset = dataset.map(lambda s: tf.reshape(tf.io.decode_raw(s, dtype, little_endian=False), (num_cols,)),num_parallel_calls=8)
dataset = dataset.batch(num_cols)
matrix_a = tf.data.experimental.get_single_element(dataset)
matrix_a = matrix_a.numpy()


dataset2 = tf.data.FixedLengthRecordDataset(imagpart,record_bytes=num_bytes * num_cols)
dataset2 = dataset2.map(lambda s: tf.reshape(tf.io.decode_raw(s, dtype, little_endian=False), (num_cols,)),num_parallel_calls=8)
dataset2 = dataset2.batch(num_cols)
matrix_b = tf.data.experimental.get_single_element(dataset2)
matrix_b = np.multiply(matrix_b, 1j)
matrix_a = np.transpose(matrix_a)
matrix_b = np.transpose(matrix_b)
matrix_c = matrix_a + matrix_b

ZZ = matrix_c

# ================================
matrix_a = None
matrix_b = None
matrix_c = None
mempool.free_all_blocks()
pinned_mempool.free_all_blocks()

RR = cp.real(ZZ)
RR = cp.ascontiguousarray(RR)
cp.cuda.Stream.null.synchronize()
XX = cp.imag(ZZ)
ZZ = cp.asnumpy(ZZ)

RR = cp.asnumpy(RR)
UU, SS, VV = np.linalg.svd(RR,full_matrices=True,compute_uv=True)

VV = None
UU = cp.array(UU)
SS = cp.array(SS)

si = cp.size(SS)
u11 = SS[0:si]
u11 = 1 / cp.sqrt(u11)
u11 = cp.diag(u11)

# ====   A=transpose(UU)*XX*UU    ====

uut = cp.transpose(UU)
uut = cp.ascontiguousarray(uut)

uutx = cp.matmul(uut,XX)
A = cp.matmul(uutx,UU)
uutx = None
# ================================

#slicing A
# Top left
A11= A[:si, :si]
# Top right
A12 = A[:si, si:]
# Bottom left
A21 = A[si:, :si]
# Bottom right
A22 = A[si:, si:]

# ======   B=(u11)*(A11-A12*inv(A22)*transpose(A12))*(u11)   ======
A = None

mempool.free_all_blocks()
pinned_mempool.free_all_blocks()

A4 = cp.matmul(A11,u11)
B = cp.matmul(u11,A4)
A4 = None
# ================================
B = cp.asnumpy(B)
UB, SB, HB = np.linalg.svd(B,full_matrices=True,compute_uv=True)
UB = None
SB = None
HB= cp.array(HB)
B = None
# ====    VB=UU*[eye(si);-inv(A22)*transpose(A12)]*u11*HB   ======
mempool.free_all_blocks()
pinned_mempool.free_all_blocks()

A22 = None
A12 = None

eyesi =  cp.eye(si)
VB1= cp.matmul(UU,eyesi)
VB2 = cp.matmul(u11,HB.conj().T)
HB  = None
VB = cp.matmul(VB1,VB2)

# ================================
VB1 = None
VB2 = None

mempool.free_all_blocks()
pinned_mempool.free_all_blocks()

IC= cp.fliplr(VB)

DD =  np.zeros((1,si))
ZZ = cp.array(ZZ)
for jm in range(0,num_modes):
	DD[0,jm] =  cp.imag(cp.matmul(cp.transpose(IC[:,jm]),cp.matmul(ZZ,IC[:,jm])))
IC = cp.asnumpy(IC)

pd.DataFrame(IC).to_csv(outpath + '/GPUIC.csv',header=None,index=None)
pd.DataFrame(DD).to_csv(outpath + '/GPUDD.csv',header=None,index=None)

quit()
