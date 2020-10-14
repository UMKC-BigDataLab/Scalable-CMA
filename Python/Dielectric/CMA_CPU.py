import tensorflow as tf
import numpy as np
import time
import sys
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

num_bytes = 8
dtype = tf.float64

realpart = sys.argv[1]
imagpart = sys.argv[2]
num_cols = int(sys.argv[3])
outpath = sys.argv[4]

dataset = tf.data.FixedLengthRecordDataset(realpart,record_bytes=num_bytes * num_cols)
dataset = dataset.map(lambda s: tf.reshape(tf.io.decode_raw(s, dtype, little_endian=False), (num_cols,)))
dataset = dataset.batch(num_cols)
matrix_a = tf.data.experimental.get_single_element(dataset)

dataset2 = tf.data.FixedLengthRecordDataset(imagpart,record_bytes=num_bytes * num_cols)
dataset2 = dataset2.map(lambda s: tf.reshape(tf.io.decode_raw(s, dtype, little_endian=False), (num_cols,)))
dataset2 = dataset2.batch(num_cols)

matrix_b = np.multiply(tf.data.experimental.get_single_element(dataset2), 1j)
matrix_a = np.transpose(matrix_a)
matrix_b = np.transpose(matrix_b)
matrix_c = matrix_a + matrix_b


Z_EE = matrix_c[:int(num_cols/2), :int(num_cols/2)]
Z_EH = matrix_c[:int(num_cols/2), int(num_cols/2):]
Z_HE = matrix_c[int(num_cols/2):, :int(num_cols/2)]
Z_HH = matrix_c[int(num_cols/2):, int(num_cols/2):]

ZZ = Z_EE - np.matmul(np.matmul(Z_EH,np.linalg.inv(Z_HH)),Z_HE)


RR = np.real(ZZ)
XX1 = np.imag(ZZ)
UU, SS, VV = np.linalg.svd(RR,full_matrices=True,compute_uv=True)


XX = np.ascontiguousarray(XX1)
si = len(SS)
SS1 = SS[0:si]
SS1  = 1 / np.sqrt(SS1)
u11 = np.diag(SS1)

uut = np.transpose(UU)


uut1 = np.ascontiguousarray(uut)
uutx = np.matmul(uut1,XX)

A = np.matmul(uutx,UU)

#slicing A
# Top left
A11= A[:si, :si]
# Top right
A12 = A[:si, si:]
# Bottom left
A21 = A[si:, :si]
# Bottom right
A22 = A[si:, si:]

#print(A22)
B1 = A11 - np.matmul(np.matmul(A12,np.linalg.inv(A22)),np.transpose(A12))
B = np.matmul(u11,np.matmul(B1,u11))


UB, SB, HB = np.linalg.svd(B,full_matrices=True,compute_uv=True)


VB1 = np.matmul((np.linalg.inv(A22) * -1),np.transpose(A12))
combined_VB1 = np.concatenate((np.eye(si),VB1))
VB = np.matmul(np.matmul(np.matmul(UU,combined_VB1),u11),HB.conj().T)


IC =  np.zeros((si,int(num_cols/2)))
DD =  np.zeros((1,int(num_cols/2)))

IC= np.fliplr(VB)


for jm in range(0,int(num_cols/2)):
	DD[0,jm] =  np.imag(np.matmul(np.transpose(IC[:,jm]),np.matmul(ZZ,IC[:,jm])))


pd.DataFrame(IC).to_csv(outpath + '/CPUIC.csv',header=None,index=None)
pd.DataFrame(DD).to_csv(outpath + '/CPUDD.csv',header=None,index=None)

quit()
