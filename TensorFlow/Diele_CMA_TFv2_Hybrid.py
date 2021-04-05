import sys
import tensorflow as tf
import numpy as np
import time
import os
import pandas as pd


print("Starting...")
print(time.ctime())

num_bytes = 8
dtype = tf.float64
inputpath =  sys.argv[1]
realpart = inputpath + '/realpart.csv.bin'
imagpart = inputpath + '/imagpart.csv.bin'
freq_num = inputpath.split("/")[-1]
num_cols = int(sys.argv[2])
outpath = sys.argv[3]

with tf.device("cpu:0"):	
	dataset = tf.data.FixedLengthRecordDataset(realpart,record_bytes=num_bytes * num_cols)
	dataset = dataset.map(lambda s: tf.reshape(tf.io.decode_raw(s, dtype, little_endian=False), (num_cols,)))
	dataset = dataset.batch(num_cols)
	matrix_a = tf.data.experimental.get_single_element(dataset)
print("A done")
print(time.ctime())

with tf.device("cpu:0"):
	dataset2 = tf.data.FixedLengthRecordDataset(imagpart,record_bytes=num_bytes * num_cols)
	dataset2 = dataset2.map(lambda s: tf.reshape(tf.io.decode_raw(s, dtype, little_endian=False), (num_cols,)))
	dataset2 = dataset2.batch(num_cols)
	matrix_b = tf.data.experimental.get_single_element(dataset2) 
print("B done")
print(time.ctime())

with tf.device("gpu:0"):
	matrix_a = tf.transpose(matrix_a)
	matrix_b = tf.transpose(matrix_b)

print("transpose done")
print(time.ctime())

with tf.device("gpu:0"):
	matrix_c = tf.complex(matrix_a,matrix_b)
print("C Done")
print(time.ctime())

with tf.device("gpu:0"):
	ZEE = tf.slice(matrix_c,[0,0],[int(num_cols/2),int(num_cols/2)])
	ZEH = tf.slice(matrix_c,[0,int(num_cols/2)],[int(num_cols/2),int(num_cols/2)])
	ZHE = tf.slice(matrix_c,[int(num_cols/2),0],[int(num_cols/2),int(num_cols/2)])
	ZHH = tf.slice(matrix_c,[int(num_cols/2),int(num_cols/2)],[int(num_cols/2),int(num_cols/2)])
print ('slicing Done')
print(time.ctime())

with tf.device("gpu:0"):
	ZHHinv = tf.linalg.inv(ZHH)
print ('ZHHinv Done')
print(time.ctime())

with tf.device("gpu:0"):
	Z1 = tf.matmul(ZEH,ZHHinv)
print ('multi Done')
print(time.ctime())

with tf.device("gpu:0"):
	Z2 = tf.matmul(Z1,ZHE)
print ('multi Done')
print(time.ctime())

with tf.device("gpu:0"):
	ZZ = ZEE - Z2
print ('- Done')
print(time.ctime())

with tf.device("gpu:0"):
	RR = tf.math.real(ZZ)	
	XX = tf.math.imag(ZZ)
print ("real imag done")
print(time.ctime())

with tf.device("cpu:0"):
	SS, UU, VV  = tf.linalg.svd(RR,full_matrices=True,compute_uv=True)

print ('SVD(RR)')
print(time.ctime())

with tf.device("gpu:0"):
	UUt = tf.transpose(UU)
print ('transpose done')
print(time.ctime())

si = tf.size(SS)

with tf.device("gpu:0"):
	UUXX = tf.matmul(XX,UU)
print ('multi done')
print(time.ctime())

with tf.device("gpu:0"):
	eyesi = tf.eye(si,dtype=tf.dtypes.float64)
print ('eyesi done')
print(time.ctime())

with tf.device("gpu:1"):
	A = tf.matmul(UUt,UUXX)
print ('multi done')
print(time.ctime())

with tf.device("gpu:1"):
	u11 = tf.slice(SS,[0],[si])
print ('u11 done')
print(time.ctime())

with tf.device("gpu:1"): 
	u11 = tf.where(u11 != 0, 1 /tf.sqrt(u11), u11)

print("u11 sqrt done")
print(time.ctime())

with tf.device("gpu:1"):
        u11 = tf.compat.v2.linalg.diag(u11)
print("u11 diag done")
print(time.ctime())

with tf.device("gpu:1"):
	fullsize = tf.shape(A)[0]
	A11 = tf.slice(A,[0,0],[si,si])
	A12 = tf.slice(A,[0,si-1],[si,fullsize - si])
	A21 = tf.slice(A,[si-1,0],[fullsize - si,si])
	A22 = tf.slice(A,[si-1,si-1],[fullsize - si,fullsize - si])
print("Slicing A done")
print(time.ctime())

with tf.device("gpu:1"):
	A12t = tf.transpose(A12)
print ('transpose done')
print(time.ctime())

with tf.device("gpu:1"):
	invA22 = tf.linalg.inv(A22)
print("invA22")
print(time.ctime())

with tf.device("gpu:1"):
	p14_res = tf.matmul(invA22,A12t)
print("multi done")
print(time.ctime())

with tf.device("gpu:2"):
	neginvInvA22 = tf.math.negative(invA22)
print("neginvInvA22 done")
print(time.ctime())

with tf.device("gpu:2"):
	p15a_res = tf.matmul(A12,p14_res)
print("multi done")
print(time.ctime())

with tf.device("gpu:2"):
	p15b_res = tf.matmul(neginvInvA22,A12t)
print("multi done")
print(time.ctime())

with tf.device("gpu:2"):
	p16a_res = tf.math.subtract(A11,p15a_res)
print("- done")
print(time.ctime())

with tf.device("gpu:2"):
	if (tf.shape(p15b_res)[0] == 0):
		p16b_resApp = eyesi
	else:
		p16b_resApp = tf.concat([eyesi,p15b_res],0)
print("p16b_resApp done")
print(time.ctime())

with tf.device("gpu:2"):
	p17a_res = tf.matmul(p16a_res,u11)
print("multi done")
print(time.ctime())

with tf.device("gpu:2"):
	p17b_res= tf.matmul(UU,p16b_resApp)
print("multi done")
print(time.ctime())

with tf.device("gpu:2"):
	B = tf.matmul(u11,p17a_res)
print("multi done")
print(time.ctime())

with tf.device("cpu:0"):
	SB, UB, HB  = tf.linalg.svd(B,full_matrices=True,compute_uv=True)
print ('SVD(B) done')
print(time.ctime())

with tf.device("gpu:1"):
	p20_res = tf.matmul(u11,HB)
print("multi done")
print(time.ctime())

with tf.device("gpu:3"):
	VB = tf.matmul(p17b_res,p20_res)
print("multi done")
print(time.ctime())

with tf.device("gpu:3"):
	IC = tf.reverse(VB,[1])
print ('flip done')
print(time.ctime())

DD =  tf.constant([],tf.float64)
with tf.device("gpu:1"):
	ICcomp = tf.dtypes.cast(IC, tf.complex128)

with tf.device("gpu:3"):
	for jm in range(0,si):
		a  =  tf.math.imag(tf.tensordot(tf.transpose(ICcomp[:,jm]),tf.linalg.matvec(ZZ,ICcomp[:,jm]),1))
		a = tf.reshape(a, [1])		
		DD =  tf.concat([DD,a],-1) 
	

print("DD done")
print ('for loop multi done')
print(time.ctime())

ICnp = IC.numpy()
DD =  DD.numpy()

pd.DataFrame(ICnp).to_csv(outpath + freq_num + '/TFICHybrid.csv',header=None,index=None)
pd.DataFrame(DD).to_csv(outpath + freq_num + '/TFDDHybrid.csv',header=None,index=None)

print ('Writing done')
print(time.ctime())

quit()
