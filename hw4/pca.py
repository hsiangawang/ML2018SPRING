import sys
import os
import skimage.io
import numpy as np
from os import listdir
from skimage import transform

image_path = sys.argv[1]
Processed_image = sys.argv[2]

def load_images_from_path(image_path):
	All_faces = []
	for filename in os.listdir(image_path):
		single_face = skimage.io.imread(os.path.join(image_path,filename))
		#single_face = transform.resize(single_face,(100,100,3))
		if single_face is not None:
			single_flatten = single_face.flatten()
			All_faces.append(single_flatten)
	return All_faces


temp = load_images_from_path(image_path)
temp = np.array(temp,dtype='float32')


## find U,s,V
# M:1080000 N=415


X_mean = np.mean(temp,axis=0)
X = temp - X_mean

#skimage.io.imsave("average.jpg",(X_mean.reshape(600,600,3))/255)

U,s,V = np.linalg.svd(X.T,full_matrices=False)

#sum = 0
#for i in range(s.shape[0]):
#	sum += s[i]
#print("first eigenvalue ratio: ",s[0]/sum)
#print("first eigenvalue ratio: ",s[1]/sum)
#print("first eigenvalue ratio: ",s[2]/sum)
#print("first eigenvalue ratio: ",s[3]/sum)

###
#tenth_eigen = U[ : , 0]
#tenth_eigen = np.negative(tenth_eigen)
#print("tenth_eigen shape",tenth_eigen.shape)
#tenth_eigen += X_mean
#tenth_eigen -= np.min(tenth_eigen)
#tenth_eigen /= np.max(tenth_eigen)
#tenth_eigen = (tenth_eigen*255).astype(np.uint8)
#skimage.io.imsave("eigen_1.jpg",tenth_eigen.reshape(600,600,3))
###

test_img = skimage.io.imread(os.path.join(image_path,Processed_image))
test_img_flatten = test_img.flatten()

Y = (test_img_flatten.T - X_mean).T
print("test_X shape: ", Y.shape)

eigen_face = U[ : , : 4].T
print("eigen face shape: ", eigen_face.shape)

weight = np.dot(eigen_face,Y)
print("weight shape",weight.shape)
print(weight)

reconstruct_img = np.dot(eigen_face.T,weight)
reconstruct_img += X_mean
reconstruct_img -= np.min(reconstruct_img)
reconstruct_img /= np.max(reconstruct_img)
reconstruct_img = (reconstruct_img*255).astype(np.uint8)

skimage.io.imsave("reconstruction.jpg",reconstruct_img.reshape(600,600,3))







