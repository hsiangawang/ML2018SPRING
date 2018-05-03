import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

imagenpy_path = sys.argv[1]
test_path = sys.argv[2]
predict_path = sys.argv[3]

test_data = pd.read_csv(test_path)

img = np.load(imagenpy_path)
img = img.astype('float32')/255 
img_pca = PCA(n_components=300, whiten=True , svd_solver='auto', random_state=0).fit_transform(img) #300 , don't use fit
print(img_pca.shape)

kmeans = KMeans(n_clusters=2,random_state=0).fit(img_pca)

IDs, Idx1, Idx2 = np.array(test_data['ID']),np.array(test_data['image1_index']),np.array(test_data['image2_index'])
o = open(predict_path,'w')
o.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs,Idx1,Idx2):
	p1 = kmeans.labels_[i1]
	p2 = kmeans.labels_[i2]

	if p1 == p2:
		predict = 1
	else:
		predict = 0
	o.write("{},{}\n".format(idx,predict))
o.close()