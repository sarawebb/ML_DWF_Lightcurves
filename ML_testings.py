import numpy as np 
import scipy as sp 
from scipy.stats import kurtosis
import pandas as pd 
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GM
import time
import os 
import seaborn as sns
import matplotlib.pyplot as plt

###### ------------- FUNCTIONS --------------------------

def lc_df(in_path):
	print('Initialzing...')
	t0 = time.time()
	std_list = []
	max_list = []
	min_list = []
	dur_list = []
	ran_list = []
	num_list = []
	ave_list = []
	kur_list = []
	ind_list = []
	ast_list = []
	fld_list = []
	fil_list = []
	j=0 
	lc_dict = {'STD' : std_list, 'MAX' : max_list, 'MIN' : min_list, 'DUR' : dur_list, 'RAN' : ran_list, 'NUM' : num_list, 'AVE' : ave_list, 'KUR': kur_list, 'IND' : ind_list }
	for filename in os.listdir(in_path):
		if filename.endswith('.lc'):
			mjd = []
			mag= []
			emag = []
			ulmag = []
			file_dir = in_path + filename
			fp_i= open(file_dir)
			header = fp_i.readline()
			#header2 = fp_i.readline() 
			for row in fp_i: 
				row = row.strip()
				columns = row.split()
				mjd.append(float(columns[:][0]))
				mag.append(float(columns[:][1]))
				emag.append(float(columns[:][2]))
				ulmag.append(float(columns[:][3]))
		try:
			MAX =np.nan_to_num(np.max(mag))
			DUR = np.nan_to_num((np.max(mjd))-(np.min(mjd)))
		except ValueError:
			fil_list.append(('Value Error', filename))
		if MAX== 0.0:
			fil_list.append(('no readings', filename))
				
		else: 
			try:
				MIN = np.nan_to_num(np.min([i for i in mag if i > 0]))
				RAN =  np.nan_to_num(np.max(mag) - MIN)
				AVE = np.nan_to_num(np.average(emag))
				KUR = np.nan_to_num(kurtosis(mag))
				STD = np.nan_to_num(np.std(mag))
				
			except ValueError:
				fil_list.append(('Value Error', filename))
			if filename[0] == 'A':
				fld_list.append(1)
			else: 
				fld_list.append(0)
				min_list.append(MIN)
				max_list.append(MAX)
				dur_list.append(DUR)
				ran_list.append(RAN)
				ave_list.append(AVE)
				kur_list.append(KUR)
				std_list.append(STD)
				ind_list.append(filename)
				num_list.append(j)
				#ast_list  here if we have one 
				j +=1
		fp_i.close()
	#print("TESTING") 
	#print(len(fld_list))
	DF = pd.DataFrame(lc_dict) 
	return DF 

def gauss_cluster_lc(in_path):
	X_lc = lc[['AVE', 'KUR', 'MAX', 'MIN', 'RAN', 'STD']]	
	y_lc = lc['NUM']	
	model = PCA(n_components = 2)
	model.fit(X_lc)
	X_2D = model.transform(X_lc)		
	lc['PCA1'] = X_2D[:, 0]
	#print(lc['PCA1'])
	lc['PCA2'] = X_2D[:, 1]	
	
	model = GM(n_components=6, covariance_type='full')
	model.fit(X_lc)
	y_gmm = model.predict(X_lc)
	lc['cluster']= y_gmm
	df = pd.DataFrame(lc)
	sdf = df.sort_values(by=['RAN'])
	return sdf 

#---------------------------- DBSCAN CLUSTERING -------------------------

lc = lc_df('/Users/swebb/Documents/ML/all_lc/')
X_lc = lc[['AVE', 'KUR', 'MAX', 'MIN' , 'RAN', 'STD']]
print(X_lc)
print('Completing PCA projection...') 
model = PCA(n_components = 2)
print(model)
model.fit(X_lc)
X_2D = model.transform(X_lc)
#print(X_2D)
lc['PCA1'] = X_2D[:, 0]
#print(lc['PCA1'])
lc['PCA2'] = X_2D[:, 1]
#fig = plt.plot(x="PCA1", y="PCA2")
#plt.savefig('testing.png') 		

data_lc= X_2D[:,:]
target_lc =X_2D[:,1]
db = DBSCAN(algorithm = 'auto', eps=1.1, min_samples =10, n_jobs=2).fit(data_lc)
print(db)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
		for each in np.linspace(0, 1, len(unique_labels))]	
		
for k, col in zip(unique_labels, colors):
	if k == -1:
		col = [0,0,0,1]		
	class_member_mask = (labels == k)
	
	xy = data_lc[class_member_mask & core_samples_mask]
	plt.plot(xy[:,0], xy[:, 1], 'o', markerfacecolor = tuple(col), markeredgecolor= 'k', markersize=12)
	
	xy = data_lc[class_member_mask & ~core_samples_mask]
	plt.plot(xy[:,0], xy[:, 1], 'o', markerfacecolor = tuple(col), markeredgecolor= 'k', markersize=6)		
					
print('Completing DBSCAN clustering... ')
plt.title('Prime, Antlia, FRB131104 field LCs: Estimated number of clusters: %d' % n_clusters_)	
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()			



#testing = gauss_cluster_lc('PATH')
#print(testing)
					
					
					
					
					
					
					
					
					
					
					
					
					
					
					
					
