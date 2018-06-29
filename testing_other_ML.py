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
		
			MJD, MAG, EMAG, ULMAG = np.loadtxt(file_dir, unpack= True)
			#print mjd
			#print mag 
			#print emag 
			#print ulmag 
			mjd = MJD
			mag = MAG 
			emag = EMAG
			ulmag = ULMAG
			
			first_obs = np.nan_to_num(np.min(mjd))
			first_obs_p1 = first_obs + 1
			
			for i  in MJD:
				if i < first_obs_p1: 
					#print 'testing'
					MAG_D1 = []
					EMAG_D1 = []
					ULMAG_D1 = []
					MJD_D1 = []
					MAG_D1.append(mag)
					EMAG_D1.append(emag)
					ULMAG_D1.append(ulmag)
					MJD_D1.append(mjd)
					
					
			#return MAG_D1
			
			
			try: 
				MAX = np.nan_to_num(np.max(MAG_D1))
				
				#print first_obs
				
			except  ValueError:
				fil_list.append(('Value Error', filename))
				

			#print MAG_D1 
			
			if MAX == 0.0:
				fil_list.append(('no readings', filename))
			else:	
			
				try:
					MIN = np.nan_to_num(np.min([i for i in mag if i < first_obs_p1]))
					RAN = np.nan_to_num(np.max(MAG_D1)-MIN)
					AVE = np.nan_to_num(np.average(MAG_D1))
					KUR = np.nan_to_num(kurtosis(MAG_D1))
					STD = np.nan_to_num(np.std(MAG_D1))
					DUR = np.nan_to_num((np.max(MJD_D1))-(np.min(MJD_D1)))
					#print(MIN)
				except ValueError:
					fil_list.append(('Value Error', filename))
				
			#print MAG_D1		
			#print(min_list)		
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
			j += 1 	
			
					
		
		#print len(min_list), len(max_list), len(fld_list), len(ran_list), len(ave_list), len(kur_list), len(std_list), len(ind_list), len(num_list), len(dur_list)
				
		#print MIN, MAX, AVE, min_list
	DF = pd.DataFrame(lc_dict)
	return DF
	print(DF)
				
lc = lc_df('/Users/swebb/Documents/ML/')
X_lc = lc[['AVE', 'KUR', 'MAX', 'MIN' , 'RAN', 'STD']]
#print X_lc 
print('PCA Projection....')
model = PCA(n_components = 2) 
model.fit(X_lc)
X_2D = model.transform(X_lc)
#print X_2D
lc['PCA1'] = X_2D[:, 0]
lc['PCA2'] = X_2D[:, 1]

#plt.plot(lc['PCA1'], lc['PCA2']) 
#plt.show()



############----------------------------------Compute DBSCAN --------------------------------####################################
data_lc = X_2D[:,:]
db = DBSCAN(algorithm = 'auto', eps=1.1, min_samples =10, n_jobs=2).fit(data_lc)

core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_




##########------------ Number of clusters in labels, igoring any noise presents ------------- #############
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0 )
print('Estimated number of clusters: %d'  %n_clusters_)




##########-------------Plot results -----------------------------------------------------------##########################
unique_labels = set(labels) 
colors = [plt.cm.spectral(each) for each in np.linspace(0, 1, len(unique_labels))]


for k, col in zip(unique_labels, colors):
	if k == -1:
		col = [0,0,0,1] #black is used for noise
		
	
	class_member_mask = (labels == k)
		
	xy = data_lc[class_member_mask & core_samples_mask]
	plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor = tuple(col), markeredgecolor = 'k', markersize = 12)
		
	xy = data_lc[class_member_mask & ~core_samples_mask]
	plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor = tuple(col), markeredgecolor = 'k', markersize =2) 

#print('completing DBSCAN clustering...')
plt.title('Antlia field LCs: Estimated number of clusters: %d' % n_clusters_)	
plt.show()

#print db 
				
	
	
			
#testing = lc_df('/Users/sara/Documents/test/')
