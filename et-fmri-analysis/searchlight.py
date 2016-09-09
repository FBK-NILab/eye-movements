#!/usr/bin/python

from mvpa2.tutorial_suite import *
import numpy as np
import sys
import glob
import time

#%% IO Block
tstart=time.time()
# CLI
myargs=sys.argv

part_ID=myargs[1]
radius=int(myargs[2])

the_idxs=np.loadtxt("./data/intext.txt")

mo_list=np.sort(glob.glob("./results/preproc/"+part_ID+"/moco."+part_ID+"_ses-movie_task-movie_run-*.nii.gz"))

thetargets=np.zeros(3599,dtype="str")
thetargets[:]="R"
thetargets[the_idxs[:,2]==100]="E"
thetargets[the_idxs[:,2]==200]="I"

mo_data=vstack([fmri_dataset(mo_list[i],mask="./results/asandbox/sub-01-automask.nii.gz") for i in np.arange(len(mo_list))])

mo_data.targets=thetargets
mo_data.sa['chunks']=list(np.hstack([np.repeat(1,451),np.repeat(2,441),np.repeat(3,438),np.repeat(4,488),np.repeat(5,462),np.repeat(6,439),np.repeat(7,542),np.repeat(8,338)]))

mo_data.sa['targets']=thetargets
mo_data.sa['scene']=the_idxs[:,1]
mo_data.sa['sync']=the_idxs[:,4]
mo_data.sa['length']=the_idxs[:,3]

# preprocessing

detrender=PolyDetrendMapper(polyord=2,chunks_attr='chunks')
detrender.train(mo_data)
d_mo_data=mo_data.get_mapped(detrender)

zscorer=ZScoreMapper(chunks_attr='chunks')
zscorer.train(d_mo_data)
zd_mo_data=d_mo_data.get_mapped(zscorer)



# Encoding
the_enc=np.zeros((80,mo_data.shape[1]))
the_starting_vol=np.zeros(80)
n=0
for s in np.unique(mo_data.sa.scene)[1:]:
	
	curr_len=zd_mo_data[zd_mo_data.sa.scene==s,:].shape[0]
	curr_scene=np.unique(zd_mo_data[zd_mo_data.sa.scene==s,:].sa.scene)
	the_starting_vol[n]=np.argwhere(zd_mo_data.sa.scene==s)[0]
	if curr_len==3:
		the_enc[n,:]=zd_mo_data[zd_mo_data.sa.scene==s,:].samples[0:].mean(axis=0)
		print n
		n=n+1
	if curr_len==4:
		the_enc[n,:]=zd_mo_data[zd_mo_data.sa.scene==s,:].samples[1:].mean(axis=0)
		print n
		n=n+1
	if curr_len==5:
		the_enc[n,:]=zd_mo_data[zd_mo_data.sa.scene==s,:].samples[2:].mean(axis=0)			
		print n
		n=n+1
	if curr_len >= 6:
		print n
		the_enc[n,:]=zd_mo_data[zd_mo_data.sa.scene==s,:].samples[3:6].mean(axis=0)		
		n=n+1

enc_zd_mo_data=zd_mo_data[the_starting_vol.astype("int"),:]
enc_zd_mo_data.samples=the_enc
enc_zd_mo_data.fa['mean_activity']=the_enc[0,:]

		###
clf = LinearNuSVMC()
cv=CrossValidation(clf, NFoldPartitioner())
center_ids=enc_zd_mo_data.fa.mean_activity.nonzero()[0]
sl = sphere_searchlight(cv, radius=radius, space='voxel_indices',center_ids=center_ids,postproc=mean_sample())
ds = enc_zd_mo_data.copy(deep=False,sa=['targets', 'chunks'],fa=['voxel_indices'],a=['mapper'])

sl_map = sl(ds)
sl_map.samples *= -1
sl_map.samples += 1

out_m=np.hstack([np.array(ds.fa['voxel_indices']),sl_map.samples.T])
np.savetxt("./results/searchlights/dumps/sl_sub-"+part_ID+"."+str(radius)+".1D",out_m)
