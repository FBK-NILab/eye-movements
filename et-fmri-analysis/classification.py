#!/usr/bin/python

import sys
import numpy as np
from mvpa2.tutorial_suite import *


myargs=sys.argv
area_ID=myargs[1]
thedata=list()


# I / O

partlist=np.loadtxt("./pipeline/part_list",dtype="str")
thelabels=np.loadtxt("./data/intext.txt")

for p in np.arange(len(partlist)):
#for p in np.arange(1):
	currbasename="./results/patterns/"+area_ID+"."+partlist[p]
#	print currbasename
	thedata.append(dataset_wizard(np.hstack([np.loadtxt(currbasename+"."+str(r)+".txt") for r in np.arange(1,9)]).T,targets=thelabels[:,2]))
	thedata[p].sa['chunks']=list(np.hstack([np.repeat(1,451),np.repeat(2,441),np.repeat(3,438),np.repeat(4,488),np.repeat(5,462),np.repeat(6,439),np.repeat(7,542),np.repeat(8,338)]))
	thedata[p].sa['participant']=list(np.repeat(p,3599))
	thedata[p].sa['isascene']=list(thelabels[:,0])
	thedata[p].sa['sceneID']=list(thelabels[:,1])
	thedata[p].sa['scenelength']=list(thelabels[:,3])
	thedata[p].sa['sync']=list(thelabels[:,4])


print("I/O DONE")



## 1. Sample selection

### 1.1 Discard all the samples within non - relevant scenes

thedata=[sd[sd.targets!=0] for sd in thedata]

meandata=list()
for p in np.arange(len(partlist)):
	currmean=np.zeros((80,thedata[p].shape[1]))
	currattr=np.zeros((80,3))
	n=0
	for s in np.unique(thedata[0].sa.sceneID):
		
		currlength=np.unique(thedata[p].sa.scenelength[np.argwhere(thedata[0].sa.sceneID==s).flatten(1)])[0].astype("int")
	
		curridxs=np.argwhere(thedata[p].sa.sceneID==s).flatten(1)[np.random.permutation(currlength)][:3]
		
		currmean[n,:]=np.mean(thedata[p][curridxs,:],axis=0)

		currattr[n,0]=np.unique(thedata[p].targets[np.argwhere(thedata[0].sa.sceneID==s).flatten(1)])[0]

		currattr[n,1]=np.unique(thedata[p].sa.sync[np.argwhere(thedata[0].sa.sceneID==s).flatten(1)])[0]
	
		currattr[n,2]=np.unique(thedata[p].sa.chunks[np.argwhere(thedata[0].sa.sceneID==s).flatten(1)])[0]
		n=n+1

	meandata.append(dataset_wizard(currmean,targets=currattr[:,0]))
	meandata[p].sa['sync']=currattr[:,1]
	meandata[p].sa['participant']=list(np.repeat(p,80))
	meandata[p].sa['chunks']=currattr[:,2]
		

## Classification - No sync

### Classification - OP
### Within Participant
clf=LinearCSVMC()
cv = CrossValidation(clf,NFoldPartitioner(attr='chunks'),errorfx=mean_match_accuracy)
all_wsc=[cv(sd) for sd in meandata]

### Between Participants
clf=LinearCSVMC()
cv = CrossValidation(clf,NFoldPartitioner(attr='participant'),errorfx=mean_match_accuracy)
all_bsc=[cv(vstack(meandata))]


## Introduce synchrony

msync=[sd[sd.sa.sync==1000] for sd in meandata]
lsync=[sd[sd.sa.sync==500] for sd in meandata]

## Classification

clf=LinearCSVMC()
cv = CrossValidation(clf,NFoldPartitioner(attr='chunks'),errorfx=mean_match_accuracy)

msync_wsc=[cv(sd) for sd in msync]
lsync_wsc=[cv(sd) for sd in lsync]

cv = CrossValidation(clf,NFoldPartitioner(attr='participant'),errorfx=mean_match_accuracy)
msync_bsc=[cv(vstack(msync))]
lsync_bsc=[cv(vstack(lsync))]


## Saving the results

### Nosync

#### Within Participants
np.savetxt("./results/classifications/"+area_ID+".all_wsc.txt",np.vstack(all_wsc).reshape(15,8).T)

#### Between Participants
np.savetxt("./results/classifications/"+area_ID+".all_bsc.txt",np.vstack(all_bsc))

### More sync

#### Within Participants
np.savetxt("./results/classifications/"+area_ID+".msync_wsc.txt",np.vstack(msync_wsc).reshape(15,8).T)

#### Between Participants
np.savetxt("./results/classifications/"+area_ID+".msync_bsc.txt",np.vstack(msync_bsc))


### Less sync
#### Within Participants
np.savetxt("./results/classifications/"+area_ID+".lsync_wsc.txt",np.vstack(lsync_wsc).reshape(15,8).T)

#### Between Participants
np.savetxt("./results/classifications/"+area_ID+".lsync_bsc.txt",np.vstack(lsync_bsc))
