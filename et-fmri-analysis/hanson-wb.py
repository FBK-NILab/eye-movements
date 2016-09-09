#!/usr/bin/python

import sys
import numpy as np
import scipy.stats as spst
#from mvpa2.tutorial_suite import *


myargs=sys.argv
scene_ID=int(myargs[1])-1

partlist=np.loadtxt("./pipeline/part_list",dtype="str")
partlist=partlist[:6]

scene_list=np.loadtxt("./pipeline/hanson_scene_list.txt")
theidxs=np.loadtxt("./pipeline/mask_dump.1D")[:,:3]
run_ID=int(scene_list[scene_ID,1])
scene_start=int(scene_list[scene_ID,2])
scene_length=int(scene_list[scene_ID,3])
area_extent=228483

thedata=np.zeros((len(partlist),area_extent,scene_length))

for p in np.arange(len(partlist)):

        curr_filename="./results/whole-brain-patterns/"+partlist[p]+"."+str(run_ID)+".1D"
        print curr_filename
        thedata[p,:,:]=np.loadtxt(curr_filename)[:,(3+scene_start):(3+scene_start+scene_length)]
#       =curr_data

#       print curr_filename,thedata.shape

print "IO done"
evmap=np.zeros(area_extent)
fmap=np.zeros(area_extent)
pmap=np.zeros(area_extent)

the_factor=scene_length/float(max(len(partlist),scene_length))

for v in np.arange(area_extent):

        currs=np.linalg.svd(np.cov(thedata[:,v,:]))[1]
        curr_cov=np.cov(thedata[:,v,:])
        print "I am here"
        if np.sign(currs).sum()==len(partlist):
                evmap[v]=currs[0]
                fmap[v]=currs[0]*the_factor
                pmap[v]=spst.f.pdf(evmap[v],the_factor,scene_length)


#print "SVD done"
#np.savetxt("./results/svd/dumps/scene-"+str(scene_ID+1)+".svd.txt",np.vstack([theidxs.T,evmap]).T)
#np.savetxt("./results/svd/dumps/scene-"+str(scene_ID+1)+".fmap.txt",np.vstack([theidxs.T,fmap]).T)
#np.savetxt("./results/svd/dumps/scene-"+str(scene_ID+1)+".pmap.txt",np.vstack([theidxs.T,pmap]).T)
