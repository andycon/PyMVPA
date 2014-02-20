# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Functional connectivity clustering based modelled on Afni's InstaCorr
"""

__docformat__ = 'restructuredtext'


import numpy as np
import scipy.stats as stats
import sys
from mvpa2.suite import *


if __debug__:
    from mvpa2.base import debug

__all__ = [ "Clustering" ]

def group_insta_clust(ds,th=90,crit='pct',overlap=False,
                score='xsub_max',xsubp=.0001,min_nodes=100,max_leftover=100,econ=True):
    """
    "InstaClust" Clustering algorithm.

    The InstaClust algorithm uses the group statistical (T) map based on the 
    group analysis of connectivity maps for dataset (ds) which contains all 
    subjects (aligned some common space) to select a set of clusters based on 
    the seed features with the highest 'scores'. 
    
    Scores are defined by default as the mean cross-subject
    correlation between data vectors at each feature. For example, the mean
    correlation across subjects for the searchlight dissimiliarty matrix at each
    node. Data vectors can be from any source (i.e., raw-time series, Beta
    coefficients, etc.). 

    At each iteration, the feature with the highest score becomes the "seed" for
    the group connectivity analysis. All features above the threshold criterion
    based on group T-map are selected for inclusion in the cluster for that
    iteration. By default, the threshold is the top 10 percent of elgible
    (unclustered) features.

    After a feature has been clustered it is no longer eligble to be used as a
    seed. By default, clusters also may not overlap, i.e., no two clusters
    may share a common feature.

    Parameters
    ----------
    ds : Dataset 
        This is a 3-dimensional dataset with subjects x features x feature-specific 
        patterns (e.g.searchlight DSMs)
    th : int or float
        Threshold to be applied in calculating the scores for each node
    crit : str 
        The criterion for thresholding the tmap to form clusters.
        Options:
            'pct' (default) for percentile, thresholds the tmap for a chosen
                seed keeping just the features with score >= the score at the top th
                percentile (where 0 < th 100)
            't' Threshold the tmap using the t-values >= th 
            'p' Threshold tmap using p-values <= th (0 < th < 1)
    overlap : bool 
        If False (default) clusters may not overlap, else, clusters may overlap
    singlevol : bool
        If True (default) return results in a single volume. If overlap is true
        singlevol will be ignored and multiple volumes are returned.
    score : str
        Method used to determine the score for each feature. At every iteration,
        the feature with the maximum score is used as a seed for next cluster.
        Options:
            'xsub_max' (default) 
                This option chooses the feature with the maximum across-subject
                correlation value in xsub_map
            'instacorr_max'
                Chooses the feature with the highest cumulative sum of the
                t-values associated with it, i.e., the feature with the highest
                overall connectivity. Does not depend on common activation
                profiles across-subjects  
    xsubp : float 
        Maximum p-value for a node to be considered as a possible seed
        (default: 0.0001). 
    min_nodes : int 
        The minumim number of nodes a cluster needs to have. (default: 100)
    max_leftover : int
        Stopping criterion. When the number of unclustered features falls below
        this number the algorithm returns, leaving the rest of the features
        unclustered. (default: 100)
    econ : Boolean
        If True (default) returns "economy size" results with just the dataset 
        containing clusters. If False returns the large matrices containing the
        statistics for the group analysis on connectivity matrices ('tmap' and
        'pmap') as well as the map of mean cross subject pattern correlations 
        for each node ('xsub_corr'). 
    Returns
    -------
    Dataset
        The dataset returned contains a binary samples matrix with one cluster
        mask per row, and where the columns are the features (nodes,voxels). The
        indices of the seeds corresponding to each cluster are stored as the sample
        attributes "seeds". Only this return when option 'econ' is True
        (default).

    or

    (Dataset, tmap, pmap, xsub_map)
        When option 'econ' is False, returns in addition to the clusters, the
        statistical maps needed to perform analysis. One may save tmap, pmap, 
        and xsub_map as eponymous feature attributes in the original dataset. 
        Calling the function again with these values saved thus in the dataset 
        will speed up subsequent analyses. 


    N.B.:
    Although the running duration of this function is somewhat longer than instant, the
    name derives from the group "InstaCorr" functionality built into the
    AFNI/SUMA (Cox,1996) software packages for exploring functional connectivity maps. In
    particular the algorithm instantiated here was inspired by the 3dGroupInCorr
    program that instantiates the group-instantaneous correlation mapping
    tool. However, the relationship between the AFNI/SUMA software and this software
    ends there. The authors of AFNI/SUMA should not be held responsible for the
    functionality (or lack thereof) of the present software. The name given to
    this function is in reverence of InstaCorr, and we
    thank Bob Cox and Ziad Saad providing us with it. For more
    information on InstaCorr check out AFNI/SUMA. And:
    http://afni.nimh.nih.gov/pub/dist/edu/latest/afni_handouts/instastuff.pdf

    References
    ----------
    Robert R. Cox (1996) AFNI: Software for Analysis and Visualization of
        Functional Magnetic Resonance Neuroimages. Computers and Biomedical Research 
        vol. 9(3), pp.162-173. 
        http://dx.doi.org/10.1006/cbmr.1996.0014 
        http://afni.nimh.nih.gov/
 


 
    """
    if ds.fa.has_key('tmap') and ds.fa.has_key('pmap'):
        print "<> Using stored pmap and tmap from feature attributes of DS."
        tmap = np.copy(ds.fa['tmap'].value)
        pmap = np.copy(ds.fa['pmap'].value)
    else:
        print "Calculating Full connectivity matrices"
        tmap, pmap = get_tstats(get_connectivity_matrices(ds))
        print "Calculating group T-stats"
        
        
    nnod = len(tmap)
    if ds.fa.has_key('xsub_map'):
        print "<> Using stored xsub_map from feature attributes of DS."
        xsub_map = np.copy((ds.fa['xsub_map'].value).transpose())
    else:
        xsub_map = get_xsub_corr_for_all_nodes(ds)
    if not econ:
        tmap_copy = np.copy(tmap)
        pmap_copy = np.copy(pmap)
        xsub_map_copy  = np.copy(xsub_map)
    pmap[tmap<0] = 1. # this excludes all negative t-scores 
    nodes = []
    masks = None
    mask_count = 1
    tmap[xsub_map[2,:]>xsubp,:] = 0 # Discard all nodes > xsubp 
    tmap[tmap<0] = 0

    while sum(np.sum(tmap,1)>0) > max_leftover:
        mask = np.zeros((1,nnod))
        nodes_left = sum(np.sum(tmap,1)>0)
        sys.stdout.write("%s/%s nodes left \r"%(nodes_left,nnod))
        sys.stdout.flush()
        if score=='instacorr_max':
            maxnode = list(np.sum(tmap,1)==np.max(np.sum(tmap,1))).index(True)
        if score=='xsub_max':
            maxnode = list(xsub_map[0,:]==np.max(xsub_map[0,:])).index(True)
            #print maxnode, np.max(xsub_map[:,0])
        print "\n"+str(maxnode)
        if crit=='t':
            nodes_in_mask = tmap[maxnode,:]>=th
        elif crit=='p':
            nodes_in_mask = pmap[maxnode,:]<=th
        elif crit=='pct':
            tm = tmap[maxnode,:]
            tm = tm[tm>0]
            if tm.shape[0]==0:
                break
            print "\n\nLength of tm: "+str(tm.shape)+"\n\n"
            pct_th =scipy.stats.scoreatpercentile(tm,th)
            nodes_in_mask = tmap[maxnode,:]>=scipy.stats.scoreatpercentile(tm,th)
        else:
            print "::: Warning: Invalid threshold criterion, assuming crit=='pct', and th=90"
            nodes_in_mask = pmap[maxnode,:]<=scipy.stats.scoreatpercentile(pmap[maxnode,:],90)
        if sum(nodes_in_mask)>=min_nodes:
            c = xsub_map[:,maxnode]
            if c[0] > 0 and c[2]<xsubp:
                print c[2]
                print "number of nodes in mask == %s"%sum(nodes_in_mask)
                print nodes_in_mask.shape
                print mask.shape

                mask[0,nodes_in_mask] = mask_count
                mask_count = mask_count + 1
                print "found node: %s, %s nodes, xsub_corr=%s"%(maxnode,sum(nodes_in_mask),c[0])
                nodes.append(maxnode)
                if masks is None:
                    masks = mask
                else:
                    masks = np.vstack((masks,mask))
                if not overlap:
                    tmap[:,nodes_in_mask] = 0
                    pmap[:,nodes_in_mask] = 1.

        tmap[nodes_in_mask,:] = 0
        xsub_map[:,nodes_in_mask] = -2.
        xsub_map[:,maxnode] = -2.

    if not overlap:
        masks = np.vstack((np.sum(masks,0).reshape((1,nnod)),masks)) #For viewing purposes
    ds = Dataset(masks)
    ds.a['seeds'] = nodes

    if econ:
        return ds
    else:
        return ds,tmap_copy,pmap_copy,xsub_map_copy

def get_connectivity_matrices(ds):
    """For a data set of shape n-subjects X n-voxels X n-pattern-dimensions
    return Dataset containing an n-subjects X n-voxels X n-voxels set of
    connectivity matrices. Based on the feature-wise pairwise correlations
    between pattern vectors. 
    """
    data = np.array(ds)
    nsubs,nnod,ldist = data.shape
    print "allocating a lot of memory ... array size %s x %s x %s" % (nsubs,nnod,nnod)
    cm = np.zeros((nsubs,nnod,nnod),dtype='float32')
    for i in range(nsubs):
        sys.stdout.write("Computing connectivity matrix %s of %s \r"%(i+1,nsubs))
        sys.stdout.flush()
        cm[i,:,:] = np.float32(np.corrcoef(data[i,:,:]))
    print ''
        
    return Dataset(cm)

def get_tstats(cm,n=500):
    print "Calculating group T-stats"
    tmap = None
    pmap = None
    for i in range(0,len(cm.transpose()),n):
        sys.stdout.write("t-test progress: %s of %s features\r" 
                           % (i,len(cm.transpose())))
        sys.stdout.flush()
       
        t,p = stats.ttest_1samp(np.array(cm[:,:,i:i+n]),0,axis=0)
        if tmap is None:
            tmap = t
            pmap = p
        else:
            tmap = np.hstack((tmap,t))
            pmap = np.hstack((pmap,p))
    # Now fix up diagonals, and any unwanted infinities and NaNs
    # since diagonals will be all t = inf, p = 0
    # give these values that will make those features inconsequential to 
    # the analysis  
    pmap[np.diag_indices(len(pmap))] = 0.
    tmap[np.isnan(tmap)] = 1.
    tmap[np.isnan(tmap)] = 0
    tmap[np.isinf(tmap)] = 0
    tmap[np.diag_indices(len(tmap))] = np.max(tmap,1)
    return tmap, pmap

def get_xsub_corr_for_all_nodes(data):
    """Return the 3 X n-features cross-subject correlation data. The first
    row is the mean cross subject correlation, followed by the t-statistic for
    mean r > 0, and the p-values. Iternally this funection iterates over all
    features calling get_xsub_corr_for_node.
    """
    data = np.array(data)
    nsubs,nnod,ldist = data.shape
    cors = np.zeros((3,nnod))
    for n in range(nnod):
        if n%100==0:
            sys.stdout.write("Calculating cross-subject correlations: %s/%s \r"%(n,nnod))
            sys.stdout.flush()
        cors[:,n] = get_xsub_corr_for_node(data,n)
    cors[np.isnan(cors)] = 0
    print '\n'
    return cors

def get_xsub_corr_for_node(data,node,mean_sample=True):
    """Returns the cross-subject correlation for patterns associated with a
    specified feature ("node"), where subjects are stacked on the first
    (samples, x) dimension, features on the second (y), and patterns on the third (z)
    dimension. Returns: (1) mean cross-subject correlation (when mean_sample==True, 
    default), (2) the cross-subject t-score for for r > 0 (dof = nsubjects-1), and 
    (3) the p-value associated with the t-stat. When mean_samples == False, the
    first return value is an n-subjects length list of containing the
    mean correlation for each subject's pattern with that of every other
    subject.
    """
    data = np.array(data)
    data = data[:,node,:]
    m,n = data.shape
    cors =[]
    for i in range(m):
        d = data[np.array(range(m))!=i,:]
        cor = np.corrcoef(d,data[i,:])
        cors.append(np.mean(cor[m-1,0:m-1]))
    tt = stats.ttest_1samp(cors,0)
    if mean_sample:
        cors = np.mean(cors)
    return cors,tt[0],tt[1]
