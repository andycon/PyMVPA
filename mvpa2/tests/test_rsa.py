# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for rsa measures"""

<<<<<<< HEAD
from mvpa2.testing import *
skip_if_no_external('scipy')

=======
from mvpa2.testing import sweepargs
>>>>>>> da0a4bfaaf9667daa7143816fc557ac109c1c72a
from mvpa2.testing.datasets import datasets
from mvpa2.measures.anova import OneWayAnova

import numpy as np
from mvpa2.mappers.fx import *
from mvpa2.datasets.base import dataset_wizard, Dataset

from mvpa2.testing.tools import *

from mvpa2.measures.rsa import *
<<<<<<< HEAD
from mvpa2.base import externals
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, pearsonr
=======
import scipy.stats as stats
>>>>>>> da0a4bfaaf9667daa7143816fc557ac109c1c72a

data = np.array([[ 0.22366105, 0.51562476, 0.62623543, 0.28081652, 0.56513533],
                [ 0.22077129, 0.63013374, 0.19641318, 0.38466208, 0.60788347],
                [ 0.64273055, 0.60455658, 0.71368501, 0.36652763, 0.51720253],
                [ 0.40148338, 0.34188668, 0.09174233, 0.33906488, 0.17804584],
                [ 0.60728718, 0.6110304 , 0.84817742, 0.33830628, 0.7123945 ],
                [ 0.32113428, 0.16916899, 0.53471886, 0.93321617, 0.22531679]])


<<<<<<< HEAD
def test_PDistConsistency():
=======
def test_DissimilarityConsistencyMeasure():
>>>>>>> da0a4bfaaf9667daa7143816fc557ac109c1c72a
    targets = np.tile(xrange(3),2)
    chunks = np.repeat(np.array((0,1)),3)
    # correct results
    cres1 = 0.41894348
<<<<<<< HEAD
    cres2 = np.array([[ 0.73062639, 0.16137995, 0.59441713]]).T
=======
    cres2 = np.array([[ 0.16137995, 0.73062639, 0.59441713]])
>>>>>>> da0a4bfaaf9667daa7143816fc557ac109c1c72a
    dc1 = data[0:3,:] - np.mean(data[0:3,:],0)
    dc2 = data[3:6,:] - np.mean(data[3:6,:],0)
    center = squareform(np.corrcoef(pdist(dc1,'correlation'),pdist(dc2,'correlation')), 
                        checks=False).reshape((1,-1))
    dsm1 = stats.rankdata(pdist(data[0:3,:],'correlation').reshape((1,-1)))
    dsm2 = stats.rankdata(pdist(data[3:6,:],'correlation').reshape((1,-1)))

    spearman = squareform(np.corrcoef(np.vstack((dsm1,dsm2))), 
                        checks=False).reshape((1,-1))
    
    ds = dataset_wizard(samples=data, targets=targets, chunks=chunks)
<<<<<<< HEAD
    dscm = PDistConsistency()
    res1 = dscm(ds)
    dscm_c = PDistConsistency(center_data=True)
    res2 = dscm_c(ds)
    dscm_sp = PDistConsistency(consistency_metric='spearman')
    res3 = dscm_sp(ds)
    ds.append(ds)
    chunks = np.repeat(['one', 'two', 'three'], 4)
    ds.sa['chunks'] = chunks
    res4 = dscm(ds)
    dscm_sq = PDistConsistency(square=True)
    res4_sq = dscm_sq(ds)
    for i, p in enumerate(res4.sa.pairs):
        sqval =  np.asscalar(res4_sq[res4_sq.sa.chunks == p[0],
                                     res4_sq.sa.chunks == p[1]])
        assert_equal(sqval, res4.samples[i, 0])
=======
    dscm = DissimilarityConsistencyMeasure()
    res1 = dscm(ds)
    dscm_c = DissimilarityConsistencyMeasure(center_data=True)
    res2 = dscm_c(ds)
    dscm_sp = DissimilarityConsistencyMeasure(consistency_metric='spearman')
    res3 = dscm_sp(ds)
    ds.append(ds)
    chunks = np.repeat(np.array((0,1,2,)),4)
    ds.sa['chunks'] = chunks
    res4 = dscm(ds)
>>>>>>> da0a4bfaaf9667daa7143816fc557ac109c1c72a
    assert_almost_equal(np.mean(res1.samples),cres1)
    assert_array_almost_equal(res2.samples, center)
    assert_array_almost_equal(res3.samples, spearman)
    assert_array_almost_equal(res4.samples,cres2)



<<<<<<< HEAD
def test_PDist():
=======
def test_DissimilarityMatrixMeasure():
>>>>>>> da0a4bfaaf9667daa7143816fc557ac109c1c72a
    targets = np.tile(xrange(3),2)
    chunks = np.repeat(np.array((0,1)),3)
    ds = dataset_wizard(samples=data, targets=targets, chunks=chunks)
    data_c = data - np.mean(data,0)
<<<<<<< HEAD
    # DSM matrix elements should come out as samples of one feature
    # to be in line with what e.g. a classifier returns -- facilitates
    # collection in a searchlight ...
    euc = pdist(data, 'euclidean')[None].T
    pear = pdist(data, 'correlation')[None].T
    city = pdist(data, 'cityblock')[None].T
    center_sq = squareform(pdist(data_c,'correlation'))

    # Now center each chunk separately
    dsm1 = PDist()
    dsm2 = PDist(pairwise_metric='euclidean')
    dsm3 = PDist(pairwise_metric='cityblock')
    dsm4 = PDist(center_data=True,square=True)
    assert_array_almost_equal(dsm1(ds).samples,pear)
    assert_array_almost_equal(dsm2(ds).samples,euc)
    dsm_res = dsm3(ds)
    assert_array_almost_equal(dsm_res.samples,city)
    # length correspondings to a single triangular matrix
    assert_equal(len(dsm_res.sa.pairs), len(ds) * (len(ds) - 1) / 2)
    # generate label pairs actually reflect the vectorform generated by
    # squareform()
    dsm_res_square = squareform(dsm_res.samples.T[0])
    for i, p in enumerate(dsm_res.sa.pairs):
        assert_equal(dsm_res_square[p[0], p[1]], dsm_res.samples[i, 0])
    dsm_res = dsm4(ds)
    assert_array_almost_equal(dsm_res.samples,center_sq)
    # sample attributes are carried over
    assert_almost_equal(ds.sa.targets, dsm_res.sa.targets)

def test_PDistTargetSimilarity():
=======
    euc = pdist(data, 'euclidean').reshape((1,-1))
    pear = pdist(data, 'correlation').reshape((1,-1))
    city = pdist(data, 'cityblock').reshape((1,-1))
    center_sq = squareform(pdist(data_c,'correlation'))

    # Now center each chunk separately
    dsm1 = DissimilarityMatrixMeasure()
    dsm2 = DissimilarityMatrixMeasure(pairwise_metric='euclidean')
    dsm3 = DissimilarityMatrixMeasure(pairwise_metric='cityblock')
    dsm4 = DissimilarityMatrixMeasure(center_data=True,square=True)
    assert_array_almost_equal(dsm1(ds).samples,pear)
    assert_array_almost_equal(dsm2(ds).samples,euc)
    assert_array_almost_equal(dsm3(ds).samples,city)
    assert_array_almost_equal(dsm4(ds).samples,center_sq)

def test_TargetDissimilarityCorrelationMeasure():
>>>>>>> da0a4bfaaf9667daa7143816fc557ac109c1c72a
    ds = Dataset(data)
    tdsm = range(15)
    ans1 = np.array([0.30956920104253222, 0.26152022709856804])
    ans2 = np.array([0.53882710751962437, 0.038217527859375197])
    ans3 = np.array([0.33571428571428574, 0.22121153763932569])
<<<<<<< HEAD
    tdcm1 = PDistTargetSimilarity(tdsm)
    tdcm2 = PDistTargetSimilarity(tdsm,
                                            pairwise_metric='euclidean')
    tdcm3 = PDistTargetSimilarity(tdsm,
                                comparison_metric = 'spearman')
    tdcm4 = PDistTargetSimilarity(tdsm,
=======
    tdcm1 = TargetDissimilarityCorrelationMeasure(tdsm)
    tdcm2 = TargetDissimilarityCorrelationMeasure(tdsm,
                                            pairwise_metric='euclidean')
    tdcm3 = TargetDissimilarityCorrelationMeasure(tdsm,
                                comparison_metric = 'spearman')
    tdcm4 = TargetDissimilarityCorrelationMeasure(tdsm,
>>>>>>> da0a4bfaaf9667daa7143816fc557ac109c1c72a
                                    corrcoef_only=True)
    a1 = tdcm1(ds)
    a2 = tdcm2(ds)
    a3 = tdcm3(ds)
    a4 = tdcm4(ds)
<<<<<<< HEAD
    assert_array_almost_equal(a1.samples.squeeze(),ans1)
    assert_array_equal(a1.fa.metrics, ['rho', 'p'])
    assert_array_almost_equal(a2.samples.squeeze(),ans2)
    assert_array_equal(a2.fa.metrics, ['rho', 'p'])
    assert_array_almost_equal(a3.samples.squeeze(),ans3)
    assert_array_equal(a3.fa.metrics, ['rho', 'p'])
    assert_array_almost_equal(a4.samples.squeeze(),ans1[0])
    assert_array_equal(a4.fa.metrics, ['rho'])
=======
    assert_array_almost_equal(a1.samples,ans1.reshape(-1,1))
    assert_array_almost_equal(a2.samples,ans2.reshape(-1,1))
    assert_array_almost_equal(a3.samples,ans3.reshape(-1,1))
    assert_array_almost_equal(a4.samples,ans1[0].reshape(-1,1))
>>>>>>> da0a4bfaaf9667daa7143816fc557ac109c1c72a




