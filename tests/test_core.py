import copy
import json
from pathlib import Path
import unittest

import numpy as np
from threadpoolctl import threadpool_limits
from fmri_observation import response,surrogates,score,paired,mechanisms,forecast,scoring,features
from fmri_observation.cli import validate_splits

ROOT=Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_response_holdout_matches_direct_fit(self):
        rng=np.random.default_rng(20260918)
        a=rng.normal(size=(7,3,5,13));x,_,y,_=response.fields(a)
        pred,_=response.lopo(x,y)
        for i in range(7):
            train=np.arange(7)!=i
            np.testing.assert_allclose(pred[i],x[i]@response.fit(x[train],y[train]),rtol=0,atol=1e-11)
        a[:,2,4]+=1000
        np.testing.assert_array_equal(response.fields(a)[0],x)

    def test_entropy_counts_and_surrogate_constraints(self):
        _,_,p=surrogates.latent_paths(2026102201);x=p[(197,1500)]
        a,b=surrogates.sampen(x[:,0]),surrogates.sampen_fast(x[:,0])
        self.assertEqual((a['A'],a['B']),(b['A'],b['B']))
        for y,_ in surrogates.iaaft_pair(x[:,0],np.random.default_rng(310)).values():
            np.testing.assert_array_equal(np.sort(y),np.sort(x[:,0]))
        _,cross,fc,_=surrogates.phase_draw(x,np.random.default_rng(410))
        self.assertLess(max(cross,fc),1e-10)
        self.assertEqual(surrogates.sampen_fast(np.ones(100))['status'],'unsupported')

    def test_joint_projection_uses_same_support(self):
        b=mechanisms.bank(197,202619000000)
        w=mechanisms.world(b,'observation_amplitude',2.)
        fields,d=paired.paired_fields(w['parcels'],b['m24'],w['proxies'],np.empty((197,0)),b['membership'])
        for name,target in [('BASE_DCT','BASE'),('A5_DCT','A5')]:
            np.testing.assert_allclose(fields[name],w['fields'][target],rtol=0,atol=1e-10)
        self.assertEqual(d['effective_A5_rank'],5)
        np.testing.assert_allclose(score.rho2(fields['BASE_DCT']),score.rho2(fields['BASE_DCT']*2),rtol=0,atol=1e-10)

    @unittest.skipUnless((ROOT/'papers/04_mci_forecast/data/row_index_splits.json').exists(),'forecast package is not in this paper-only download')
    def test_forecast_training_isolation(self):
        data=forecast.fixture()
        split=json.loads((ROOT/'papers/04_mci_forecast/data/row_index_splits.json').read_text())[0]
        a=forecast.candidate_model(data,split,'combined')
        changed=copy.deepcopy(data)
        changed['combined'][split['test']]+=300
        changed['target'][split['test']]+=1000
        b=forecast.candidate_model(changed,split,'combined')
        self.assertEqual(a['selected'],b['selected'])
        np.testing.assert_array_equal(a['outer_transform']['MRI']['train_scores'],b['outer_transform']['MRI']['train_scores'])
        np.testing.assert_array_equal(a['outer_ridge_parameters'][2],b['outer_ridge_parameters'][2])

    @unittest.skipUnless((ROOT/'papers/04_mci_forecast/data/row_index_splits.json').exists(),'forecast package is not in this paper-only download')
    def test_splits_reject_leakage(self):
        splits=json.loads((ROOT/'papers/04_mci_forecast/data/row_index_splits.json').read_text())
        validate_splits(splits,48)
        splits[0]['train'].append(splits[0]['test'][0])
        with self.assertRaises(ValueError):validate_splits(splits,48)

    def test_conditional_scores_are_causal_and_unit_covariant(self):
        y=np.random.default_rng(101).normal(size=(97,7));alpha=np.full(7,.35)
        v,j=scoring.transform(y,alpha,58)
        model={'pi':[.5,.5],'trans':[[.8,.2],[.2,.8]],'mean':np.zeros((2,7)), 'var':np.ones((2,7))}
        a=scoring.hmm_frames(v,model)+j
        changed=y.copy();changed[-1]+=1000
        v2,j2=scoring.transform(changed,alpha,58)
        np.testing.assert_array_equal(a[:-1],(scoring.hmm_frames(v2,model)+j2)[:-1])
        units=np.array([.125,.25,.5,1,2,4,8])
        v3,j3=scoring.transform(y*units,alpha,58)
        np.testing.assert_allclose(a-np.log(units).sum(),scoring.hmm_frames(v3,model)+j3,rtol=0,atol=1e-10)

    def test_feature_windows_preserve_full_static_vector(self):
        x=np.random.default_rng(732401).normal(size=(197,7))
        out=features.extract(x,3.)
        self.assertEqual((out['window_samples'],out['stride_samples']),(20,10))
        np.testing.assert_allclose(out['static'],np.corrcoef(x,rowvar=False)[np.triu_indices(7,1)],rtol=0,atol=1e-10)
        self.assertEqual(out['dispersion'].shape,(21,))
        with self.assertRaises(ValueError):features.extract(x[:20],3.)


if __name__=='__main__':unittest.main()
