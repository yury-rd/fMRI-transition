"""Portable entry points around the preserved study implementations."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def response_demo(folder, args):
    from . import response as m
    from .participant_operator_atlas_complete import qualify_complete_participant_operator_atlas
    if args.mode == 'msc':
        source = args.input or folder/'data/msc_sessions123.npz'
        if not source.exists():
            raise ValueError('Download the paper-01 release ZIP or msc_sessions123.npz; see README.')
        with np.load(source, allow_pickle=False) as z:
            a = z['responses']
        params = json.loads((folder/'data/fixed_coefficients.json').read_text())
        x, b, y, _ = m.fields(a)
        personal = m.losses(x@np.array(params['personal4']), y)
        base = m.losses(b@np.array(params['base3']), y)
        mean_query = x.copy()
        mean_query[:, :, 1:] = np.load(folder/'data/locked_adni_mean_history.npy')
        fixed = m.losses(mean_query@np.array(params['personal4']), y)
        mean_target = np.load(folder/'data/locked_adni_mean_target.npy')
        universal = m.losses(np.broadcast_to(mean_target, y.shape), y)
        result = {'n':len(a), 'personal_vs_base3':m.contrast(personal,base),
                  'personal_vs_fixed_adni_mean_history':m.contrast(personal,fixed),
                  'personal_vs_adni_mean_target':m.contrast(personal,universal)}
        expected = json.loads((folder/'data/msc_expected.json').read_text())
        result['maximum_mean_loss_replay_error'] = max(
            abs(result[k][q]-expected[k][q]) for k in result if k.startswith('personal_')
            for q in ['candidate_mean_loss','reference_mean_loss'])
        return result
    if args.input:
        with np.load(args.input, allow_pickle=False) as z:
            a=z['responses']; reliability=z['reliability']; design=z['design']
        scope='Local user-supplied response fields'
    else:
        rng=np.random.default_rng(20260918)
        a=rng.normal(size=(7,3,5,13))
        reliability=rng.normal(size=(7,2)); design=rng.normal(size=(7,3))
        scope='Synthetic software example; these are not paper measurements'
    atlas=qualify_complete_participant_operator_atlas(
        a,early_visits=(0,1),future_visits=(2,),base_node=0,source_nodes=(1,2,3),
        target_node=4,additive_nodes=(1,2),seen_path_node=3,
        reliability_features=reliability,design_summary_features=design,ridge=1.)
    x,b,y,_=m.fields(a)
    p,_=m.lopo(x,y); baseline,_=m.lopo(b,y)
    return {'scope':scope,'participants':len(a),'personal_loss':atlas.atlas_loss,
            'seven_control_losses':atlas.admissible_baseline_losses,
            'participant_oracle_envelope_loss':atlas.best_admissible_baseline_loss,
            'personal_vs_repeated_base':m.contrast(m.losses(p,y),m.losses(baseline,y))}


def surrogate_demo(folder,args):
    from . import surrogates as m
    from .surrogate_support import measurement,vectors
    if args.input:
        with np.load(args.input,allow_pickle=False) as z: x=z['timeseries']
    else:
        _,_,paths=m.latent_paths(2026102201)
        x=paths[(197,1500)]
    if x.ndim!=2 or x.shape[1]!=7 or not np.isfinite(x).all():
        raise ValueError('Input timeseries must be finite N-by-7')
    observed=[measurement(x[:,j]) for j in range(7)]
    # The fixed first IAAFT output also becomes a pseudo-observation.
    pseudo=np.column_stack([m.iaaft(x[:,j],m.rng_for(2026102201,j,100))[0] for j in range(7)])
    out={}
    for label,field in [('observed',x),('iaaft_as_observation',pseudo)]:
        bank={k:[[measurement(field[:,j]) for j in range(7)]]
              for k in ['iaaft25','iaaft250','common_phase']}
        errors={'iaaft25':[],'iaaft250':[],'common_phase':[]}
        for draw in range(20):
            pair=[m.iaaft_pair(field[:,j],m.rng_for(2026102201,len(x),1500,draw,j,310)) for j in range(7)]
            for budget,name in [(25,'iaaft25'),(250,'iaaft250')]:
                bank[name].append([measurement(v[budget][0]) for v in pair])
                errors[name].extend(v[budget][1] for v in pair)
            shifted,cross,fc,_=m.phase_draw(field,m.rng_for(2026102201,len(x),1500,draw,410))
            bank['common_phase'].append([measurement(shifted[:,j]) for j in range(7)])
            errors['common_phase'].append(max(cross,fc))
        ranks={}
        for family,draws in bank.items():
            ranks[family]={}
            for metric in ['SampEn','MFDFA_shape','Higuchi_shape']:
                result=[]
                for channel in range(7):
                    values=[vectors(row[channel])[metric] for row in draws]
                    result.append(m.rank_stat(np.stack(values))[0] if all(v is not None for v in values) else None)
                ranks[family][metric]=result
        out[label]={'ranks':ranks,'maximum_constraint_error':{k:max(v) for k,v in errors.items()}}
    return {'N':len(x),'surrogates':20,'observed_descriptors':observed,
            'calibration':'IAAFT applied to its own output; finite generator diagnostic','readouts':out}


def clinical_demo(folder,args):
    from . import score,mechanisms,lag_control,paired
    if args.input:
        with np.load(args.input,allow_pickle=False) as z:
            fields,diagnostics=paired.paired_fields(z['parcels'],z['motion24'],z['a5'],z['cosines'],z['membership'])
        a=score.fullscore(fields['BASE_DCT'],seed_start=202620000000)
        b=score.fullscore(fields['A5_DCT'],seed_start=202620000000)
        return {'scope':'Local paired measurement; no clinical model is fitted',
                'BASE':a,'A5':b,'delta_J':b['excess']-a['excess'],'projection':diagnostics}
    b=mechanisms.bank(197,202619000000)
    worlds=[]
    for name,level in [('observation_amplitude',2.),('proxy_contamination',4.)]:
        w=mechanisms.world(b,name,level)
        scores={k:score.fullscore(v,seed_start=202620000000) for k,v in w['fields'].items()}
        worlds.append({'world':name,'level':level,'BASE_J':scores['BASE']['excess'],
                       'A5_J':scores['A5']['excess'],
                       'delta_J':scores['A5']['excess']-scores['BASE']['excess'],
                       'diagnostics':w['diagnostics']})
    innovations=np.random.default_rng(202619000000).standard_normal((198,7))
    lag=[{'alpha':alpha,'J':score.fullscore(lag_control.lag_world(innovations,alpha),
          seed_start=202620000000)['excess']} for alpha in [0.,.5,1.]]
    return {'scope':'Known-world software examples; see saved aggregates for full panels',
            'phase_references':199,'worlds':worlds,'lag_control':lag}


def validate_splits(splits,n):
    all_rows=set(range(n));by_seed={}
    for s in splits:
        tr,te=set(s['train']),set(s['test'])
        if len(tr)!=len(s['train']) or len(te)!=len(s['test']) or tr&te or tr|te!=all_rows:
            raise ValueError('Outer split must partition distinct participant rows')
        by_seed.setdefault(s['seed'],[]).extend(s['test'])
        inner_tests=[]
        for inner in s['inner']:
            a,b=set(inner['train']),set(inner['test'])
            if a&b or a|b!=tr or len(a)!=len(inner['train']) or len(b)!=len(inner['test']):
                raise ValueError('Inner split must partition outer training rows')
            inner_tests.extend(inner['test'])
        if sorted(inner_tests)!=sorted(tr):raise ValueError('Inner test folds must cover training rows once')
    if any(sorted(v)!=list(range(n)) for v in by_seed.values()):
        raise ValueError('Outer test folds must cover each participant once per repeat')


def forecast_demo(folder,args):
    from . import forecast as m
    if args.input:
        with np.load(args.input,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        data['combined']=np.column_stack([data['static'],data['dispersion']])
        scope='Local user-supplied cohort'
    else:
        data=m.fixture();scope='48 synthetic rows from the published software fixture'
    n=len(data['target'])
    for key in ['static','dispersion']:
        if data[key].shape!=(n,21) or not np.isfinite(data[key]).all():raise ValueError(key+' must be finite participant-by-21')
    if data['target'].shape!=(n,) or not np.isfinite(data['target']).all():raise ValueError('Finite target vector required')
    if data['clinical_numeric'].ndim!=2 or len(data['clinical_numeric'])!=n or not np.isfinite(data['clinical_numeric']).all():raise ValueError('Finite clinical_numeric matrix required')
    if data['categories'].ndim!=2 or len(data['categories'])!=n:raise ValueError('categories must be participant-by-column')
    if 'participant_ids' in data and len(set(data['participant_ids'].tolist()))!=n:raise ValueError('One row per participant is required')
    splits=json.loads((args.splits or folder/'data/row_index_splits.json').read_text())
    validate_splits(splits,n)
    seeds=list(dict.fromkeys(s['seed'] for s in splits));results={}
    for rep in ['clinical','static','dispersion','combined']:
        predictions=np.full((len(seeds),n),np.nan);choices=[]
        for s in splits:
            fit=m.candidate_model(data,s,rep)
            predictions[seeds.index(s['seed']),s['test']]=fit['outer_prediction']
            choices.append({'seed':s['seed'],'fold':s['fold'],**fit['selected']})
        if not np.isfinite(predictions).all():raise ValueError('Incomplete predictions')
        errors=np.abs(predictions-data['target'])
        results[rep]={'MAE_by_repeat':errors.mean(1),'mean_MAE':float(errors.mean()),'choices':choices}
    return {'scope':scope,'participants':n,'outer_repeats':len(seeds),'results':results}


def transition_demo(folder,args):
    from . import transitions as m,observation
    cfg=json.loads((folder/'data/config.json').read_text())
    cases=[c for c in cfg['production_cases'] if c['id']==args.case]
    if not cases:raise ValueError('Unknown case; see data/config.json')
    case=cases[0];x,truth=m.simulate(case,cfg);cut=int(.6*len(x));end=int(.8*len(x))
    branches=m.observe(x,cfg,end)
    with np.load(folder/'data/synthetic_216_cases.npz',allow_pickle=False) as z:
        exact=all(np.array_equal(y,z[case['id']+'__'+name]) for name,y in branches.items())
    selected_branches=list(branches) if args.mode=='case' else [args.branch]
    baselines={}
    with (folder/'data/candidate_scores.csv').open() as f:
        for row in csv.DictReader(f):
            if row['id']==case['id']:baselines[(row['branch'],row['candidate'])]=row
    output={};max_error=0.
    for name in selected_branches:
        y=branches[name];alphas=observation.alpha_candidates(y,cut);fits={}
        states=cfg['states'] if args.mode=='case' else [3]
        orders=cfg['var_orders'] if args.mode=='case' else [1]
        filters=cfg['filter_order'] if args.mode=='case' else ['none']
        for k in states:
            for f in filters:fits[f'HMM{k}:{f}']=m.hmm_candidate(y,alphas[f],k,case['seed'],cut,end)
        for k in orders:
            for f in filters:fits[f'VAR{k}:{f}']=m.var_candidate(y,alphas[f],k,cut,end)
        hs=m.select(fits,[k for k in fits if k.startswith('HMM')]);vs=m.select(fits,[k for k in fits if k.startswith('VAR')])
        small={}
        for k,v in fits.items():
            small[k]={a:v[a] for a in ['validation','test']}
            if 'flux' in v:small[k]['flux']=v['flux']
            max_error=max(max_error,*(abs(v[q]-float(baselines[(name,k)][q])) for q in ['validation','test']))
        output[name]={'candidates':small,'selected_hmm':hs,'selected_var':vs,
                      'hmm_minus_var_test':fits[hs]['test']-fits[vs]['test']}
    return {'case':case,'generator_exact_replay':exact,'maximum_saved_score_error':max_error,
            'menu':'full' if args.mode=='case' else 'two-candidate smoke example','branches':output}


def main(paper_folder):
    folder=Path(paper_folder).resolve()
    p=argparse.ArgumentParser(description=folder.name)
    allowed={'01_response_transfer':['demo','msc','summary'],'02_temporal_surrogates':['demo','summary'],
             '03_clinical_response':['demo','summary'],'04_mci_forecast':['demo','summary'],
             '05_transition_benchmark':['demo','case','summary']}[folder.name]
    p.add_argument('--mode',choices=allowed,default='demo')
    p.add_argument('--input',type=Path)
    p.add_argument('--splits',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--case',default='cyclic_N97_A1.0_S101')
    p.add_argument('--branch',choices=['identity','stationary_filter','filter_change_at_test','unit_change'],default='identity')
    args=p.parse_args()
    if args.mode=='summary':
        result={q.name:json.loads(q.read_text()) for q in sorted((folder/'data').glob('*.json'))
                if q.name not in ['config.json','row_index_splits.json']}
    else:
        fn={'01':response_demo,'02':surrogate_demo,'03':clinical_demo,'04':forecast_demo,'05':transition_demo}[folder.name[:2]]
        with threadpool_limits(limits=1): result=fn(folder,args)
    payload=json.dumps(plain(result),indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(payload)
        print(json.dumps({'output':str(args.output),'paper':folder.name,'mode':args.mode}))
    else:print(payload,end='')
