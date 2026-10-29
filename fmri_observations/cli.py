"""Command-line runner for the transition benchmark."""
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


def transition_demo(folder,args):
    from . import transitions as m,observation
    cfg=json.loads((folder/'data/config.json').read_text())
    cases=[c for c in cfg['production_cases'] if c['id']==args.case]
    if not cases:raise ValueError(f'Unknown case; see {folder / "data/config.json"}')
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


def main(benchmark_folder, argv=None):
    folder = Path(benchmark_folder).resolve()
    parser = argparse.ArgumentParser(
        description="Compare HMM and VAR models on synthetic fMRI observations."
    )
    parser.add_argument("--mode", choices=["demo", "case", "summary"], default="demo",
                        help="demo: two fits; case: all 64 fits; summary: saved readouts")
    parser.add_argument("--output", type=Path, help="write JSON to this path")
    parser.add_argument("--case", default="cyclic_N97_A1.0_S101",
                        help="production case ID from the benchmark data/config.json")
    parser.add_argument(
        "--branch",
        choices=["identity", "stationary_filter", "filter_change_at_test", "unit_change"],
        default="identity",
        help="observation branch for demo mode",
    )
    args = parser.parse_args(argv)
    if args.mode == "summary":
        result = {"readout.json": json.loads((folder / "data/readout.json").read_text())}
    else:
        with threadpool_limits(limits=1):
            result = transition_demo(folder, args)
    payload = json.dumps(plain(result), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(json.dumps({"output": str(args.output), "benchmark": folder.name, "mode": args.mode}))
    else:
        print(payload, end="")
