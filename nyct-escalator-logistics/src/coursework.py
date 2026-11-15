"""Reproducible offline orchestration and statistical comparison."""
import csv
import gzip
import importlib.metadata
import json
import sys
from collections import defaultdict
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from model_g import load_json,write_json,build_graph,graph_analysis,export_graph
from model_h import STATES,ARCS,analytical_experiment,priority,t_interval
from simulation import run_replication


def aggregate_details(details):
    out={}
    for name in ('engineer_utilization','warehouse_utilization','sku_shortages','sku_used',
                 'sku_wait_hours','asset_failures','site_wait_hours','state_hours'):
        group=defaultdict(float)
        for d in details:
            for k,v in d[name].items():
                group[k]+=v/len(details)
        out[name]=dict(sorted(group.items(),key=lambda x:(-x[1],x[0])))
    return out


def save_event_log(out, case, detail):
    with gzip.open(out/f'events_{case}.jsonl.gz','wt',encoding='utf-8') as f:
        for row in detail['events']:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    pd.DataFrame(detail['cycles']).to_csv(out/f'cycles_{case}_rep01.csv',index=False)


def run(root,quick=False):
    cfg=load_json(root/'config/coursework.json')
    sc=cfg['simulation']
    out=root/'results'/('quick' if quick else 'coursework')
    out.mkdir(parents=True,exist_ok=True)
    if quick:
        sc.update(horizon_hours=672,warmup_hours=48,replications=3)
    g,road,travel,fit,calib,faults=build_graph(root,cfg)
    a=graph_analysis(g,road,travel,fit,cfg)
    if not quick:
        export_graph(root,g,a,fit)
    write_json(out/'graph_analysis.json',a)
    write_json(out/'calibration.json',calib)
    focal=fit.sort_values(['months','rate_per_up_hour','id'],ascending=[False,False,True]).iloc[0]
    f=faults[cfg['reference']['fault_index']]
    ep=priority(g.nodes[g.nodes[focal.id]['station_id']]['escalator_count_in_scope'],sc)
    h=analytical_experiment(focal.rate_per_up_hour,f,f['criticality']=='high',ep,sc,cfg['reference'])
    samples=h.pop('sample_restoration_hours')
    h.update(asset_id=focal.id,fault_id=f['id'])
    hx=nx.DiGraph()
    hx.add_nodes_from(STATES)
    for x,y in ARCS:
        hx.add_edge(STATES[x],STATES[y],reference_probability=h['p'][x][y])
    for i,n in enumerate(STATES):
        hx.add_edge(n,n,reference_probability=h['p'][i][i])
    assert nx.is_strongly_connected(hx) and nx.is_aperiodic(hx)
    write_json(out/'graph_H.json',nx.node_link_data(hx,edges='edges'))
    write_json(out/'markov_reference.json',h)
    pd.DataFrame({'restore_h':samples}).to_csv(out/'reference_cycles.csv',index=False)
    # Replicate seeds are independent across r and aligned across alternatives.
    seeds=[int(s.generate_state(1)[0]) for s in np.random.SeedSequence(sc['seed']).spawn(sc['replications'])]
    summaries={}; rows=[]; baseline_rows=None; by_case={}
    confidence=sc['confidence']
    for case in cfg['scenarios']:
        ident=case['id']
        values=[]; details=[]
        for i,seed in enumerate(seeds):
            kpi,detail=run_replication(g,travel,fit,faults,cfg,case,seed,keep_log=(i==0))
            if i==0:
                save_event_log(out,ident,detail)
                del detail['events'],detail['cycles']
            values.append(kpi); details.append(detail)
            rows.append({'scenario':ident,'replication':i+1,'seed':seed,**kpi})
        if baseline_rows is None:
            baseline_rows=values
        by_case[ident]=values
        kpis={k:t_interval([v[k] for v in values],confidence) for k in values[0]}
        diff={k:t_interval([v[k]-b[k] for v,b in zip(values,baseline_rows)],confidence) for k in values[0]}
        r=kpis['mean_cycle_hours']; rel=(r['hi']-r['lo'])/2/r['mean']
        summaries[ident]={'case':case,'kpi':kpis,'paired_delta_vs_baseline':diff,
                         'details':aggregate_details(details),'relative_cycle_ci_half_width':rel,
                         'precision_target_met':rel<=sc['max_relative_cycle_ci'],
                         'all_invariants_passed':all(all(d['checks'].values()) for d in details)}
        write_json(out/f'scenario_{ident}.json',summaries[ident])
        print(f"{ident}: {r['mean']:.3f} h, availability={kpis['availability']['mean']:.5f}, "
              f"cost={kpis['cost_usd_day']['mean']:.2f} USD/day, relative CI={rel:.2%}",flush=True)
    pd.DataFrame(rows).to_csv(out/'replications.csv',index=False)
    table=[{'scenario':s,'label':d['case']['label'],'metric':k,**v} for s,d in summaries.items() for k,v in d['kpi'].items()]
    pd.DataFrame(table).to_csv(out/'kpi_summary.csv',index=False)
    wide=[]
    for s,d in summaries.items():
        wide.append({'scenario':s,'label':d['case']['label'],
                     **{k:d['kpi'][k]['mean'] for k in ('mean_cycle_hours','utilization','availability','cost_usd_day','stockout_share')}})
    pd.DataFrame(wide).to_csv(out/'scenario_comparison.csv',index=False)
    choose=lambda cases: min(cases,key=lambda s:(summaries[s]['kpi']['cost_usd_day']['mean'],s))
    best_staff=choose(['hire','stock']); best_wh=choose([s for s in summaries if s.startswith('warehouse_')])
    best_training=choose([s for s in summaries if s.startswith('train_')])
    training_by_cycle=min((s for s in summaries if s.startswith('train_')),key=lambda s:summaries[s]['kpi']['mean_cycle_hours']['mean'])
    feasible=[s for s in summaries if s not in ('stress','disruption') and
              summaries[s]['kpi']['availability']['mean']>=cfg['decision']['minimum_availability']]
    recommended=choose(feasible) if feasible else None
    decisions={'hire_vs_stock':best_staff,'warehouse_min_cost':best_wh,'training_min_cost':best_training,
               'training_min_cycle':training_by_cycle,'recommended_with_availability_floor':recommended,
               'availability_floor':cfg['decision']['minimum_availability'],
               'selection_rule':'minimum point mean total USD/day, conditional on declared model; compare paired CI separately'}
    comparisons={}
    for first,second in [('hire','stock')]+[(best_wh,s) for s in summaries if s.startswith('warehouse_') and s!=best_wh]+[(best_training,s) for s in summaries if s.startswith('train_') and s!=best_training]:
        comparisons[first+'__minus__'+second]={k:t_interval([a[k]-b[k] for a,b in zip(by_case[first],by_case[second])],confidence)
             for k in ('mean_cycle_hours','availability','utilization','cost_usd_day')}
    write_json(out/'paired_policy_comparisons.json',comparisons)
    write_json(out/'experiment.json',{'settings':sc,'seeds':seeds,'focal_reference_asset':focal.id,
              'scenarios':list(summaries),'decisions':decisions,
              'versions':{'python':sys.version.split()[0],**{p:importlib.metadata.version(p) for p in ['networkx','numpy','scipy','pandas','matplotlib','pytest']}}})
    from figures import make_figures
    make_figures(out,g,road,a,h,samples,summaries,cfg)
    print(f'Finished: {out}',flush=True)
