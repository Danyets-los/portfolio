import json
import math
from pathlib import Path
import numpy as np
import pytest
import networkx as nx
from model_h import availability,wait_probability,kit_demand,assignment,reference_matrix,analytical_experiment,U,W,D
from model_g import build_graph,graph_analysis
from simulation import run_replication

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def model():
    cfg=json.loads((ROOT/'config/coursework.json').read_text(encoding='utf-8'))
    return cfg,build_graph(ROOT,cfg)

def test_edge_cases_and_partial_inverse(model):
    cfg,_=model
    sc=cfg['simulation']
    for u,e in ((.6,1),(1,1.2)):
        assert wait_probability(availability(0,3),u,e,sc)==1
        assert wait_probability(availability(3,3),u,e,sc)==0
        for q in (1,2):
            p=wait_probability(availability(q,3),u,e,sc)
            assert 0<p<1
            assert kit_demand(1-p-1e-8,3,u,e,sc)<=q
            assert kit_demand(1-p+1e-8,3,u,e,sc)>q
    assert assignment(0,1,1,sc)==0
    assert 0<assignment(.5,1,1,sc)<assignment(1,1,1,sc)
    with pytest.raises(ValueError):
        availability(1,0)

def test_analytical_reference_against_exact_renewal(model):
    cfg,(_,_,_,_,_,faults)=model
    sc,ref=cfg['simulation'],cfg['reference']
    p=reference_matrix(.003,faults[0],True,1,sc,ref)
    assert np.all(p>=0) and np.allclose(p.sum(axis=1),1)
    result=analytical_experiment(.003,faults[0],True,1,sc,ref)
    assert result['stationary_residual']<1e-12
    assert math.isclose(result['availability_analytic'],1/(1+result['restoration_analytic_hours']*
                        (1-p[U,U])/sc['step_hours']),rel_tol=1e-12)
    assert p[D,W]>0 and result['cycles']>=1000

def test_routes_flows_and_louvain_partition(model):
    cfg,(g,road,time,fit,calib,_)=model
    a=graph_analysis(g,road,time,fit,cfg)
    assert g.is_multigraph() and g.is_directed()
    assert len(fit)==216 and calib['assets_with_data']>180 and nx.is_connected(road)
    r=a['route']
    assert all(road.has_edge(x,y) for x,y in zip(r['dijkstra'],r['dijkstra'][1:]))
    assert math.isclose(r['hours'],time[r['origin']][r['destination']],abs_tol=1e-12)
    f=a['flow']
    assert f['value_kits_week']==f['cut_kits_week']>0
    assert sum(f['served_by_site'].values())==f['value_kits_week']
    assert all(0<=e['flow_kits_week']<=e['capacity_kits_week'] for e in f['edges'])
    assert all(v==0 for k,v in f['balances'].items() if k not in ('SUPPLY','DEMAND'))
    sites=[s for c in a['clusters'] for s in c['stations']]
    assert len(sites)==len(set(sites))==len(road)

def test_simulation_reproducible_and_resource_accounting(model):
    base,(g,road,time,fit,calib,faults)=model
    cfg=json.loads(json.dumps(base))
    cfg['simulation'].update(horizon_hours=300,warmup_hours=24)
    case=cfg['scenarios'][0]
    a,d=run_replication(g,time,fit,faults,cfg,case,421,True)
    b,_=run_replication(g,time,fit,faults,cfg,case,421,False)
    assert a==b and all(d['checks'].values())
    assert a['cycles_completed']+a['unfinished_cohort']==a['failures_started']
    assert 0<a['availability']<=1 and 0<a['utilization']<=1
    assert all(e['stock_after']>=0 for e in d['events'] if e['event']=='reserved')
    assert all(e['occupied']<=e['capacity'] for e in d['events'] if e['event']=='dispatch')
    assert a['cost_usd_day']==pytest.approx(sum(v for k,v in a.items() if k.startswith('cost_') and k!='cost_usd_day'))
