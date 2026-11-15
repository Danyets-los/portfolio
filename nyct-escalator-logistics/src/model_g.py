"""Typed service graph, transparent geographic approximation and audited flow."""
import copy
import hashlib
import json
import math
from collections import Counter
from itertools import combinations
import networkx as nx
import pandas as pd
from model_h import priority,urgency,wait_probability


def roster(cfg):
    base=cfg['baseline']
    originals=base['engineers']
    next_id=base['first_replica_post_index']
    result={n:n for n in originals}
    for profile in originals:
        for _ in range(base['posts_per_profile']-1):
            result[f'ENG-M{next_id:02d}']=profile
            next_id+=1
    return result


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def geo_km(a,b):
    p,q = map(math.radians,(a['latitude'],b['latitude']))
    d = math.radians(b['longitude']-a['longitude'])
    h = math.sin((q-p)/2)**2 + math.cos(p)*math.cos(q)*math.sin(d/2)**2
    return 6371.0088*2*math.asin(min(1,math.sqrt(h)))


def fit_history(graph, rows, cfg):
    """Join by asset code and station MRN; shrink the Poisson rate toward pool."""
    df = pd.DataFrame(rows)
    eligible = {d['equipment_code']:(n,str(int(d['station_mrn']))) for n,d in graph.nodes(data=True)
                if d['node_type']=='escalator' and d['eligible_for_future_service_model']}
    df['month'] = df.month.str[:7]
    df['station_mrn'] = df.station_mrn.map(lambda x: str(int(x)) if pd.notna(x) and str(x).isdigit() else '')
    selected = df[df.equipment_code.isin(eligible)].copy()
    matched = selected[selected.apply(lambda r: r.station_mrn == eligible[r.equipment_code][1],axis=1)].copy()
    matched['available_hours'] = pd.to_numeric(matched['_24_hour_hours_available'],errors='coerce')
    matched['total_hours'] = pd.to_numeric(matched['_24_hour_total_hours'],errors='coerce')
    matched['events'] = pd.to_numeric(matched.unscheduled_outages,errors='coerce')
    good = matched[(matched.available_hours>0)&(matched.total_hours>=matched.available_hours-1e-6)
                   &(matched.events>=0)&(matched.events%1==0)].copy()
    if good.duplicated(['equipment_code','month']).any():
        raise ValueError('Nonunique asset-month in history')
    tab = good.groupby('equipment_code').agg(events=('events','sum'),hours=('available_hours','sum'),months=('month','size'))
    pooled = float(good.events.sum()/good.available_hours.sum())
    prior = cfg['calibration']['prior_hours']
    result = []
    for code,(n,_) in sorted(eligible.items()):
        events,hours,months = (tab.loc[code,['events','hours','months']].tolist()
                              if code in tab.index else (0,0,0))
        rate = (events+prior*pooled)/(hours+prior)
        graph.nodes[n].update(rate_per_up_hour=float(rate),history_events=int(events),
                              history_up_hours=float(hours),history_months=int(months),
                              rate_origin='shrunken_MTA_2025' if hours else 'MTA_2025_pool_fallback')
        result.append({'id':n,'equipment_code':code,'history_events':int(events),
                       'up_hours':hours,'months':int(months),'rate_per_up_hour':rate})
    return pd.DataFrame(result), {'source_rows':len(df),'asset_matched_rows':len(selected),
        'station_matched_rows':len(matched),'valid_rows':len(good),'assets_with_data':len(tab),
        'assets_no_history':len(eligible)-len(tab),'pool_rate_per_up_hour':pooled,
        'history_events':int(good.events.sum()),'history_up_hours':float(good.available_hours.sum()),
        'historical_observed_availability':float(good.available_hours.sum()/good.total_hours.sum()),
        'prior_hours':prior}


def build_graph(root,cfg):
    base = load_json(root/'data/graph_vertices.json')
    graph = nx.node_link_graph(base,edges='edges')
    graph.graph['stage']='coursework'
    graph.graph['case']='NYCT escalators'
    nc = cfg['network']
    rows_path = root/'source/mta_availability_2025.json'
    fit,calib = fit_history(graph,load_json(rows_path),cfg)
    calib['sha256'] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    stations = sorted(n for n,d in graph.nodes(data=True) if d['node_type']=='station')
    faults = load_json(root/'config/vertices.json')['fault_classes']
    pair = nx.Graph()
    pair.add_nodes_from(stations)
    for a,b in combinations(stations,2):
        sa,sb = graph.nodes[a],graph.nodes[b]
        km = geo_km(sa,sb)*nc['road_factor']
        if sa['borough']!=sb['borough']:
            km *= nc['cross_borough_factor']
        pair.add_edge(a,b,distance_km=km,travel_h=nc['leg_overhead_hours']+km/nc['km_per_hour'])
    road = nx.minimum_spanning_tree(pair,weight='travel_h')
    for a in stations:
        ranked = sorted(pair[a],key=lambda b:(pair[a][b]['travel_h'],b))
        for b in ranked[:nc['nearest']]:
            road.add_edge(a,b,**pair[a][b])
    for a,b,d in road.edges(data=True):
        for s,t in ((a,b),(b,a)):
            graph.add_edge(s,t,edge_type='travel',**d,capacity_kits_week=nc['corridor_kits_week'],origin='geographic_proxy')
    active=set(fit.id)
    for n,d in list(graph.nodes(data=True)):
        if d['node_type']=='escalator':
            graph.add_edge(n,d['station_id'],edge_type='located_at',origin='inventory')
            graph.add_edge('OP-NYCT',n,edge_type='oversees',origin='inventory')
            if n in active:
                for f in faults:
                    graph.add_edge(n,f['id'],edge_type='may_fail_as',origin='scenario_assumption')
    for i,f in enumerate(faults):
        f.update(cfg['faults'][i])
        graph.nodes[f['id']].update(f)
        graph.add_edge(f['id'],f['part_id'],edge_type='requires_catalog',max_kits=f['max_kits'],origin='scenario_assumption')
    homes=cfg['baseline']['homes']
    for eng,profile in roster(cfg).items():
        home=homes[profile]
        if eng not in graph:
            graph.add_node(eng,node_type='engineer',data_origin='model',label='Инженерный пост '+eng,
                           skills=list(graph.nodes[profile]['skills']))
        graph.nodes[eng].update(home_station_id=home,competency_profile=profile)
        graph.add_edge(eng,home,edge_type='starts_at',origin='scenario_assumption')
        for f in faults:
            if f['skill'] in graph.nodes[eng]['skills']:
                graph.add_edge(eng,f['id'],edge_type='qualified_for',origin='scenario_assumption')
    for s in cfg['scenarios']:
        if 'hire' in s:
            e=s['hire']; n=e['id']
            graph.add_node(n,node_type='engineer',data_origin='model',label='Кандидат '+n,
                           skills=e['skills'],home_station_id=e['home'],active_baseline=False)
            graph.add_edge(n,e['home'],edge_type='starts_at',origin='scenario_assumption')
            for f in faults:
                if f['skill'] in e['skills']:
                    graph.add_edge(n,f['id'],edge_type='qualified_for',origin='scenario_assumption')
        if 'warehouse' in s:
            w=s['warehouse']; site=w['site']
            if w['id'] not in graph:
                graph.add_node(w['id'],node_type='warehouse',label='Модельный склад '+graph.nodes[site]['label'],
                               data_origin='model',anchor_station_id=site,
                               latitude=graph.nodes[site]['latitude'],longitude=graph.nodes[site]['longitude'])
    for w,wd in list(graph.nodes(data=True)):
        if wd['node_type']=='warehouse':
            graph.add_edge(w,wd['anchor_station_id'],edge_type='dispatch_from',origin='scenario_assumption',
                           active_baseline=(w==cfg['baseline']['warehouse']))
            for f in faults:
                graph.add_edge(w,f['part_id'],edge_type='catalog_stock',origin='scenario_assumption',
                               initial_kits=cfg['inventory']['base_stock'][faults.index(f)] if w==cfg['baseline']['warehouse'] else 0)
    time=dict(nx.all_pairs_dijkstra_path_length(road,weight='travel_h'))
    for f in faults:
        graph.nodes[f['part_id']]['cost_usd_kit']=cfg['economics']['kit_usd'][faults.index(f)]
    return graph,road,time,fit,calib,faults


def graph_analysis(graph,road,time,fit,cfg):
    demand=Counter()
    kits=Counter()
    faults=[d for _,d in graph.nodes(data=True) if d['node_type']=='fault_class']
    sc=cfg['simulation']
    active=fit.id.tolist()
    for n in active:
        d=graph.nodes[n]
        demand[d['station_id']]+=d['rate_per_up_hour']
        e=priority(graph.nodes[d['station_id']]['escalator_count_in_scope'],sc)
        mean=sum(x['weight']*sum(wait_probability(k/x['max_kits'],urgency(x['criticality']=='high',sc),e,sc)
                  for k in range(x['max_kits'])) for x in faults)
        kits[d['station_id']]+=168*d['rate_per_up_hour']*mean
    wh=cfg['baseline']['warehouse']
    src=graph.nodes[wh]['anchor_station_id']
    target=max(demand,key=lambda a:(time[src][a],a))
    r1=nx.dijkstra_path(road,src,target,weight='travel_h')
    r2=nx.bellman_ford_path(road,src,target,weight='travel_h')
    lengths=[sum(road[a][b]['travel_h'] for a,b in zip(r,r[1:])) for r in (r1,r2)]
    assert math.isclose(*lengths,rel_tol=1e-10)
    g=nx.DiGraph()
    nc=cfg['network']
    source,sink='SUPPLY','DEMAND'
    g.add_edge(source,src,capacity=nc['supplier_kits_week'],weight=0)
    for a,b,d in road.edges(data=True):
        for x,y in ((a,b),(b,a)):
            g.add_edge(x,y,capacity=nc['corridor_kits_week'],weight=max(1,round(d['travel_h']*60)))
    requests={s:max(1,math.ceil(value)) for s,value in kits.items()}
    for s,q in requests.items():
        g.add_edge(s,sink,capacity=min(q,nc['site_kits_week']),weight=0)
    flow=nx.max_flow_min_cost(g,source,sink)
    value=sum(flow[source].values())
    cut,_=nx.minimum_cut(g,source,sink)
    balances={n:sum(flow[u][n] for u in g.predecessors(n))-sum(flow[n].values()) for n in g}
    assert value==cut and all(0<=flow[a][b]<=d['capacity'] for a,b,d in g.edges(data=True))
    assert all(v==(value if n==sink else -value if n==source else 0) for n,v in balances.items())
    fixed=g.copy()
    fixed.nodes[source]['demand']=-value
    fixed.nodes[sink]['demand']=value
    cost_check,_=nx.network_simplex(fixed)
    assert cost_check==nx.cost_of_flow(g,flow)
    nx.set_edge_attributes(road,{(a,b):1/d['travel_h'] for a,b,d in road.edges(data=True)},'affinity')
    comm=nx.community.louvain_communities(road,weight='affinity',resolution=nc['community_resolution'],seed=nc['community_seed'])
    comm=sorted(comm,key=lambda s:(-len(s),min(s)))
    bc=nx.betweenness_centrality(road,weight='travel_h')
    closeness=nx.closeness_centrality(road,distance='travel_h')
    cluster=[]
    for i,c in enumerate(comm,1):
        best=min(c,key=lambda a:(sum(demand[b]*time[a][b] for b in c),a))
        cluster.append({'cluster':i,'stations':sorted(c),'medoid':best,
                        'medoid_label':graph.nodes[best]['label'],'size':len(c),'rate_per_up_hour':sum(demand[b] for b in c)})
    rankings=[]
    for site in road:
        if site==src: continue
        rankings.append({'site':site,'label':graph.nodes[site]['label'],
                         'weighted_travel_h':sum(demand[b]*min(time[src][b],time[site][b]) for b in demand)/sum(demand.values())})
    rankings.sort(key=lambda a:(a['weighted_travel_h'],a['site']))
    return {'node_count':len(graph),'edge_count':graph.number_of_edges(),'road_sites':len(road),'road_edges':road.number_of_edges(),
            'edge_types':dict(Counter(d['edge_type'] for *_,d in graph.edges(data=True))),
            'route':{'origin':src,'destination':target,'dijkstra':r1,'bellman_ford':r2,'hours':lengths[0],
                     'distance_km':sum(road[a][b]['distance_km'] for a,b in zip(r1,r1[1:]))},
            'flow':{'value_kits_week':value,'cut_kits_week':cut,'demand_kits_week':sum(requests.values()),
                    'cost_kit_minutes':nx.cost_of_flow(g,flow),'fixed_value_cost_check':cost_check,
                    'balances':balances,'requested_by_site':requests,
                    'served_by_site':{s:flow[s][sink] for s in requests},
                    'edges':[{'from':a,'to':b,'capacity_kits_week':d['capacity'],'cost_minutes_per_kit':d['weight'],'flow_kits_week':flow[a][b]}
                              for a,b,d in g.edges(data=True)]},
            'metrics':{'density':nx.density(road),'components':nx.number_connected_components(road),
                       'mean_degree':sum(dict(road.degree()).values())/len(road),
                       'mean_shortest_travel_h':nx.average_shortest_path_length(road,weight='travel_h'),
                       'average_clustering':nx.average_clustering(road),
                       'articulation_points':sorted(nx.articulation_points(road)),
                       'bridges':[list(x) for x in nx.bridges(road)],
                       'modularity':nx.community.modularity(road,comm,weight='affinity',resolution=nc['community_resolution'])},
            'critical_sites':[{'id':s,'label':graph.nodes[s]['label'],'betweenness':bc[s],'closeness_per_hour':closeness[s],
                               'degree':road.degree(s)} for s in sorted(road,key=lambda s:(-bc[s],s))],
            'clusters':cluster,'warehouse_geographic_ranking':rankings,
            'baseline_weighted_travel_h':sum(demand[b]*time[src][b] for b in demand)/sum(demand.values())}


def export_graph(root,graph,analysis,fit):
    data=root/'data'
    write_json(data/'graph_G.json',nx.node_link_data(graph,edges='edges'))
    fit.to_csv(data/'calibrated_rates.csv',index=False)
    fields=data/'field_dictionary.json'
    dictionary=load_json(fields)
    write_json(data/'field_dictionary_G.json',{'prior_vertices_dictionary':dictionary,
        'new_node_fields':{
            'rate_per_up_hour':'1/hour, shrunken MTA history', 'history_up_hours':'hour',
            'history_events':'outage count','history_months':'month count','rate_origin':'provenance label',
            'initial_kits':'kit count on catalog_stock edge','weight':'fault mix probability',
            'diagnosis_h':'hour, assumed mean','repair_h':'hour, assumed mean','max_kits':'kits',
            'cost_usd_kit':'USD/kit','home_station_id':'projected station node ID',
            'competency_profile':'prototype ID for a modeled duty position'},
        'edge_fields':{'edge_type':'relation type, directed','origin':'inventory or model assumption',
            'distance_km':'model km','travel_h':'model hours per trip','capacity_kits_week':'kits/week on flow projection',
            'max_kits':'kits/repair','initial_kits':'kits','active_baseline':'boolean'},
        'edge_types':analysis['edge_types'],'simulation_units':'config/coursework.json and report section 2'} )
