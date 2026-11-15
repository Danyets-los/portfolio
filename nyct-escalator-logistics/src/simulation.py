"""Discrete-time Monte Carlo with shared queues, inventory and dispatch slots."""
from collections import Counter
from heapq import heappush, heappop
import math
import numpy as np
from model_g import roster
from model_h import (STATES,U,QD,TD,D,W,L,QR,TR,R,C, urgency,priority,
                     progress,kit_demand,availability,wait_probability,assignment,fail_probability)


def run_replication(g, travel, fit, faults, cfg, case, seed, keep_log=False):
    sc,inv,eco=cfg['simulation'],cfg['inventory'],cfg['economics']
    dt=sc['step_hours']
    horizon=round(sc['horizon_hours']/dt)
    warm=round(sc['warmup_hours']/dt)
    ids=fit.id.tolist()
    n=len(ids)
    site=[g.nodes[x]['station_id'] for x in ids]
    rates=fit.rate_per_up_hour.to_numpy(copy=True)
    pclass=np.cumsum([f['weight'] for f in faults])
    pclass[-1]=1
    priority_factor=np.array([priority(g.nodes[s]['escalator_count_in_scope'],sc) for s in site])
    if 'stress_fraction' in case:
        k=math.ceil(n*case['stress_fraction'])
        order=sorted(range(n),key=lambda i:(-rates[i],ids[i]))[:k]
        rates[order]*=case['stress_multiplier']
    p_fail=fail_probability(rates,dt)
    w0=cfg['baseline']['warehouse']
    wh_ids=[w0]
    if 'warehouse' in case:
        wh_ids.append(case['warehouse']['id'])
    wh_sites=[g.nodes[w]['anchor_station_id'] for w in wh_ids]
    engs=list(roster(cfg))
    eng_data=[{'id':e,'skills':set(g.nodes[e]['skills']),'home':g.nodes[e]['home_station_id'],'job':-1,'busy_steps':0} for e in engs]
    if 'hire' in case:
        e=case['hire']; engs.append(e['id'])
        eng_data.append({'id':e['id'],'skills':set(e['skills']),'home':e['home'],'job':-1,'busy_steps':0})
    if 'train' in case:
        eng_data[engs.index(case['train'])]['skills'].add(cfg['training_skill'])
    # Mutable, typed G node attributes for this replication; immutable topology is shared.
    node_state={x:{'sim_state':'U','downtime_h':0.,'fault_count':0} for x in ids}
    node_state.update({e['id']:e for e in eng_data})
    base=np.ceil(np.asarray(inv['base_stock'])*case.get('stock_multiplier',1)).astype(int)
    s0=np.ceil(np.asarray(inv['reorder'])*case.get('stock_multiplier',1)).astype(int)
    wh=[]
    for j,w in enumerate(wh_ids):
        if len(wh_ids)==1:
            cap, trigger=base.copy(),s0.copy()
        else:
            allocation=np.ceil(base*inv['split_share']).astype(int)
            trigger_allocation=np.ceil(s0*inv['split_share']).astype(int)
            cap=allocation if j==0 else base-allocation
            trigger=trigger_allocation if j==0 else s0-trigger_allocation
        d={'id':w,'site':wh_sites[j],'on_hand':cap.copy(),'order_up_to':cap,'reorder':trigger,
           'on_order':np.zeros(len(faults),dtype=int),'slots':inv['slots'],'occupied':0}
        if any(cap<np.array([f['max_kits'] for f in faults])) or any(trigger>=cap):
            raise ValueError('Warehouse inventory too small for kits or reorder policy')
        wh.append(d); node_state[w]=d
    etime=np.array([[travel[e['home']][s] for s in site] for e in eng_data])
    wtime=np.array([[travel[w['site']][s] for s in site] for w in wh])
    eligible=[[j for j,e in enumerate(eng_data) if f['skill'] in e['skills']] for f in faults]
    if any(not e for e in eligible):
        raise ValueError('A fault has no qualified resource')
    furg=[urgency(f['criticality']=='high',sc) for f in faults]
    dr=[progress(f['diagnosis_h'],sc) for f in faults]
    rp=[progress(f['repair_h'],sc) for f in faults]
    pc=progress(sc['control_hours'],sc)
    state=np.full(n,U,dtype=np.int8)
    fidx=np.full(n,-1,dtype=int)
    worker=np.full(n,-1,dtype=int)
    chosen_wh=np.full(n,-1,dtype=int)
    q=np.zeros(n,dtype=int)
    opened=np.full(n,-1,dtype=int)
    since=np.full(n,-1,dtype=int)
    p_step=np.zeros(n)
    in_delivery=np.zeros(n,dtype=bool)
    had_stockout=np.zeros(n,dtype=bool)
    jobs=set()
    replen=[]
    rng=np.random.default_rng(seed)
    count_states=np.zeros(len(STATES),dtype=np.int64)
    engineer_busy=np.zeros(len(engs),dtype=int)
    equip_counts=np.zeros(n,dtype=int)
    sku_used=np.zeros(len(faults),dtype=int)
    sku_short=np.zeros(len(faults),dtype=int)
    wait_by_fault=np.zeros(len(faults),dtype=int)
    site_wait=Counter()
    busy_wh=np.zeros(len(wh),dtype=int)
    delivery_cost=0.
    inv_hours=0.
    time_sample=[]
    event_log=[]
    jobs_opened=0
    delays_other_queue=0
    checks={'stock_never_negative':True,'slot_capacity_respected':True,'unique_engineer_assignment':True,
            'inventory_conservation':True,'jobs_conservation':True}
    withdrawals=np.zeros((len(wh),len(faults)),dtype=int)
    arrivals=np.zeros_like(withdrawals)
    start_stock=np.stack([w['on_hand'].copy() for w in wh])
    max_slots=np.zeros(len(wh),dtype=int)

    def log(t,i,event,**extra):
        if keep_log:
            event_log.append({'hour':t*dt,'asset_id':ids[i],'site':site[i],
                              'fault':faults[fidx[i]]['id'] if fidx[i]>=0 else None,
                              'event':event,**extra})

    def change(i,target,t):
        origin=int(state[i]); state[i]=target; since[i]=t
        node_state[ids[i]]['sim_state']=STATES[target]
        log(t,i,'transition',source=STATES[origin],target=STATES[target])

    def release(i):
        j=worker[i]
        if j<0 or eng_data[j]['job']!=i:
            raise AssertionError('release without resource')
        eng_data[j]['job']=-1; worker[i]=-1

    def replenish(j,k,t):
        w=wh[j]
        position=int(w['on_hand'][k]+w['on_order'][k])
        if position <= w['reorder'][k]:
            count=int(w['order_up_to'][k]-position)
            lead=inv['lead_hours']*(case['lead_multiplier'] if case.get('disruption_sku')==k else 1)
            w['on_order'][k]+=count
            heappush(replen,(t+math.ceil(lead/dt),j,k,count))

    def reserve(i,k,t):
        nonlocal delivery_cost
        options=[j for j,w in enumerate(wh) if w['on_hand'][k]>=q[i]]
        if not options:
            return False
        j=min(options,key=lambda j:(wtime[j,i],wh[j]['occupied'],wh[j]['id']))
        w=wh[j]
        w['on_hand'][k]-=q[i]; withdrawals[j,k]+=q[i]
        assert w['on_hand'][k]>=0
        chosen_wh[i]=j
        if t>warm:
            sku_used[k]+=q[i]
            delivery_cost+=eco['delivery_usd_per_hour']*wtime[j,i]
        replenish(j,k,t)
        change(i,L,t)
        log(t,i,'reserved',warehouse=w['id'],kits=int(q[i]),stock_after=int(w['on_hand'][k]))
        return True

    for tick in range(horizon):
        t=tick+1
        random=rng.random((5,n))
        while replen and replen[0][0]<=tick:
            when,j,k,count=heappop(replen)
            w=wh[j]; w['on_hand'][k]+=count; w['on_order'][k]-=count; arrivals[j,k]+=count
            if keep_log:
                event_log.append({'hour':tick*dt,'event':'replenished','warehouse':w['id'],
                                  'sku':faults[k]['part_id'],'kits':count,'stock_after':int(w['on_hand'][k])})
        if tick>=warm:
            count_states+=np.bincount(state,minlength=len(STATES))
            for j,e in enumerate(eng_data):
                engineer_busy[j]+=(e['job']>=0)
            for j,w in enumerate(wh):
                busy_wh[j]+=w['occupied']
                inv_hours+=w['on_hand'].sum()*dt
        old_jobs=sorted(jobs)
        queues=[]
        deliveries=[]
        for i in old_jobs:
            st=int(state[i]); k=int(fidx[i])
            if tick>=warm:
                if st==W:
                    wait_by_fault[k]+=1; site_wait[site[i]]+=1
                if st in (QD,QR) and k!=cfg['comparison_sku_index']:
                    delays_other_queue+=1
            if st in (QD,QR):
                queues.append(i)
            elif st==W:
                reserve(i,k,t)
            elif st==L:
                j=chosen_wh[i]
                if not in_delivery[i]:
                    deliveries.append(i)
                elif random[2,i]<p_step[i]:
                    wh[j]['occupied']-=1; in_delivery[i]=False
                    change(i,QR,t)
            elif random[2,i]<p_step[i]:
                if st==TD:
                    change(i,D,t); p_step[i]=dr[k]
                elif st==D:
                    release(i)
                    a=availability(max(w['on_hand'][k] for w in wh),faults[k]['max_kits'])
                    p_wait=wait_probability(a,furg[k],priority_factor[i],sc)
                    q[i]=kit_demand(random[3,i],faults[k]['max_kits'],furg[k],priority_factor[i],sc)
                    ok=reserve(i,k,t)
                    # Uses the same quantile: P(q > max stock)=p_wait.
                    assert ok == (random[3,i] <= 1-p_wait)
                    log(t,i,'diagnosis_decision',availability=a,p_wait=p_wait,required_kits=int(q[i]))
                    if not ok:
                        had_stockout[i]=True
                        if tick>=warm:
                            sku_short[k]+=1
                        change(i,W,t)
                elif st==TR:
                    change(i,R,t); p_step[i]=rp[k]
                elif st==R:
                    change(i,C,t); p_step[i]=pc
                elif st==C:
                    dur=(t-opened[i])*dt
                    node_state[ids[i]]['downtime_h']+=dur
                    if opened[i]>warm:
                        time_sample.append((dur,bool(had_stockout[i]),ids[i],faults[k]['id']))
                    release(i); change(i,U,t); jobs.remove(i)
        def rank(i):
            return (-(furg[fidx[i]]*priority_factor[i]+sc['priority_age_per_hour']*(tick-since[i])*dt),since[i],ids[i])
        for i in sorted(deliveries,key=rank):
            j=chosen_wh[i]; w=wh[j]
            if w['occupied']<w['slots']:
                w['occupied']+=1; in_delivery[i]=True
                max_slots[j]=max(max_slots[j],w['occupied'])
                p_step[i]=progress(wtime[j,i]+inv['handling_hours'],sc)
                log(t,i,'dispatch',warehouse=w['id'],occupied=w['occupied'],capacity=w['slots'])
        for i in sorted(queues,key=rank):
            k=fidx[i]
            capable=eligible[k]
            free=[j for j in capable if eng_data[j]['job']<0]
            p_assign=assignment(len(free)/len(capable),furg[k],priority_factor[i],sc)
            if free and random[4,i]<p_assign:
                j=min(free,key=lambda x:(etime[x,i],engs[x]))
                assert worker[i]<0
                eng_data[j]['job']=i; worker[i]=j
                change(i,TD if state[i]==QD else TR,t)
                p_step[i]=progress(etime[j,i],sc)
                log(t,i,'assigned',engineer=engs[j],p_assign=p_assign)
        # Prevent a just-repaired machine from failing in an interval it did not work.
        possible=(state==U)&(since<t)&(random[0]<p_fail)
        for i in np.flatnonzero(possible):
            i=int(i)
            fidx[i]=min(len(faults)-1,np.searchsorted(pclass,random[1,i],side='right'))
            opened[i]=t; had_stockout[i]=False; q[i]=0; chosen_wh[i]=-1
            jobs.add(i); node_state[ids[i]]['fault_count']+=1
            if tick>=warm:
                equip_counts[i]+=1; jobs_opened+=1
            change(i,QD,t)
    for j,w in enumerate(wh):
        assert np.array_equal(start_stock[j]+arrivals[j]-withdrawals[j],w['on_hand'])
        assert np.all(w['on_hand']>=0) and np.all(w['on_order']>=0) and max_slots[j]<=w['slots']
    assert len({e['job'] for e in eng_data if e['job']>=0})==sum(e['job']>=0 for e in eng_data)
    assert len(time_sample)+sum(opened[i]>warm for i in jobs)==jobs_opened
    assert count_states.sum()==n*(horizon-warm)
    eval_h=(horizon-warm)*dt
    durations=[x[0] for x in time_sample]
    if not durations:
        raise ValueError('No observed restoration after warm-up')
    down=(count_states.sum()-count_states[U])*dt
    costs={'downtime':float(down*eco['downtime_usd_h']),'engineers':eval_h*len(engs)*eco['wage_usd_h'],
           'parts':float(sku_used@eco['kit_usd']),'holding':float(inv_hours*eco['holding_usd_kit_h']),
           'warehouse':len(wh)*eco['warehouse_usd_day']*eval_h/24,
           'training':eco['training_usd']*eval_h/eco['training_amortization_hours'] if 'train' in case else 0.,
           'delivery':float(delivery_cost)}
    kpi={'mean_cycle_hours':float(np.mean(durations)),'p90_cycle_hours':float(np.quantile(durations,.9)),
         'availability':float(count_states[U]/count_states.sum()),
         'utilization':float(engineer_busy.sum()/len(engs)/(horizon-warm)),
         'max_engineer_utilization':float(engineer_busy.max()/(horizon-warm)),
         'cost_usd_day':sum(costs.values())/(eval_h/24),
         'downtime_hours':float(down),'cycles_completed':len(durations),'failures_started':jobs_opened,
         'unfinished_cohort':sum(opened[i]>warm for i in jobs),
         'stockout_share':sum(x[1] for x in time_sample)/len(time_sample),
         'mean_inventory_kits':float(inv_hours/eval_h),
         'non_step_queue_h':delays_other_queue*dt,
         'parts_wait_hours':int(count_states[W])*dt,
         'warehouse_utilization':float(busy_wh.sum()/sum(w['slots'] for w in wh)/(horizon-warm))}
    for key,c in costs.items():
        kpi['cost_'+key+'_usd_day']=c/(eval_h/24)
    detail={'engineer_utilization':{e:float(engineer_busy[j]/(horizon-warm)) for j,e in enumerate(engs)},
            'warehouse_utilization':{w['id']:float(busy_wh[j]/w['slots']/(horizon-warm)) for j,w in enumerate(wh)},
            'sku_shortages':dict(zip((f['part_id'] for f in faults),sku_short.tolist())),
            'sku_used':dict(zip((f['part_id'] for f in faults),sku_used.tolist())),
            'sku_wait_hours':{f['part_id']:int(wait_by_fault[j])*dt for j,f in enumerate(faults)},
            'asset_failures':dict(zip(ids,equip_counts.tolist())), 'site_wait_hours':dict(site_wait),
            'state_hours':{s:int(count_states[j])*dt for j,s in enumerate(STATES)},'checks':checks}
    if keep_log:
        detail['events']=event_log
        detail['cycles']=[{'hours':h,'stockout':b,'asset_id':i,'fault':f} for h,b,i,f in time_sample]
    return kpi,detail
