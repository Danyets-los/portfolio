"""Finite-state reference and dynamic transition laws (time unit: hour)."""
import math
import numpy as np
from scipy import linalg, stats

STATES = ('U', 'QD', 'TD', 'D', 'W', 'L', 'QR', 'TR', 'R', 'C')
U, QD, TD, D, W, L, QR, TR, R, C = range(len(STATES))
ARCS = ((U,QD),(QD,TD),(TD,D),(D,W),(D,L),(W,L),(L,QR),(QR,TR),(TR,R),(R,C),(C,U))


def availability(stock, requirement):
    if requirement <= 0 or stock < 0:
        raise ValueError('requirement > 0 and stock >= 0 required')
    return min(stock / requirement, 1.)


def urgency(critical, cfg):
    return cfg['critical_urgency'] if critical else cfg['ordinary_urgency']


def priority(escalators_at_station, cfg):
    return cfg['priority_busy'] if escalators_at_station >= cfg['priority_site_escalators'] else cfg['priority_base']


def exponent(u, e, cfg):
    return math.exp(-cfg['urgency_beta'] * (u - cfg['ordinary_urgency'])
                    -cfg['priority_beta'] * (e - cfg['priority_base']))


def wait_probability(a, u, e, cfg):
    if not 0 <= a <= 1:
        raise ValueError('A must be in [0, 1]')
    return (1 - a) ** exponent(u, e, cfg)


def kit_demand(draw, limit, u, e, cfg):
    # Inversion of F(k)=1-(1-k/limit)^kappa for integer k=1..limit.
    return next(k for k in range(1, limit+1)
                if draw <= 1 - wait_probability(k/limit, u, e, cfg))


def progress(mean, cfg):
    dt = cfg['step_hours']
    return dt / max(dt * cfg['minimum_transition_steps'], mean)


def assignment(free_fraction, u, e, cfg):
    return -math.expm1(-cfg['dispatch_intensity'] * free_fraction * u * e)


def fail_probability(rate, step):
    return -np.expm1(-rate * step)


def reference_matrix(rate, fault, critical, eq_priority, cfg, ref):
    q = np.zeros((len(STATES),len(STATES)))
    u = urgency(critical, cfg)
    a = assignment(ref['free_fraction'], u, eq_priority, cfg)
    pw = wait_probability(ref['availability'], u, eq_priority, cfg)
    d = progress(fault['diagnosis_h'], cfg)
    for i,j,v in (
        (U,QD,fail_probability(rate,cfg['step_hours'])), (QD,TD,a),
        (TD,D,progress(ref['travel_h'],cfg)), (D,W,d*pw),(D,L,d*(1-pw)),
        (W,L,progress(ref['stock_wait_h'],cfg)),(L,QR,progress(ref['delivery_h'],cfg)),
        (QR,TR,a),(TR,R,progress(ref['travel_h'],cfg)),
        (R,C,progress(fault['repair_h'],cfg)),(C,U,progress(cfg['control_hours'],cfg))):
        q[i,j] = v
    return q + np.diag(1-q.sum(axis=1))


def t_interval(values, confidence=.95):
    x = np.asarray(values,dtype=float)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x)>1 else 0.
    half = float(stats.t.ppf((1+confidence)/2,len(x)-1)*sd/np.sqrt(len(x))) if len(x)>1 else 0.
    return {'mean':mean,'sd':sd,'lo':mean-half,'hi':mean+half,'n':int(len(x))}


def analytical_experiment(rate, fault, critical, eq_priority, cfg, ref):
    p = reference_matrix(rate, fault, critical, eq_priority, cfg, ref)
    if np.min(p)<0 or not np.allclose(p.sum(axis=1),1):
        raise ValueError('P not stochastic')
    n = len(STATES)
    eq = p.T-np.eye(n)
    eq[-1] = 1
    rhs = np.zeros(n)
    rhs[-1] = 1
    pi = linalg.solve(eq,rhs)
    dt = cfg['step_hours']
    mean = dt/(1-np.diag(p))
    tau = linalg.solve(np.eye(n-1)-p[1:,1:],np.full(n-1,dt))
    rng = np.random.default_rng(ref['seed'])
    size = ref['cycles']
    times = rng.geometric(1-np.diag(p),size=(size,n))*dt
    wait = rng.random(size) < p[D,W]/(1-p[D,D])
    times[:,W] *= wait
    restores = times[:,1:].sum(axis=1)
    duration = times.sum(axis=1)
    avail = float(times[:,U].sum()/duration.sum())
    influence = (times[:,U]-avail*duration)/duration.mean()
    ci_a = t_interval(influence,cfg['confidence'])
    for field in ('mean','lo','hi'):
        ci_a[field] += avail
    ci = t_interval(restores,cfg['confidence'])
    return {'states':list(STATES),'p':p.tolist(),'stationary':pi.tolist(),
            'stationary_residual':float(np.max(np.abs(pi@p-pi))),
            'holding_hours':dict(zip(STATES,mean.tolist())),
            'restoration_analytic_hours':float(tau[QD-1]),'availability_analytic':float(pi[U]),
            'restoration_mc':ci,'availability_mc':ci_a,'cycles':size,
            'analytic_in_ci_restoration':bool(ci['lo']<=tau[QD-1]<=ci['hi']),
            'analytic_in_ci_availability':bool(ci_a['lo']<=pi[U]<=ci_a['hi']),
            'transient':[{'hour':h,'p':np.linalg.matrix_power(p,round(h/dt))[U].tolist()} for h in ref['transient_hours']],
            'sample_restoration_hours':restores.tolist()}
