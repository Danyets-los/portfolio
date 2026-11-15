"""Figures for the reproducible report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np
from model_h import STATES,ARCS,wait_probability


def make_figures(out,g,road,a,h,samples,res,cfg):
    path=out/'figures'; path.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(name,fig):
        fig.savefig(path/(name+'.png'),dpi=cfg['figures']['dpi'],bbox_inches='tight',pad_inches=.14)
        plt.close(fig)

    pos={n:(g.nodes[n]['longitude'],g.nodes[n]['latitude']) for n in road}
    for mode in ('network','clusters'):
        fig,ax=plt.subplots(figsize=(7.6,5.5))
        nx.draw_networkx_edges(road,pos,ax=ax,edge_color='#839ab3',alpha=.43,width=.8)
        if mode=='network':
            nx.draw_networkx_nodes(road,pos,ax=ax,node_size=29,node_color='#174776')
            r=a['route']['dijkstra']
            nx.draw_networkx_edges(road,pos,edgelist=list(zip(r,r[1:])),ax=ax,edge_color='#b54627',width=2.2)
            for n,m,c in [(r[0],'s','#146f61'),(r[-1],'D','#b54627')]:
                ax.scatter(*pos[n],s=84,marker=m,c=c,zorder=6,label=n)
            ax.legend(title='Начало и конец маршрута',loc='upper left',bbox_to_anchor=(1.02,1))
            ax.set_title('Проекция G на станционные площадки')
        else:
            for c,color in zip(a['clusters'],plt.cm.tab20(np.linspace(0,1,len(a['clusters'])))):
                nx.draw_networkx_nodes(road,pos,nodelist=c['stations'],ax=ax,node_size=42,node_color=[color],
                                       edgecolors='white',linewidths=.6,label='Группа '+str(c['cluster']))
                ax.scatter(*pos[c['medoid']],marker='s',s=76,c=[color],edgecolors='#1e293b',linewidths=1.1,zorder=4)
            for rank,node in enumerate(a['critical_sites'][:3],1):
                ax.annotate(f'{rank}. {node["id"]}',pos[node['id']],xytext=(1.13,.53-rank*.075),
                            textcoords='axes fraction',fontsize=9,
                            arrowprops={'arrowstyle':'-','color':'#526477','lw':.9},
                            bbox={'boxstyle':'round,pad=.19','fc':'white','ec':'#c0cad5'},zorder=7)
            ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=8,title='Louvain; квадраты — медоиды')
            ax.set_title('Региональные группы и узлы с высокой посреднической ролью')
        ax.set_aspect(1/np.cos(np.radians(40.75)))
        save(mode,fig)

    fig,ax=plt.subplots(figsize=(9,4))
    coords=dict(zip(STATES,[(0,2),(1.4,2),(2.8,2),(4.2,2),(5.6,2.8),(7,2),
                             (7,.65),(5.25,.65),(3.5,.65),(1.75,.65)]))
    names=dict(zip(STATES,['Работа','Очередь\nдиагностики','Выезд','Диагностика','Ожидание\nзапаса',
                            'Доставка','Очередь\nремонта','Выезд','Ремонт','Контроль']))
    for i,j in ARCS:
        x,y=coords[STATES[i]]; u,v=coords[STATES[j]]
        dx,dy=u-x,v-y
        f=min(.566/abs(dx) if dx else float('inf'),.318/abs(dy) if dy else float('inf'))
        ax.add_patch(FancyArrowPatch((x+dx*f,y+dy*f),(u-dx*f,v-dy*f),arrowstyle='-|>',
                    mutation_scale=11,color='#54738f',lw=1.4,shrinkA=0,shrinkB=0,zorder=2))
    for n,(x,y) in coords.items():
        ax.add_patch(FancyBboxPatch((x-.53,y-.28),1.06,.56,boxstyle='round,pad=.03',
                                   fc='#dceaf4' if n!='W' else '#f3dfcb',ec='#66839d',zorder=3))
        ax.text(x,y,n+'\n'+names[n],ha='center',va='center',fontsize=8.4,zorder=4)
    ax.set(xlim=(-.65,7.65),ylim=(.2,3.25)); ax.axis('off')
    ax.set_title('Граф H  ·  самопереходы опущены для читаемости')
    save('states',fig)

    fig,ax=plt.subplots(figsize=(7.4,3.7))
    sc=cfg['simulation']; x=np.linspace(0,1,101)
    for u,e,col in [(sc['ordinary_urgency'],sc['priority_base'],'#174776'),
                    (sc['critical_urgency'],sc['priority_base'],'#146f61'),
                    (sc['critical_urgency'],sc['priority_busy'],'#b54627')]:
        ax.plot(x,[wait_probability(v,u,e,sc) for v in x],label=f'U={u:g}; E={e:g}',color=col,lw=2.2)
    ax.set(xlabel='A — доля доступного запаса',ylabel='Вероятность ожидания',xlim=(0,1),ylim=(0,1.03))
    ax.grid(alpha=.24); ax.legend()
    save('sensitivity',fig)

    fig,ax=plt.subplots(figsize=(7.4,3.8))
    data=np.array(samples)
    ax.hist(data,bins=range(0,int(data.max())+3,2),density=True,color='#557c9e',alpha=.83)
    ax.axvline(h['restoration_analytic_hours'],color='#b54627',lw=2,label='Аналитическое среднее')
    ax.set(xlabel='Время восстановления, ч',ylabel='Плотность',xlim=(0,data.max()+2))
    ax.legend(); ax.grid(axis='y',alpha=.24)
    save('distribution',fig)

    fig,axes=plt.subplots(1,2,figsize=(10.8,5.9),sharey=True)
    cases=list(res)[::-1]
    labels=[res[c]['case']['label'] for c in cases]
    for ax,key,label,scale in zip(axes,('mean_cycle_hours','cost_usd_day'),('Длительность цикла, ч','Полная стоимость, тыс. USD/сут'),(1,1e-3)):
        vals=[res[c]['kpi'][key] for c in cases]
        m=np.array([v['mean'] for v in vals])*scale
        e=np.array([(v['hi']-v['lo'])/2 for v in vals])*scale
        for i,c in enumerate(cases):
            ax.errorbar(m[i],i,xerr=e[i],fmt='o',capsize=3,color='#b54627' if c in ('stress','disruption') else '#174776')
        ax.axvline(res['baseline']['kpi'][key]['mean']*scale,color='#94a3b8',ls='--')
        ax.set(xlabel=label,yticks=range(len(cases))); ax.grid(axis='x',alpha=.24)
    axes[0].set_yticklabels(labels)
    fig.tight_layout()
    save('scenarios',fig)

    fig,ax=plt.subplots(figsize=(7.5,4.1))
    sku=res['baseline']['details']['sku_wait_hours']; names=list(sku)
    x=np.arange(len(names))
    ax.bar(x-.18,[sku[k] for k in names],.36,label='Базовый',color='#557c9e')
    ax.bar(x+.18,[res['disruption']['details']['sku_wait_hours'][k] for k in names],
           .36,label='Задержка STEP',color='#b56647')
    ax.set(xticks=x,xticklabels=[k.replace('MODEL-SKU-','') for k in names],
           ylabel='Среднее время ожидания запаса, машино-ч')
    ax.legend(); ax.grid(axis='y',alpha=.22)
    save('bottlenecks',fig)
