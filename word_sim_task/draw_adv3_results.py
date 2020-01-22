import pickle
from os.path import join

import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

ar_simlex = []
ar_simverb = []
hrswe_simlex = []
hrswe_simverb = []
ratios = [0.2,0.3,0.4,0.5]
for r in ratios:
    ta_dir = join('results', 'adv', '3', str(r)) # 3/3.1/3.2
    with open(join(ta_dir,'ar_results.pickle'), 'rb') as f:
        ar_res = pickle.load(f)
    with open(join(ta_dir,'hrswe_results.pickle'), 'rb') as f:
        hrswe_res = pickle.load(f)

    ar_simlex.append(ar_res['test_res']['SIMLEX999']*100)
    ar_simverb.append(ar_res['test_res']['SIMVERB3000-test']*100)

    hrswe_simlex.append(hrswe_res['test_res']['SIMLEX999']*100)
    hrswe_simverb.append(hrswe_res['test_res']['SIMVERB3000-test']*100)

trace1 = go.Scatter(
    x=ratios,
    y=ar_simlex,
    mode='lines+markers',
    name='AR_simlex'  # 'HRSWE'
)
trace2 = go.Scatter(
    x=ratios,
    y=hrswe_simlex,
    mode='lines+markers',
    name='HRSWE_simlex'  # 'HRSWE'
)
trace3 = go.Scatter(
    x=ratios,
    y=ar_simverb,
    mode='lines+markers',
    name='AR_simverb'  # 'HRSWE'
)
trace4 = go.Scatter(
    x=ratios,
    y=hrswe_simverb,
    mode='lines+markers',
    name='HRSWE_simverb'  # 'HRSWE'
)
plot({
    "data": [trace1,trace2],
    # "data": results_trace + benchmark_scores_trace,
    "layout": go.Layout(),
    "layout_margin":dict(l=0, r=0, t=0, b=0)
}, filename='simlex' + '.html')
plot({
    "data": [trace3,trace4],
    # "data": results_trace + benchmark_scores_trace,
    "layout": go.Layout(),
    "layout_margin":dict(l=0, r=0, t=0, b=0)
}, filename='simverb' + '.html')