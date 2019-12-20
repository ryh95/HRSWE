import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

ar_simlex = []
ar_simverb = []
hrswe_simlex = []
hrswe_simverb = []
for r in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    if r == 0.3:
        ar_simlex.append(0.67)
        ar_simverb.append(0.63)
        hrswe_simlex.append(0.62)
        hrswe_simverb.append(0.61)
        continue
    elif r == 1.0:
        ar_simlex.append(0.45)
        ar_simverb.append(0.31)
        hrswe_simlex.append(0.50)
        hrswe_simverb.append(0.50)
        continue
    results = np.load('ar_results_'+str(r)+'.npy',allow_pickle=True).item()
    ar_simlex.append(float("{0:.2f}".format(results['SIMLEX999'])))
    ar_simverb.append(float("{0:.2f}".format(results['SIMVERB3000-test'])))
    results = np.load('hrswe_results_'+str(r)+'.npy',allow_pickle=True).item()
    hrswe_simlex.append(float("{0:.2f}".format( results['SIMLEX999'])))
    hrswe_simverb.append(float("{0:.2f}".format(results['SIMVERB3000-test'])))

trace1 = go.Scatter(
    x=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    y=ar_simlex,
    mode='lines+markers',
    name='AR_simlex'  # 'HRSWE'
)
trace2 = go.Scatter(
    x=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    y=hrswe_simlex,
    mode='lines+markers',
    name='HRSWE_simlex'  # 'HRSWE'
)
trace3 = go.Scatter(
    x=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    y=ar_simverb,
    mode='lines+markers',
    name='AR_simverb'  # 'HRSWE'
)
trace4 = go.Scatter(
    x=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    y=hrswe_simverb,
    mode='lines+markers',
    name='HRSWE_simverb'  # 'HRSWE'
)
plot({
    "data": [trace1,trace2],
    # "data": results_trace + benchmark_scores_trace,
    "layout": go.Layout(),
}, filename='simlex' + '.html')
plot({
    "data": [trace3,trace4],
    # "data": results_trace + benchmark_scores_trace,
    "layout": go.Layout(),
}, filename='simverb' + '.html')