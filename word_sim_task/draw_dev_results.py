import plotly.graph_objs as go
from plotly.offline import plot
import pickle

with open('wn_ro_n_ld_1' + '.pickle', 'rb') as handle:
    results_L = pickle.load(handle)

with open('wn_ro_pd_ld_0' + '.pickle', 'rb') as handle:
    results = pickle.load(handle)


benchmark_scores_trace = []
results_trace = []
for name in ["SIMVERB500-dev"]:
    results_trace.append(
        go.Scatter(
            x=results['beta_range'],
            y=results[name+'_scores'],
            mode='lines+markers',
            name='HRSWE'
        )
    )
for name in ["SIMVERB500-dev"]:
    results_trace.append(
        go.Scatter(
            x=results_L['beta_range'],
            y=results_L[name+'_scores'],
            mode='lines+markers',
            name='LHRSWE'
        )
    )
plot({
        "data" : results_trace+benchmark_scores_trace,
        "layout": go.Layout(),
    },filename='combine_dev.html')