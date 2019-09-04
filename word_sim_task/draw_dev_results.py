import plotly.graph_objs as go
from plotly.offline import plot
import pickle
import numpy as np

with open('wn_ro_n_ld_1' + '.pickle', 'rb') as handle:
    results_L = pickle.load(handle)

with open('wn_ro_pd_ld_0' + '.pickle', 'rb') as handle:
    results = pickle.load(handle)

x_grid,y_grid = np.meshgrid(results['beta_range1'],results['beta_range2'])
n = results['beta_range1'].size
z_grid = np.array(results['SIMVERB500-dev_scores']).reshape(n,n).T
z_l_grid = np.array(results_L['SIMVERB500-dev_scores']).reshape(n,n).T
# fig = go.Figure()
trace = go.Surface(
    x=x_grid,
    y=y_grid,
    z=z_grid,
    name='HRSWE',
)
trace_L = go.Surface(
    x=x_grid,
    y=y_grid,
    z=z_l_grid,
    showscale=False,
    name='LHRSWE',
)

plot({
        "data" : [trace,trace_L],
        "layout": go.Layout(
            scene=dict(
                xaxis=dict(
                    title='beta_1',
                    # gridcolor='rgb(255, 255, 255)',
                    # zerolinecolor='rgb(255, 255, 255)',
                    # showbackground=True,
                    # backgroundcolor='rgb(230, 230,230)'
                ),
                yaxis=dict(
                    title='beta_2',
                    # gridcolor='rgb(255, 255, 255)',
                    # zerolinecolor='rgb(255, 255, 255)',
                    # showbackground=True,
                    # backgroundcolor='rgb(230, 230,230)'
                ),
                zaxis=dict(
                    title='correlation',
                    # gridcolor='rgb(255, 255, 255)',
                    # zerolinecolor='rgb(255, 255, 255)',
                    # showbackground=True,
                    # backgroundcolor='rgb(230, 230,230)'
                ),
                # xaxis_title='beta_1',
                # yaxis_title='beta_2',
                # zaxis_title='correlation score'
            )
        ),
    },filename='combine_dev.html')