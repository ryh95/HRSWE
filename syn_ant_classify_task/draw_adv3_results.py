import pickle
from os.path import join

import plotly.graph_objects as go
from plotly.offline import plot

# ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ratios = [0.2,0.3,0.4,0.5]
models = ['ar', 'hrswe']

adj_f1s, n_f1s, verb_f1s = [], [], []
for model in models:
    adj_f1, n_f1, verb_f1 = [], [], []
    for r in ratios:
        ta_dir = join('results', 'adv', '3', str(r))
        with open(join(ta_dir, model + '_results.pickle'), 'rb') as f:
            res = pickle.load(f)
        for k,v in res['test_res'].items():
            word_type = k.split('-')[0]
            score_type = k.split('_')[-1]
            if score_type == 'f1':
                if word_type == 'adjective':
                    adj_f1.append(v*100)
                elif word_type == 'noun':
                    n_f1.append(v*100)
                elif word_type == 'verb':
                    verb_f1.append(v*100)
    adj_f1s.append(adj_f1)
    n_f1s.append(n_f1)
    verb_f1s.append(verb_f1)

adj_traces = []

for adj_f1,model in zip(adj_f1s,models):
    adj_traces.append(
        go.Scatter(
            x=ratios,
            y=adj_f1,
            mode='lines+markers',
            name=model.upper()+'_adj'  # 'HRSWE'
        )
    )

n_traces = []

for n_f1,model in zip(n_f1s,models):
    n_traces.append(
        go.Scatter(
            x=ratios,
            y=n_f1,
            mode='lines+markers',
            name=model.upper()+'_noun'  # 'HRSWE'
        )
    )

verb_traces = []
for verb_f1,model in zip(verb_f1s,models):
    verb_traces.append(
        go.Scatter(
            x=ratios,
            y=verb_f1,
            mode='lines+markers',
            name=model.upper()+'_verb'  # 'HRSWE'
        )
    )


plot({
    "data": adj_traces,
    "layout": go.Layout(),
    "layout_margin":dict(l=0, r=0, t=0, b=0)
}, filename='adjective' + '.html')
plot({
    "data": n_traces,
    "layout": go.Layout(),
    "layout_margin":dict(l=0, r=0, t=0, b=0)
}, filename='noun' + '.html')
plot({
    "data": verb_traces,
    "layout": go.Layout(),
    "layout_margin":dict(l=0, r=0, t=0, b=0)
}, filename='verb' + '.html')