from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np


type_name = ['mse', 'negative log loss']

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Loss function'),
    dcc.Dropdown(
        id='dropdown',
        options=type_name,
        value="negative log loss",
        clearable=False
    ),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"), 
    Input("dropdown", "value"))
def display(name):
    #pts = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/ekaratnida/Applied-machine-learning/master/Week05-Logistic/data/convex.txt'))
    sFile = ''
    if name == 'mse':
        sFile = 'mse.txt'
    else:
        sFile = 'nll.txt'

    pts = np.loadtxt(sFile)
    x, y, z = pts.T
    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z)])
    return fig


app.run_server(debug=True)