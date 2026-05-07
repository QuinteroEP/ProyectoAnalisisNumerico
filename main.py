import json
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import plotly.graph_objs as go
import plotly.io as pio

def alg():
    x_coord = []
    y_coord = []

    with open('data.json', 'r') as file:
        data = json.load(file)

    puntos = data[0]["puntos"][0]
    for x_json, y_json in puntos:
        x_coord.append(x_json)
        y_coord.append(y_json)

    x = np.array(x_coord)
    y = np.array(y_coord)

    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    x_interp = np.linspace(np.min(x), np.max(x), 50)
    y_interp = interp1d(x, y, kind="cubic")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        name='Datos'
    ))

    fig.add_trace(go.Scatter(
        x=x_interp, y=y_interp,
        mode='lines',
        name='Spline Cubico'
    ))

    # Convert to HTML
    graph_html = pio.to_html(fig, full_html=False)