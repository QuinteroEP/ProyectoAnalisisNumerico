import os
from flask import Flask, render_template
import json
import numpy as np 
from scipy.interpolate import CubicSpline, CubicHermiteSpline
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px

app = Flask(__name__)

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    os.makedirs(app.instance_path, exist_ok=True)

    return app

@app.route('/')
def graphs():
    x_coord = []
    y_coord = []
    danger = []

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
    cs = CubicSpline(x,y)

    x_interp = np.linspace(np.min(x), np.max(x), 50)
    y_interp = cs(x_interp)

    F = cs.antiderivative()
    integrals = np.diff(F(x))

    print("integrales")
    for i, integral in enumerate(integrals):
        print(integral)
        if integral >= 90:
            danger.append(i)
    pmavg = round(sum(integrals)/len(integrals))

    if pmavg < 50:
        badge_class = "good"
        badge_text = "NORMAL"
    elif pmavg < 70:
        badge_class = "moderate"
        badge_text = "MODERADO"
    elif pmavg >= 90:
        badge_class = "bad"
        badge_text = "MALO"

    spline_fig = go.Figure()

    spline_fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        name='Sensores',
        marker=dict(color='LightSkyBlue', size=12, line=dict(color='white', width=1))
    ))

    spline_fig.add_trace(go.Scatter(
        x=x_interp, y=y_interp,
        mode='lines',
        name='Interpolación',
        line=dict(color='firebrick', dash='dash', width=4)
    ))

    spline_fig.update_layout(
        xaxis_title="Sensor",
        yaxis_title="PM2.5 (µg/m³)",
        plot_bgcolor="#121e23",
        paper_bgcolor="#121e23",
        font=dict(color="white")
    )

    bar_segs=[0,1,2,3,4,5,6,7,8]

    bar_fig = px.bar(
        x=bar_segs, y=integrals,
        labels={"x": "Sector", "y": "Exposicion total a PM2.5 (µg/m³)", "color": "Nivel de alerta"},
        color=integrals,
        color_continuous_scale=[
            (0.0, "green"),
            (0.7, "yellow"),
            (0.9, "orange"),
            (1.0, "red")
        ],
    )

    bar_fig.update_layout(
        plot_bgcolor="#121e23",
        paper_bgcolor="#121e23",
        font=dict(color="white")
    )

    # Convert to HTML
    splines_html = pio.to_html(spline_fig, full_html=False)
    bars_html = pio.to_html(bar_fig, full_html=False)

    return render_template('index.html', splines=splines_html, bars=bars_html, danger=danger, pmavg=pmavg, badge_class=badge_class, badge_text=badge_text)

@app.route('/verify')
def verification():
    x_coord = []
    y_coord = []

    with open('data.json', 'r') as file:
        data = json.load(file)

    puntos = data[0]["puntos"][0]
    for x_json, y_json in puntos:
        x_coord.append(x_json)
        y_coord.append(y_json)

    #Splines
    x = np.array(x_coord)
    y = np.array(y_coord)

    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    cs = CubicSpline(x,y)

    x_interp = np.linspace(np.min(x), np.max(x), 50)
    y_interp = cs(x_interp)

    #Hermite
    dydx = np.gradient(y, x)
    hermite = CubicHermiteSpline(x, y, dydx)

    x_hermite = np.linspace(np.min(x), np.max(x), 50)
    y_hermite = hermite(x_hermite)

    spline_fig = go.Figure()

    spline_fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        name='Datos Originales',
        marker=dict(color='LightSkyBlue', size=12, line=dict(color='white', width=1))
    ))

    spline_fig.add_trace(go.Scatter(
        x=x_interp, y=y_interp,
        mode='lines',
        name='Interpolación Por Splines Cubicos',
        line=dict(color='firebrick', width=4)
    ))

    spline_fig.add_trace(go.Scatter(
        x=x_hermite, y=y_hermite,
        mode='lines',
        name='Interpolación Por Hermite',
        line=dict(color='green', width=4)
    ))

    spline_fig.update_layout(
        xaxis_title="Sensor",
        yaxis_title="PM2.5 (µg/m³)",
        plot_bgcolor="#121e23",
        paper_bgcolor="#121e23",
        font=dict(color="white")
    )

    bootstrap_graph = bootstrap(x, y)
    error_calc = error(x, y)
    rmse = error_calc[0]
    error_graph = error_calc[1]

    bootstrap_html = pio.to_html(bootstrap_graph, full_html=False)
    splines_html = pio.to_html(spline_fig, full_html=False)
    error_html = pio.to_html(error_graph, full_html=False)

    return render_template('verification.html', splines=splines_html, bootstrap=bootstrap_html, rmse=rmse, errors=error_html)

def bootstrap(x, y):
    x_interp = np.linspace(np.min(x), np.max(x), 200)

    n_boot = 1000
    bootstrap_curves = []

    for _ in range(n_boot):
        # resample indices WITH replacement
        idx = np.random.choice(len(x), len(x), replace=True)

        x_boot = x[idx]
        y_boot = y[idx]

        # sort because splines require increasing x
        order = np.argsort(x_boot)
        x_boot = x_boot[order]
        y_boot = y_boot[order]

        # remove duplicate x values
        x_unique, unique_idx = np.unique(x_boot, return_index=True)
        y_unique = y_boot[unique_idx]

        # need enough points
        if len(x_unique) < 4:
            continue

        try:
            spline = CubicSpline(x_unique, y_unique)

            y_interp = spline(x_interp)

            bootstrap_curves.append(y_interp)

        except:
            pass

    bootstrap_curves = np.array(bootstrap_curves)

    # mean prediction
    mean_curve = np.mean(bootstrap_curves, axis=0)

    # 95% confidence interval
    lower = np.percentile(bootstrap_curves, 2.5, axis=0)
    upper = np.percentile(bootstrap_curves, 95.0, axis=0)

    boostrap_fig = go.Figure()

    boostrap_fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        name='Datos Originales',
        marker=dict(color='LightSkyBlue', size=10, line=dict(color='white', width=1))
    ))

    boostrap_fig.add_trace(go.Scatter(
        x=x_interp, y=lower,
        mode='lines',
        line=dict(color='green', width=1),
        name="2.5%"
    ))

    boostrap_fig.add_trace(go.Scatter(
        x=x_interp, y=upper,
        mode='lines',
        fill='tonexty',
        name="95%"
    ))

    boostrap_fig.add_trace(go.Scatter(
        x=x_interp, y=mean_curve,
        mode='lines',
        line=dict(color='firebrick', width=2),
        name="Bootstrap"
    ))

    boostrap_fig.update_layout(
        plot_bgcolor="#121e23",
        paper_bgcolor="#121e23",
        font=dict(color="white")
    )

    return boostrap_fig

def error(x, y):
    errors = []
    predictions = []

    for i in range(len(x)):
        x_train = np.delete(x, i)
        y_train = np.delete(y, i)

        spline = CubicSpline(x_train, y_train)

        y_pred = spline(x[i])

        predictions.append(y_pred)

        errors.append((y[i] - y_pred) ** 2)

    predictions = np.array(predictions)

    rmse = np.sqrt(np.mean(errors))
    abs_errors = np.abs(y - predictions)
    
    error_fig = go.Figure()
    
    error_fig.add_trace(
        go.Scatter(
            x=x,
            y=predictions,
            mode="markers",
            name="Predicciones",
            marker=dict(color='cyan', size=12, line=dict(color='white', width=1)),
            error_y=dict(
                type="data",
                array=abs_errors,
                visible=True
            )
        )
    )

    error_fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Valores reales",
            marker=dict(color='red', size=12, line=dict(color='white', width=1))
        )
    )

    error_fig.update_layout(
        plot_bgcolor="#121e23",
        paper_bgcolor="#121e23",
        font=dict(color="white")
    )

    return rmse, error_fig

if __name__ == "__main__":
    app.run(debug=True)
    