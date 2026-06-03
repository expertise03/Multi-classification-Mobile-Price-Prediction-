"""
Mobile Price Prediction Dashboard
Multiclass Classification using KNN, Logistic Regression, and SVM
Run: pip install pandas numpy scikit-learn matplotlib seaborn plotly dash
Then: python dashboard.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64, warnings
warnings.filterwarnings("ignore")

# ── Scikit-learn ───────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ── Dash / Plotly ──────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# ══════════════════════════════════════════════════════════════════════════════
#  1.  DATA  (synthetic if CSV not found – works out of the box)
# ══════════════════════════════════════════════════════════════════════════════
try:
    df = pd.read_csv("mobile_price.csv")
    print("✅ Loaded mobile_price.csv")
except FileNotFoundError:
    print("⚠️  mobile_price.csv not found – generating synthetic data …")
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        "battery_power": np.random.randint(500, 2000, n),
        "blue":          np.random.randint(0, 2, n),
        "clock_speed":   np.round(np.random.uniform(0.5, 3.0, n), 1),
        "dual_sim":      np.random.randint(0, 2, n),
        "fc":            np.random.randint(0, 20, n),
        "four_g":        np.random.randint(0, 2, n),
        "int_memory":    np.random.randint(2, 64, n),
        "m_dep":         np.round(np.random.uniform(0.1, 1.0, n), 1),
        "mobile_wt":     np.random.randint(80, 200, n),
        "n_cores":       np.random.randint(1, 8, n),
        "pc":            np.random.randint(0, 20, n),
        "px_height":     np.random.randint(0, 1960, n),
        "px_width":      np.random.randint(500, 1998, n),
        "ram":           np.random.randint(256, 3998, n),
        "sc_h":          np.random.randint(5, 19, n),
        "sc_w":          np.random.randint(0, 18, n),
        "talk_time":     np.random.randint(2, 20, n),
        "three_g":       np.random.randint(0, 2, n),
        "touch_screen":  np.random.randint(0, 2, n),
        "wifi":          np.random.randint(0, 2, n),
    })
    # Price range driven by RAM + battery + px_width (realistic signal)
    score = (df["ram"] / 4000 + df["battery_power"] / 2000 + df["px_width"] / 2000)
    df["price_range"] = pd.cut(score, bins=4, labels=[0, 1, 2, 3]).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
#  2.  PREPROCESSING & MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
X = df.drop("price_range", axis=1)
y = df["price_range"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM":                 SVC(kernel="rbf", random_state=42, probability=True),
}

results, preds, reports = {}, {}, {}
for name, m in models.items():
    m.fit(X_train_s, y_train)
    y_pred = m.predict(X_test_s)
    cv     = cross_val_score(m, X_train_s, y_train, cv=5, scoring="accuracy")
    results[name] = {
        "Test Accuracy":  accuracy_score(y_test, y_pred),
        "CV Mean":        cv.mean(),
        "CV Std":         cv.std(),
    }
    preds[name]   = y_pred
    reports[name] = classification_report(y_test, y_pred, output_dict=True)
    print(f"  {name:<22} Acc={results[name]['Test Accuracy']:.4f}  CV={cv.mean():.4f}±{cv.std():.4f}")

best_model = max(results, key=lambda k: results[k]["Test Accuracy"])
print(f"\n🏆 Best model: {best_model}")

# ══════════════════════════════════════════════════════════════════════════════
#  3.  FIGURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
PRICE_LABELS = ["Budget (0)", "Economy (1)", "Mid-Range (2)", "Premium (3)"]
COLORS = ["#00d2ff", "#7b2ff7", "#f7b731", "#ff4757"]
BG     = "#0d0f1a"
CARD   = "#13162a"
TEXT   = "#e8ecff"
ACCENT = "#7b2ff7"
GRID   = "rgba(255,255,255,0.06)"

LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=TEXT, family="'DM Sans', sans-serif", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
)

def fig_accuracy_bar():
    names = list(results.keys())
    accs  = [results[n]["Test Accuracy"] for n in names]
    cvs   = [results[n]["CV Mean"] for n in names]
    stds  = [results[n]["CV Std"] for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Test Accuracy", x=names, y=accs,
                         marker_color=["#00d2ff","#7b2ff7","#f7b731"],
                         text=[f"{a:.2%}" for a in accs], textposition="outside"))
    fig.add_trace(go.Bar(name="CV Mean ± Std", x=names, y=cvs,
                         error_y=dict(type="data", array=stds, visible=True),
                         marker_color=["#00d2ff88","#7b2ff788","#f7b73188"],
                         text=[f"{c:.2%}" for c in cvs], textposition="outside"))
    fig.update_layout(**LAYOUT_BASE, title="Model Accuracy Comparison",
                      barmode="group", yaxis=dict(tickformat=".0%", gridcolor=GRID, range=[0, 1.1]),
                      xaxis=dict(gridcolor=GRID), legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def fig_confusion(model_name):
    cm = confusion_matrix(y_test, preds[model_name])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Purples",
                    labels=dict(x="Predicted", y="Actual"),
                    x=PRICE_LABELS, y=PRICE_LABELS)
    fig.update_layout(**LAYOUT_BASE, title=f"Confusion Matrix – {model_name}",
                      coloraxis_showscale=False)
    return fig

def fig_corr_heatmap():
    corr = df.corr(numeric_only=True)
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    text_auto=".2f")
    fig.update_layout(**LAYOUT_BASE, title="Feature Correlation Heatmap",
                      height=550, margin=dict(l=80, r=20, t=50, b=80))
    fig.update_traces(textfont_size=8)
    return fig

def fig_feature_importance():
    corr  = df.corr(numeric_only=True)["price_range"].drop("price_range")
    corr_sorted = corr.abs().sort_values(ascending=True)
    colors_ = [COLORS[3] if corr[i] > 0 else COLORS[0] for i in corr_sorted.index]
    fig = go.Figure(go.Bar(
        y=corr_sorted.index, x=corr_sorted.values,
        orientation="h", marker_color=colors_,
        text=[f"{v:.3f}" for v in corr_sorted.values], textposition="outside"
    ))
    fig.update_layout(**LAYOUT_BASE, title="Feature Importance (|Correlation with Price Range|)",
                      xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID), height=550)
    return fig

def fig_price_dist():
    fig = px.histogram(df, x="price_range", color="price_range",
                       color_discrete_sequence=COLORS,
                       category_orders={"price_range": [0,1,2,3]})
    fig.update_layout(**LAYOUT_BASE, title="Price Range Distribution",
                      showlegend=False, bargap=0.15,
                      xaxis=dict(tickvals=[0,1,2,3], ticktext=PRICE_LABELS, gridcolor=GRID),
                      yaxis=dict(gridcolor=GRID))
    return fig

def fig_ram_vs_price():
    fig = px.box(df, x="price_range", y="ram", color="price_range",
                 color_discrete_sequence=COLORS,
                 category_orders={"price_range": [0,1,2,3]})
    fig.update_layout(**LAYOUT_BASE, title="RAM Distribution by Price Range",
                      showlegend=False,
                      xaxis=dict(tickvals=[0,1,2,3], ticktext=PRICE_LABELS, gridcolor=GRID),
                      yaxis=dict(gridcolor=GRID))
    return fig

def fig_cv_radar():
    categories = list(results.keys())
    metrics = ["Test Accuracy", "CV Mean"]
    fig = go.Figure()
    for m_name, color in zip(metrics, [COLORS[0], COLORS[1]]):
        vals = [results[k][m_name] for k in categories]
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=categories + [categories[0]],
                                      fill="toself", name=m_name,
                                      line_color=color, fillcolor=color.replace(")", ",0.15)").replace("rgb","rgba") if "rgb" in color else color + "26"))
    fig.update_layout(**LAYOUT_BASE, title="Test vs CV Accuracy Radar",
                      polar=dict(bgcolor=CARD,
                                 radialaxis=dict(visible=True, range=[0.8, 1.0],
                                                 gridcolor=GRID, color=TEXT),
                                 angularaxis=dict(gridcolor=GRID)),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def make_metrics_table():
    rows = []
    for name in results:
        r = reports[name]
        rows.append({
            "Model":           name,
            "Accuracy":        f"{results[name]['Test Accuracy']:.4f}",
            "CV Mean":         f"{results[name]['CV Mean']:.4f}",
            "CV Std":          f"±{results[name]['CV Std']:.4f}",
            "Precision (W)":   f"{r['weighted avg']['precision']:.4f}",
            "Recall (W)":      f"{r['weighted avg']['recall']:.4f}",
            "F1-Score (W)":    f"{r['weighted avg']['f1-score']:.4f}",
        })
    return rows

# ══════════════════════════════════════════════════════════════════════════════
#  4.  DASH APP
# ══════════════════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.DARKLY,
    "https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600;700&family=Space+Grotesk:wght@700&display=swap"
])
app.title = "📱 Mobile Price Predictor"

CARD_STYLE = {
    "background":   CARD,
    "borderRadius": "16px",
    "padding":      "20px",
    "marginBottom": "18px",
    "boxShadow":    "0 4px 30px rgba(0,0,0,0.5)",
    "border":       "1px solid rgba(123,47,247,0.18)",
}

def kpi_card(title, value, subtitle="", color="#00d2ff"):
    return html.Div([
        html.P(title, style={"color": "rgba(255,255,255,0.5)", "fontSize": "12px",
                              "marginBottom": "4px", "textTransform": "uppercase",
                              "letterSpacing": "1px"}),
        html.H2(value, style={"color": color, "fontFamily": "'Space Grotesk', sans-serif",
                               "fontSize": "2rem", "margin": "0"}),
        html.P(subtitle, style={"color": "rgba(255,255,255,0.4)", "fontSize": "11px",
                                 "margin": "4px 0 0"}),
    ], style={**CARD_STYLE, "textAlign": "center"})

best_acc  = results[best_model]["Test Accuracy"]
total_rows = len(df)
n_feats    = X.shape[1]

app.layout = html.Div(style={"backgroundColor": BG, "minHeight": "100vh",
                               "fontFamily": "'DM Sans', sans-serif", "color": TEXT}, children=[

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("📱", style={"fontSize": "2.5rem"}),
            html.Div([
                html.H1("Mobile Price Prediction",
                        style={"fontFamily": "'Space Grotesk', sans-serif",
                               "fontSize": "2rem", "margin": "0",
                               "background": "linear-gradient(90deg,#00d2ff,#7b2ff7)",
                               "-webkit-background-clip": "text",
                               "-webkit-text-fill-color": "transparent"}),
                html.P("Multiclass Classification Dashboard  ·  KNN · Logistic Regression · SVM",
                       style={"margin": "0", "color": "rgba(255,255,255,0.45)", "fontSize": "13px"}),
            ]),
        ], style={"display": "flex", "alignItems": "center", "gap": "18px"}),
        html.Div([
            html.Span(f"🏆 Best: {best_model}",
                      style={"background": "linear-gradient(90deg,#7b2ff7,#00d2ff)",
                             "borderRadius": "50px", "padding": "6px 18px",
                             "fontSize": "13px", "fontWeight": "600"}),
        ]),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
               "padding": "24px 32px", "borderBottom": "1px solid rgba(255,255,255,0.06)"}),

    # ── KPI Row ───────────────────────────────────────────────────────────────
    html.Div([
        dbc.Col(kpi_card("Best Accuracy",   f"{best_acc:.2%}", best_model, "#00d2ff"), md=3),
        dbc.Col(kpi_card("Dataset Size",    f"{total_rows:,}", "samples", "#7b2ff7"),  md=3),
        dbc.Col(kpi_card("Features",        str(n_feats), "input variables", "#f7b731"), md=3),
        dbc.Col(kpi_card("Price Classes",   "4", "Budget → Premium", "#ff4757"),      md=3),
    ], style={"display": "flex", "gap": "16px", "padding": "24px 32px 0"}),

    # ── Row 1: Accuracy Bar + Radar ───────────────────────────────────────────
    html.Div([
        html.Div([dcc.Graph(figure=fig_accuracy_bar(), config={"displayModeBar": False})],
                 style={**CARD_STYLE, "flex": "1.6"}),
        html.Div([dcc.Graph(figure=fig_cv_radar(), config={"displayModeBar": False})],
                 style={**CARD_STYLE, "flex": "1"}),
    ], style={"display": "flex", "gap": "18px", "padding": "18px 32px 0"}),

    # ── Row 2: Confusion Matrix (interactive) ─────────────────────────────────
    html.Div([
        html.Div([
            html.P("Select Model to View Confusion Matrix",
                   style={"color": "rgba(255,255,255,0.5)", "fontSize": "12px",
                          "textTransform": "uppercase", "letterSpacing": "1px",
                          "marginBottom": "10px"}),
            dcc.RadioItems(
                id="model-radio",
                options=[{"label": k, "value": k} for k in models],
                value=best_model,
                inline=True,
                style={"color": TEXT, "gap": "24px", "display": "flex"},
                inputStyle={"marginRight": "6px", "accentColor": ACCENT},
                labelStyle={"marginRight": "24px"},
            ),
            dcc.Graph(id="confusion-graph", config={"displayModeBar": False}),
        ], style={**CARD_STYLE, "flex": "1"}),

        html.Div([
            html.P("Per-Class F1-Score",
                   style={"color": "rgba(255,255,255,0.5)", "fontSize": "12px",
                          "textTransform": "uppercase", "letterSpacing": "1px",
                          "marginBottom": "10px"}),
            dcc.Graph(id="f1-graph", config={"displayModeBar": False}),
        ], style={**CARD_STYLE, "flex": "1"}),
    ], style={"display": "flex", "gap": "18px", "padding": "0 32px"}),

    # ── Row 3: Feature correlation + distribution ──────────────────────────────
    html.Div([
        html.Div([dcc.Graph(figure=fig_feature_importance(), config={"displayModeBar": False})],
                 style={**CARD_STYLE, "flex": "1"}),
        html.Div([
            dcc.Graph(figure=fig_price_dist(), config={"displayModeBar": False},
                      style={"marginBottom": "0"}),
            dcc.Graph(figure=fig_ram_vs_price(), config={"displayModeBar": False}),
        ], style={**CARD_STYLE, "flex": "1"}),
    ], style={"display": "flex", "gap": "18px", "padding": "0 32px"}),

    # ── Row 4: Full Correlation Heatmap ───────────────────────────────────────
    html.Div([
        dcc.Graph(figure=fig_corr_heatmap(), config={"displayModeBar": False}),
    ], style={**CARD_STYLE, "margin": "0 32px"}),

    # ── Row 5: Metrics Table ──────────────────────────────────────────────────
    html.Div([
        html.P("Detailed Model Metrics",
               style={"color": "rgba(255,255,255,0.5)", "fontSize": "12px",
                      "textTransform": "uppercase", "letterSpacing": "1px",
                      "marginBottom": "12px"}),
        dash_table.DataTable(
            data=make_metrics_table(),
            columns=[{"name": c, "id": c} for c in make_metrics_table()[0].keys()],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#1a1d35", "color": "#00d2ff",
                          "fontWeight": "700", "border": "1px solid rgba(255,255,255,0.08)"},
            style_cell={"backgroundColor": CARD, "color": TEXT, "textAlign": "center",
                        "border": "1px solid rgba(255,255,255,0.06)", "padding": "10px 16px"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#0f1225"},
            ],
        ),
    ], style={**CARD_STYLE, "margin": "18px 32px 32px"}),
])

# ── Callbacks ──────────────────────────────────────────────────────────────────
@app.callback(
    Output("confusion-graph", "figure"),
    Output("f1-graph", "figure"),
    Input("model-radio", "value"),
)
def update_confusion(model_name):
    cm_fig = fig_confusion(model_name)

    # Per-class F1
    r = reports[model_name]
    classes = [str(i) for i in range(4)]
    f1s = [r[c]["f1-score"] for c in classes]
    f1_fig = go.Figure(go.Bar(
        x=PRICE_LABELS, y=f1s,
        marker_color=COLORS,
        text=[f"{v:.3f}" for v in f1s], textposition="outside"
    ))
    f1_fig.update_layout(**LAYOUT_BASE,
                          title=f"Per-Class F1-Score – {model_name}",
                          yaxis=dict(range=[0, 1.1], gridcolor=GRID),
                          xaxis=dict(gridcolor=GRID))
    return cm_fig, f1_fig


if __name__ == "__main__":
    print("\n🚀 Starting dashboard at http://127.0.0.1:8050\n")
    app.run(debug=True, port=8050)
