import pandas as pd
import plotly.graph_objects as go
import os

# CONFIGURATION
INPUT_CSV = os.path.join("..", "data", "processed", "turbulence_points_3d.csv")
OUTPUT_HTML = os.path.join("..", "docs", "reports", "turbulence_3d_map.html")

def visualize():
    if not os.path.exists(INPUT_CSV):
        print(f"Erreur : {INPUT_CSV} introuvable. Lancez turbulence_to_3d.py d'abord.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Création de la figure 3D
    fig = go.Figure()

    # 1. La trajectoire de l'avion
    fig.add_trace(go.Scatter3d(
        x=df['lon'],
        y=df['lat'],
        z=df['alt'],
        mode='lines',
        line=dict(color='gray', width=2),
        name='Trajectoire de l\'avion'
    ))

    # 2. Les points de détection (Colorés par intensité)
    fig.add_trace(go.Scatter3d(
        x=df['lon'],
        y=df['lat'],
        z=df['alt'],
        mode='markers',
        marker=dict(
            size=5 + df['intensity'] * 10,
            color=df['intensity'],
            colorscale='Viridis',
            colorbar=dict(title="Probabilité de Turbulence"),
            opacity=0.8
        ),
        text=[f"Point {i}<br>Confiance: {p:.2%}<br>Alt: {a:.0f}m" 
              for i, p, a in zip(df.index, df['intensity'], df['alt'])],
        name='Alertes IA'
    ))

    # Mise en page (Styling pour un rendu "Premium")
    fig.update_layout(
        title="TurbulenceWatch - Visualisation 3D de la Trajectoire et Détections",
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Altitude (m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
            bgcolor="rgb(10, 10, 20)" # Fond sombre pour un look tech
        ),
        margin=dict(r=0, l=0, b=0, t=40),
        template="plotly_dark"
    )

    # Sauvegarde
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    fig.write_html(OUTPUT_HTML)
    print(f"La carte 3D interactive a été générée : {OUTPUT_HTML}")

if __name__ == "__main__":
    visualize()
