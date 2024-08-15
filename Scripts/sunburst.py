import  pandas as pd

df =pd.read_csv("sunburst-chart-data.csv")

import numpy as np

import  plotly.express as px

colors = [
    'rgb(0, 0, 0)',   # Black
    'rgb(64, 64, 64)',   # Dark gray
    'rgb(128, 128, 128)',   # Medium gray
    'rgb(192, 192, 192)',   # Light gray
    'rgb(224, 224, 224)',   # Very light gray
    'rgb(255, 255, 255)'   # White
]

fig =px.sunburst(data_frame=df, path =["category", "name", "subtopic_name"], values="question_share",
                 color_discrete_sequence=colors)

fig.update_traces(textinfo="label+value")

fig.show()