import pandas as pd
import plotly.express as px
import os

# 1. Standard Tasks
DATE_OVERRIDES = {
    "Data Processing (15min)":  {"start": "2026-01-04", "end": "2026-01-09"}, 
    "Preprocessing data (1s)":  {"start": "2026-01-14", "end": "2026-01-17"},
    "Model":                    {"start": "2026-01-04", "end": "2026-01-25"},
    "Backend":                  {"start": "2026-01-12", "end": "2026-01-25"}, 
    "Frontend":                 {"start": "2026-01-04", "end": "2026-01-17"},
    "Integration":              {"start": "2026-01-21", "end": "2026-01-27"},
    "Documentation":            {"start": "2026-01-25", "end": "2026-01-27"},
}

# 2. Define the individual Fridays for Paperwork
paperwork_fridays = ["2026-01-09", "2026-01-16", "2026-01-23"]
presentations = ["2026-01-14", "2026-01-27"]

# 3. Define the EXACT order (Top to Bottom)
# Technical tasks first, then Midterm, then Admin/Docs at the end
TASK_ORDER = [
    "Data Processing (15min)",
    "Preprocessing data (1s)",
    "Model",
    "Backend",
    "Frontend",
    "Integration",
    "Presentations",
    "Documentation",
    "Weekly Paperwork"
]

tasks = []

# Add standard tasks
for name, dates in DATE_OVERRIDES.items():
    tasks.append({
        "Task Name": name,
        "Start Date": pd.to_datetime(dates["start"]),
        "End Date": pd.to_datetime(dates["end"])
    })

# Add separate blocks for Weekly Paperwork
for friday in paperwork_fridays:
    start_dt = pd.to_datetime(friday)
    tasks.append({
        "Task Name": "Weekly Paperwork",
        "Start Date": start_dt,
        "End Date": start_dt + pd.Timedelta(hours=18) 
    })

for presentation_day in presentations:
    pres_dt = pd.to_datetime(presentation_day)
    tasks.append({
        "Task Name": "Presentations",
        "Start Date": pres_dt,
        "End Date": pres_dt + pd.Timedelta(hours=18) 
    })

df = pd.DataFrame(tasks)

# Visibility fix for tasks that start and end on the same day
mask = df['Start Date'] == df['End Date']
df.loc[mask, 'End Date'] += pd.Timedelta(hours=12)

# 4. Create the Chart
fig = px.timeline(
    df,
    x_start="Start Date",
    x_end="End Date",
    y="Task Name",
    color="Task Name",
    template="plotly_white",
    title="Project Gantt Chart"
)

# Apply the custom order
# 'reversed' makes the first item in TASK_ORDER appear at the top
fig.update_yaxes(
    categoryorder="array", 
    categoryarray=TASK_ORDER,
    autorange="reversed"
)

fig.update_layout(
    xaxis_title="January 2026",
    yaxis_title="Tasks",
    height=600,
    xaxis=dict(
        tickformat="%d %b",
        dtick="D1" 
    ),
    showlegend=False # Legend is often redundant if names are on the Y axis
)

# 5. Save and Show
fig.show()

script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
output_path = os.path.join(script_dir, "gantt_ordered.html")
fig.write_html(output_path)
print(f"File created: {output_path}")