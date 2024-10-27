# !pip install plotly

import pandas as pd
import plotly.express as px

# Read the data from the CSV file
df = pd.read_csv('/kaggle/input/emotions/emotions_user1.csv')  # Replace with your actual file name

# Melt the DataFrame to long format for easier plotting
df_melted = df.melt(id_vars='Week', value_vars=['Anxiety', 'Depression', 'Eating Disorder'],
                    var_name='Condition', value_name='Score')

# Create an interactive line plot
fig = px.line(df_melted, x='Week', y='Score', color='Condition', markers=True,
              title='Mental Health Trends Over Weeks',
              labels={'Score': 'Score (1-10)', 'Week': 'Week'},
              template='plotly_white')

# Update layout for better aesthetics
fig.update_layout(
    title_font=dict(size=20, color='black', family='Arial, sans-serif'),
    xaxis_title_font=dict(size=16, family='Arial, sans-serif'),
    yaxis_title_font=dict(size=16, family='Arial, sans-serif'),
    legend_title_font=dict(size=14, family='Arial, sans-serif'),
    yaxis=dict(range=[0, 10]),
)

# Show the plot
fig.show()