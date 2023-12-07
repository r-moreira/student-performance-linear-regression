import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Exploratory Analysis', layout="wide")

df = pd.read_csv('./datasets/Student_Performance.csv')

st.title('Exploratory Analysis')
st.write('## Descriptive statistics of numeric values')
st.table(df.describe())

st.write('## Descriptive statistics of categoric values')
st.table(df['Extracurricular Activities'].describe())
st.write(f"Extracurricular Activities unique values: {df['Extracurricular Activities'].unique()}")

# Matplotlib Dark Mode :D
plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "white",
    "axes.facecolor": "black",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})

def plot_count_plot(dataframe, column_name):
    st.write(f"## {column_name} Analytics")
    
    plt.figure(figsize=(15, 6))
    graph = sns.countplot(x=column_name, data=dataframe, hue=None, order=dataframe[column_name].value_counts().index)
    
    for container in graph.containers:
        graph.bar_label(container)
    
    plt.tight_layout()
    
    st.pyplot(plt)

st.write("## Analyzing distribution and outliers in numerical features")
numeric_columns = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced', 'Performance Index']
plt.figure(figsize=(12, 8))
for column in numeric_columns:
    plt.subplot(2, 3, numeric_columns.index(column) + 1)
    sns.boxplot(data=df, y=column)
    plt.title(f'Box Plot de {column}')
plt.tight_layout()
st.pyplot(plt)

plot_count_plot(df, "Hours Studied")
plot_count_plot(df, "Sleep Hours")
plot_count_plot(df, "Sample Question Papers Practiced")

st.write("## Analyzing Hours Studied x Performance Index")
avg_performance_by_hours = df.groupby('Hours Studied')['Performance Index'].mean()
data_for_line_chart = pd.DataFrame({'Hours Studied': avg_performance_by_hours.index, 'Average Performance Index': avg_performance_by_hours.values})
st.line_chart(data_for_line_chart.set_index('Hours Studied'))


st.write("## Correlation Map")
plt.figure(figsize = (10,6))
sns.heatmap(df.select_dtypes(exclude = object).corr(), annot = True, fmt = ".2f", linewidths = 0.2)
st.pyplot(plt)

st.write("### About quality improvements")
st.write(">This dataset is an atypical case where no improvement in data quality was necessary")
