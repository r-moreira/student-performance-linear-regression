import streamlit as st
from st_aggrid import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title='Linear Regression', layout="wide")

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

df = pd.read_csv('./datasets/Student_Performance.csv')

st.title("Linear Regression")

encoder = LabelEncoder()
df["Extracurricular Activities"] =  encoder.fit_transform(df["Extracurricular Activities"])

train = df.drop(columns = "Performance Index")
target = df["Performance Index"]

x_train, x_test, y_train, y_test = train_test_split(train, target, test_size = 0.2, random_state = 7)

model = LinearRegression()
model.fit(x_train,y_train)

st.write("## Predictions")
predictions = np.round(model.predict(x_test), decimals = 1)
df_result = pd.DataFrame({"Actual Performance" : y_test, "Predicted Performance" : predictions})
AgGrid(df_result, fit_columns_on_grid_load=True, height=300)

st.write("## Linear Regression Distribution")
plt.scatter(y_test, predictions)
st.pyplot(plt)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
score = model.score(x_train, y_train)

st.write("## Results")
st.write(f'MSE: {mse:.2f}')
st.write(f'RMSE: {rmse:.2f}')
st.write(f'MAE: {mae:.2f}')
st.write(f"Score: {score:.4f}")

st.write("""
**a) MSE (Mean Squared Error)** : 4.21

MSE measures the average squared errors between the model predictions and the actual values. The lower the MSE, the better. A MSE of 4.21 suggests that, on average, the squares of the errors between predictions and actual values are relatively small.

**b) RMSE (Root Mean Squared Error)** : 2.05

RMSE is the square root of MSE, making it more interpretable on the same scale as the target variable. An RMSE of 2.05 indicates that, on average, the model makes errors of approximately 2.05 units on the performance index scale. Again, lower values are desirable.

**c) MAE (Mean Absolute Error)** : 1.63

MAE measures the average of the absolute values of errors between predictions and actual values. A MAE of 1.63 suggests that, on average, the model makes absolute errors of about 1.63 units on the performance index scale. Similar to the previous metrics, lower MAE is better.

**d) Score (Coefficient of determination - R-squared)** : 0.98

Score refers to accuracy for a classification model, it ranges from 0 to 1 as well, a score of 0.98 would suggest that the model correctly predicts the target variable for 98% of the instances.

> In summary, based on these metrics, your model seems to have reasonably good performance as the metrics indicate relatively low average errors.
""")
