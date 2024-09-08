
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


# Load the data set using pd.read_csv in a variable called data
data = pd.read_csv(r"C:\Users\azhar\Desktop\Data Sets\student_marks\Student_Marks.csv")

# Check if there are any missing value in the dataset
null_data=data.isnull().sum()
print(null_data)

# Check any duplicate values in the dataset
duplicate_data=data.duplicated().sum()
print(duplicate_data)

# Check the relationship between the variables
grouped = data.groupby('number_courses')['Marks'].value_counts()

# Visualize the relationship between the variables
fig = px.scatter(data,
                 x='number_courses',
                 y='Marks',
                 size = 'time_study',
                trendline  = 'ols')
fig.show()

fig = px.scatter(data,
                y = 'number_courses',
                x = 'time_study',
                size  = 'Marks',
                trendline  = 'ols')
fig.show()

# Check the correlation between the variables
correlation  = data.corr()
correlation['Marks'].sort_values(ascending = False)

# Split the data set into independent and dependent variables
x = data[["time_study", "number_courses"]]
y = data["Marks"]
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.2, random_state = 42)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Test the model
score = model.score(x_test, y_test)
# Print the score
print(score)
# Print the coefficient and the intercept
print(model.coef_)
print(model.intercept_)

# Predict the model
print(model.predict([[5, 2]]))
# Save the model
joblib.dump(model, 'Marks_model.joblib')
