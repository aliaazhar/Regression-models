import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load the dataset
data2 = pd.read_csv(r"/co2_consumption/datasets/Energy Data1.csv")

# Rename columns for consistency and readability
renamed_data2 = data2.rename(columns={
    'Energy Related CO2missions (Gigatonnes)': 'co2_emissions',
    'Oil Production (Million barrels per day)': 'Oil',
    'Natural Gas Production (Billion Cubic Metres)': 'natural_gas',
    'Coal Production (million tons)': 'coal',
    'Electricity Generation (Terawatt-hours)': 'electricity',
    'Hydroelectricity consumption in TWh': 'hydro',
    'Nuclear energy consumption in TWh': 'nuclear',
    'Installed Solar Capacity (GW)': 'solar',
    'Installed Wind Capacity in GW': 'wind'
})


train_data = (renamed_data2)

train_data.fillna(0, inplace = True)
train_data.drop(index=31, inplace = True)
train_data.drop(['electricity', 'solar','wind'], axis=1, inplace = True)

# draw boxplot of train_data
for column in train_data.columns:
    plt.boxplot(train_data[column])
    plt.title(f'boxplot of {column}')
    # plt.show()

# draw scatter plot of train_data
for col in train_data:
    if col != 'co2_emissions':
        fig = px.scatter(train_data,
                        x = f'{col}',
                        y = 'co2_emissions',
                        title = f'scatter between {col} & co2_emissions',
                        trendline = 'ols',
                        size = col
                        )
        # fig.show()
    else:
        pass


# Split the Dataset into Training and Testing Sets
X = train_data[['Year','coal','Oil','natural_gas',]]
Y = train_data['co2_emissions']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and Train the Linear Regression Model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

print(f'Linear Regression Coefficients: {linear_reg.coef_}')
print(f'Linear Regression Intercept: {linear_reg.intercept_}')

# Make Predictions and Evaluate the Model
linear_reg_pred = linear_reg.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, linear_reg_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Print the R2 Score to check the accuracy
linear_reg_r2_score = r2_score(y_test, linear_reg_pred)
accuracy_percentage = linear_reg_r2_score * 100
print(f"Accurecy: {accuracy_percentage: .2f} %")

# check residual error
residual = y_test - linear_reg_pred
print(f'Residuals are : {residual}')





# Predict the model using another dataset

test_data = pd.read_csv(r"/co2_consumption/datasets/Percentage of Energy Consumption Global.csv")

test_data = test_data.rename(columns={'Coal (TWh; direct energy)': 'coal',
                               'Oil (TWh; direct energy)': 'Oil',
                               'Gas (TWh; direct energy)': 'natural_gas',
                               'Hydropower (TWh; direct energy)': 'hydro',
                               'Nuclear (TWh; direct energy)': 'nuclear',
                               'Solar (TWh; direct energy)': 'solar',
                               'Other renewables (TWh; direct energy)': 'renewables',
                               'Traditional biomass (TWh; direct energy)': 'biomass',
                              'Wind (TWh; direct energy)' : 'wind',
                              'Biofuels (TWh; direct energy)' : 'biofuels'})

# test_data.drop(['electricity', 'solar','wind'], axis=1, inplace = True)
test_data = test_data[['Year','coal','Oil','natural_gas']]
pred = linear_reg.predict(test_data)
print(pred)

