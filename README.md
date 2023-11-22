# car_price_predictions
AI ML Learning Project Assignment to work on preduction models 

I developed several utility function to help in this assignment

1) loading the data
2) scoping the data
3) handling outliers (detect and cleanup)
4) encoding non numerical values
5) initializing dataframe for a specific manufacturer and model
6) searching best correclating car model
7) drawing various diagrams
8) evaluating test set with my selected prediction model which was RandomForestRegressor
9) evaluating test set using feature engineering with my selected prediction model which was RandomForestRegressor
10) Feature engineering function to calculate mse and r2 for RandomForestRegressor and plot corresponding diagram 
11) function to predict prince, one with basic columns year, condition and odotomer, and one with additional feature engoineered columns such as year_odometer_interaction, odometer_sqrt, odometer_condition_interaction etc
12) function to plot model comparison charts (single and multiple)
13) simple scatterplot function to easy the pain rewrite this code over and over again


### Data Preparation

1) Load Data
2) Scope Data based on specific manufacturer, year range, select only needed features, drop values do not make sense (outliers)
3) Encode non numerical data using label encoder
4) Calculate correclation martix and plot heat map of all car models to detect correlation of the features if any across the whole data
5) Search car model start has the best correlation so we can use it as test case
6) Print scatter plots price vs year and price for the specific model with the best correlation
7) Calculate correclation martix and plot heat map of the specific model with the best correlation. W
8) We actually found that there is significant positive correlation with year, significant negative correlation with odometer, and some meaningful positive correlation with condition
9) The next step is modeling (see Modeling section)


When test correlation for all 'Chevrolet' models, year had significant correction and odometer notable negative correlation
- price                 1.000000
- year                  0.486560
- condition             0.028224
- manufacturer_model    0.001751
- odometer             -0.323382
- Name: price, dtype: float64

When searching for the best correlating Chevrolet model, it was
Brand:       Chevrolet
Model:       Silverado 1500
Encoded:     744
price        1.000000
year         0.848118
condition    0.317607
odometer    -0.789283

This is also nicely visible in the heatmap.

I tested several predictional models:
- linear_model = LinearRegression()
- random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
- gb_model = GradientBoostingRegressor(random_state=42)
- ridge_model = Ridge(alpha=1.0)
- lasso_model = Lasso(alpha=0.001)

- Mean Squared Error Linear: 24303473.101552382
- R^2 Score Linear: 0.8061823095944434
- Mean Squared Error Ridge: 24303446.008851625
- R^2 Score Ridge: 0.8061825256559365
- Mean Squared Error Lasso: 24303473.05861214
- R^2 Score Lasso: 0.8061823099368874
- Mean Squared Error Random Forest: 14389334.538277978
- R^2 Score Random Forest: 0.8852465417173728
- Mean Squared Error Gradient Boosting: 18961781.52669878
- R^2 Score Gradient Boosting: 0.8487817487598199

Therefore I selected Random Forest model.

After test evalution and price_prediction testing using difference car details, my feature engineered model, year value seems to work yand odometer have also some impact, conditional not so much.
I shall continue working on feature engineering and precition model test to find even better way to predict have also strong impact on condition, now it has very little effect if any, despite of the fact that I did convert condition labels to numbers from high (new) to low (salvage)
also my ulity functions have some overhead to copy the original data multiple times and I can likely optimze that to make this process much faster.
predictions you can see from adove.

- My customer was asking how much she would get from 2014 Mercedes-Benz, mileage: 50000 and condition: good
- My model says: $23023

```
car_details8 = {  
  'manufacturer': 'mercedes-benz',  
  'carmodel': 'm-class',  
  'year': 2014,  
  'condition': 'good',  
  'odometer': 50000  
}
```

- pred_price = predict_price_feat(random_forest_model, **car_details8)
- printcar(car_details8)
- print(f"random_forest model predicted price: ${pred_price}")
- print("\n")

- manufacturer: mercedes-benz
- carmodel: m-class
- year: 2014
- condition: good
- odometer: 50000
- random_forest model predicted price: $23023.6025

- #reference column from the dataset
- #7315746793,birmingham,24997,2014,mercedes-benz,m-class,,8 cylinders,gas,82553,clean,automatic,4JGDA7DBXEA336979,,,SUV,blue,al
