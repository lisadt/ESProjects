## **Project 1: Bikeshare Demand Prediction** ##
- Worked on this project as part of my assigments while attending the Data Science course at General Assembly. Data source: https://www.kaggle.com/c/bike-sharing-demandwas. 
- The data, collected hourly over a period of two years, gives information regarding the number of rented bikes, weather conditions, day of the week and seasons. 
- The scope was to build a robust model to predict the demand of bike rentals. Both GBR and XGB regressors were used and results compared. 

The following table shows the Root Mean Squared Logarithmic Error for the three methods used: 

![Bikesharing_scores_table](https://user-images.githubusercontent.com/68543397/96701656-c0705780-1388-11eb-90a2-d6fe9b542f6f.jpg)

**Comments**:\
Adding a column with daly time shift gave much better scores, \
Using ln(count) in the GBRegressor improved the accuracy of the predicive model,\
Next: try XGBoost with ln(count)

## Project 2: Healthcare Analytics II ##
- Dataset from Kaggle (https://www.kaggle.com/nehaprabhavalkar/av-healthcare-analytics-ii) collects data regarding length of stays of patients in different hospitals and departments.
- Being able to predict the length of stay has grewidth=at importance in the optimisation of bed allocations, treatment plans, as well as minimisations of infection spreading.
- Method: XGBClassifier.
**Model Performance:** \
Confusion Matrix and Class Statistics
![Healthcare Analytics II - ConfusionMatrix](https://github.com/lisadt/ESProjects/blob/main/Healthcare/ConfusionMatrix_HealthcareAnalyticsII.jpg width="100" HEIGHT="100" )

**Comments**: 
**To do**:
- Try different models besides XGBClassifier. 
- Improve feature engineering: look at other relatioships between fields that might be relevant to the problem. 
- Look at way to deal with imbalanced data. 
- Check the null values and different ways to replace them.
