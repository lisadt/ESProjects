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
- Data set from Kaggle (https://www.kaggle.com/nehaprabhavalkar/av-healthcare-analytics-ii) collects data regarding length of stays of patients in different hospitals and departments.
- Being able to predict the length of stay has grewidth=at importance in the optimisation of bed allocations, treatment plans, as well as minimisations of infection spreading.
- Method: XGBClassifier.
**Model Performance:** \
Confusion Matrix and Class Statistics
<img src="https://github.com/lisadt/ESProjects/blob/main/Healthcare/ConfusionMatrix_HealthcareAnalyticsII.jpg" width="500" />
<img src="https://github.com/lisadt/ESProjects/blob/main/Healthcare/HealthcareAnalyticsReport.jpg" width="500" />

**Comments**: \
**To do**:
- Try different models besides XGBClassifier. 
- Improve feature engineering: look at other relatioships between fields that might be relevant to the problem. 
- Look at way to deal with imbalanced data. 
- Check the null values and different ways to replace them.

## Project 3: Online Retail ##
Data set from Kaggle: https://www.kaggle.com/vijayuv/onlineretail?select=OnlineRetail.csv .
Data: Collection of online retail purchases made over a period of 12 months (+ 9 days). Fields: invoice number, stock code and description, quantity, invoice date and time of transaction, unit price and customer ID.
The goal was to build a machine learning model to predict Customer Lifetime Value :
The workflow consisted of:
•	Data cleaning, analysis, and visualization
•	Implementation of Customer Segmentation to better understand customer behavior. Customers were grouped into three tiers (high, medium and low value) using the Recency, Frequency and Monetary method (RFM). Segmentation is an important step in marketing as it helps the development of strategies to improve loyalty and lifetime value. The RFM Groups were created using KMeans Clustering
•	A 3-month period dataset was selected to define RFM scores and the following 7-month period to calculate the customer LTV. The 3-month set was then used to train the model and predict the LTV .
**Results**:
*Classification Report*: while the model seems to work well for the cluster “0” (precision 0.91 and recall 0.95), improvements are need for the other two classes where the recall values are 0.50, 0.60.
Ways to improve the model:
•	Try different Classification models
•	Apply feature engineering to add fields to the data 
•	Change the time interval used to create the training set
•	Parameter tuning on the model
