## **Project 1: Bikeshare Demand Prediction** ##
- Worked on this project as part of my assigments while attending the Data Science course at General Assembly. Data source: https://www.kaggle.com/c/bike-sharing-demand. 
- The data, collected hourly over a period of two years, gives information regarding the number of rented bikes, weather conditions, day of the week and seasons. 
- The scope was to build a robust model to predict the demand of bike rentals. Both GBR and XGB regressors were used and results compared. 

The following table shows the Root Mean Squared Logarithmic Error for the three methods used: 

![Bikesharing_scores_table](https://github.com/lisadt/ESProjects/blob/main/Bikeshare/Bikesharing_scores_table.jpg)

**Comments**:\
Adding a column with daly time shift gave much better scores, \
Using ln(count) in the GBRegressor improved the accuracy of the predicive model,\
Next: try XGBoost with ln(count)

## Project 2: Healthcare Analytics II ##
- Dataset from Kaggle (https://www.kaggle.com/nehaprabhavalkar/av-healthcare-analytics-ii) collects data regarding length of stay of patients in different hospitals and departments.
- Being able to predict the length of stay has great importance in the optimisation of bed allocation, treatment plans, as well as the minimisation of infection spreading.
- Method: XGBClassifier.
**Model Performance:** \
Confusion Matrix and Class Statistics
<img src="https://github.com/lisadt/ESProjects/blob/main/Healthcare/ConfusionMatrix_HealthcareAnalyticsII.jpg" width="450" />
<img src="https://github.com/lisadt/ESProjects/blob/main/Healthcare/HealthcareAnalyticsReport.jpg" width="400" />

**Comments**: \
**To do**:
- Try different models besides XGBClassifier. 
- Improve feature engineering: look at other relatioships between fields that might be relevant to the problem. 
- Look at way to deal with imbalanced data. 
- Check the null values and different ways to replace them.

## **Project 3: Online Retail** ## 
Dataset from Kaggle: https://www.kaggle.com/vijayuv/onlineretail?select=OnlineRetail.csv \
Data: Collection of online retail purchases made over a period of 12 months (+ 9 days). Fields: invoice number, stock code and description, quantity, invoice date and time of transaction, unit price and customer ID. \
The goal was to perform Customers Segmentation and predict Customer Lifetime Value : \
The workflow consisted of: 
- Data cleaning, analysis, and visualization.
- Implementation of Customer Segmentation to better understand customer behavior. Customers were grouped into five tiers ("Lost", "At Risk", "Potential", "Frequent", "Best") using the Recency, Frequency and Monetary method (RFM). Segmentation is an important step in marketing for the development of strategies to improve loyalty and lifetime value. 
The RFM groups were created using KMeans Clustering.
  ### * Workflow with lifetimes library models/utilities:* 
- Recency, Frequency, and Monetary Value from transactions data;
- BetaGeoFitter Model to evaluate the probability of the customer being "Alive";
- Expected future number of transactions;
- Expected average sale per customer using the Gamma-Gamma model;
- Prediction of customer's lifetime value;
- Tested the model by dividing the dataset into calibration and holdout. 

<img src="https://github.com/lisadt/ESProjects/blob/main/OnlineRetail/ConfusionMatrix2.jpg" width="300" />
<img src="https://github.com/lisadt/ESProjects/blob/main/OnlineRetail/ClassificationReport2.jpg" width="400" />


## **Applications Folder** ## 
Data source for the Movie Rating :  https://ai.stanford.edu/~amaas/data/sentiment/.
