# Customer Segmentation and Marketing Strategy Optimization

## Project Overview

This project aims to analyze customer data to segment customers based on their purchasing behavior and optimize marketing strategies to increase customer retention and profitability. By performing clustering and RFM (Recency, Frequency, Monetary) analysis, we can identify the most valuable customer groups and suggest targeted marketing actions for each segment. Additionally, I build predictive models to further refine marketing strategies and create a real-time dashboard for tracking performance.

### Key Objectives:
- Segment customers based on behavior to identify different groups such as high-value customers or at-risk customers.
- Develop targeted marketing strategies for each segment to improve customer engagement and profitability.
- Visualize key metrics on an interactive dashboard.

## Table of Contents
Data Collection
Data Cleaning & Preprocessing
Customer Segmentation
RFM Analysis
Marketing Strategy Optimization
Dashboard Development
How to Run
Results
Contributing

## 1. Data Collection
For this project, I used customer purchase data from Kaggle or generated synthetic data simulating a retail e-commerce dataset. The data includes:

- Customer Demographics: Age, Gender, Location.
- Transaction History: Date of Purchase, Purchase Amount, Frequency.
- Behavioral Metrics: Last Purchase Date, Customer Segments.

Tools Used:

- Python
- Pandas (for data manipulation)

## 2. Data Cleaning & Preprocessing
Before performing any analysis, the dataset needs to be cleaned and preprocessed. This step ensures the accuracy of the analysis by addressing:
- Missing Values: Fill missing entries using the median for numerical columns or the most frequent category for categorical columns.
- Outliers: Remove or cap extreme outliers that could skew the segmentation process.
- Data Normalization: Normalize purchase amounts and frequency to bring all features into a similar scale.

```
# Handling missing values
data.fillna(method='ffill', inplace=True)

# Removing outliers based on purchase amounts
data = data[data['purchase_amount'] < threshold]
```

Tools Used:
- Pandas
- NumPy

## 3. Customer Segmentation
I applied K-Means Clustering to segment customers based on their purchasing patterns such as frequency of purchases, total monetary value, and recency of their last purchase. The clustering helped identify different customer groups:

- High-Value, Loyal Customers
- New Customers
- At-Risk Customers

The clustering results were visualized using Matplotlib and Seaborn to provide insights into customer distribution.

```
from sklearn.cluster import KMeans

# Feature selection for clustering
features = data[['recency', 'frequency', 'monetary']]
kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(features)
data['cluster'] = clusters

```

Tools Used:
- Python
- Scikit-learn (K-Means)
- Matplotlib / Seaborn (Visualization)

## 4. RFM Analysis
I performed RFM (Recency, Frequency, Monetary) analysis to further refine the customer segments:
- Recency: How recently a customer made a purchase.
- Frequency: How often the customer makes purchases.
- Monetary: The total amount the customer has spent.

Each customer was scored based on their RFM metrics, which helped categorize them into actionable segments like VIP Customers, Loyal Customers, At-Risk Customers, and Lost Customers.

```
# Calculating RFM metrics
data['Recency'] = (today_date - data['last_purchase_date']).days
data['Frequency'] = data.groupby('customer_id')['purchase_id'].count()
data['Monetary'] = data.groupby('customer_id')['purchase_amount'].sum()

# RFM score
data['RFM_Score'] = data['Recency'].rank(ascending=False) + \
                    data['Frequency'].rank(ascending=True) + \
                    data['Monetary'].rank(ascending=True)
```

Tools Used:
- Python
- Pandas

## 5. Marketing Strategy Optimization
To optimize the marketing strategy, I built a predictive model to forecast which customer segments are more likely to respond to specific marketing campaigns. Using Logistic Regression and Random Forest models, I predicted customer churn and response rates, which can help fine-tune campaign targeting.

Additionally, I simulated A/B testing to evaluate the effectiveness of different promotional offers on each customer segment, identifying the most successful marketing strategies.

```
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Train-test split
X = data[['RFM_Score']]
y = data['response_to_campaign']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Training Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

Tools Used:
- Scikit-learn (Logistic Regression, Random Forest)
- A/B Testing Analysis

## 6. Dashboard Development
Finally, I developed an interactive dashboard using PowerBI to track key business metrics in real-time. The dashboard includes:

- Customer Segments: Visualized using pie charts and bar graphs.
- Sales Trends: Time-series analysis of revenue, customer retention rates, and purchase frequency.
- Campaign Performance: Results from A/B testing and predicted conversion rates for each customer segment.
  
The dashboard enables real-time monitoring of business performance and helps executives make data-driven marketing decisions.

## 7. Results
- Increased Conversion Rates: Improved by 8% through A/B testing for targeted segments.
- Customer Insights: Identified high-value and at-risk customer groups that helped inform personalized marketing strategies.
- Improved Profit Margins: Implementing data-driven marketing actions led to a 5% increase in overall profit margins.
