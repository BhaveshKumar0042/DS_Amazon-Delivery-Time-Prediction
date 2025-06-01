# Amazon Delivery Time Prediction

This project aims to predict the delivery time for orders on an e-commerce platform, simulating Amazon's operations. The prediction is based on various features related to the order, vendor, customer, and location. A Random Forest Regressor model was found to provide the best performance after evaluating several regression algorithms.

## Table of Contents
1.  [Overview](#overview)
2.  [Dataset](#dataset)
3.  [Project Workflow](#project-workflow)
4.  [Exploratory Data Analysis (EDA) Highlights](#eda-highlights)
5.  [Preprocessing and Feature Engineering](#preprocessing-and-feature-engineering)
6.  [Modeling and Evaluation](#modeling-and-evaluation)
7.  [Results](#results)
8.  [Setup and Installation](#setup-and-installation)
9.  [How to Run](#how-to-run)
10. [Files in this Repository](#files-in-this-repository)
11. [Future Work](#future-work)

## Overview
Predicting delivery times accurately is crucial for customer satisfaction and logistics management in e-commerce. This project tackles this challenge by analyzing a dataset of Amazon-like delivery orders. Various features such as order details, vendor information, customer demographics, and location data are used to train machine learning models. After comparing several regression models, the Random Forest Regressor, tuned with RandomizedSearchCV, demonstrated the highest accuracy in predicting delivery times.

## Dataset
* **Source File:** `amazon_delivery.csv`
* **Description:** The dataset contains simulated Amazon delivery order records. Key features include:
    * `order_id`: Unique identifier for each order.
    * `vendor_id`: Unique identifier for each vendor.
    * `item_id`: Unique identifier for each item/product.
    * `location_id`: Identifier for the delivery location.
    * `quantity`: Number of items ordered.
    * `order_date`: Date and time the order was placed.
    * `delivery_date`: Date and time the order was delivered.
    * `delivery_charges`: Charges for the delivery.
    * `coupon_discount`: Discount applied through coupons.
    * `address_type`: Type of address (e.g., Home, Work).
    * `location_type`: Type of location.
    * `payment_type`: Method of payment.
    * `vendor_category`: Category of the vendor.
    * `cust_id`: Unique identifier for each customer.
    * `gender`: Gender of the customer.
    * `device_type`: Device used to place the order.
* **Target Variable:** `delivery_time` (engineered as the difference between `delivery_date` and `order_date` in days).

## Project Workflow
1.  **Data Collection & Loading:** The `amazon_delivery.csv` dataset is loaded into a pandas DataFrame.
2.  **Data Preprocessing:**
    * **Null Value Handling:** Missing values in `delivery_charges` are filled with the median, while `coupon_discount` and `address_type` are filled using the forward fill method.
    * **Date-Time Conversion:** `order_date` and `delivery_date` columns are converted to datetime objects.
    * **Feature Engineering:**
        * The primary target variable, `delivery_time` (in days), is calculated.
        * Date and time components (day, month, year, hour, day of the week) are extracted from both `order_date` and `delivery_date`.
3.  **Exploratory Data Analysis (EDA):**
    * Basic data inspection: `.head()`, `.info()`, `.describe()`, `.isnull().sum()`.
    * Visualizations: Histograms for numerical features, count plots for categorical features, box plots to identify outliers, and a correlation heatmap to understand feature relationships.
4.  **Feature Scaling & Encoding:**
    * Irrelevant columns (`order_id`, original date columns) are dropped.
    * Categorical features (`vendor_id`, `item_id`, `location_type`, `address_type`, `payment_type`, `vendor_category`, `cust_id`, `gender`, `device_type`) are converted to numerical format using Label Encoding or One-Hot Encoding (the notebook uses a mix, with `pd.get_dummies` for some like `location_id`).
    * Numerical features are scaled using `StandardScaler` after splitting the data to prevent data leakage.
5.  **Model Building & Training:**
    * The data is split into training and testing sets.
    * Several regression models are trained and evaluated:
        * Linear Regression
        * Random Forest Regressor (with hyperparameter tuning using `RandomizedSearchCV`)
        * Gradient Boosting Regressor
6.  **Performance Evaluation:** Models are evaluated based on Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² (Coefficient of Determination) score.

## EDA Highlights
* Delivery times primarily range between 0 to 7 days, with a few outliers.
* Certain `vendor_category` types and `location_id`s show correlations with delivery times.
* No strong linear correlations were observed between most numerical features and the delivery time from the initial heatmap, suggesting non-linear models might perform better.

## Preprocessing and Feature Engineering
* **Delivery Time Calculation:** `delivery_df['delivery_time'] = (delivery_df['delivery_date'] - delivery_df['order_date']).dt.days`
* **Date-Time Components Extraction:** Features like `order_day`, `order_month`, `order_year`, `order_hour`, `order_dayofweek`, and similar for `delivery_date` were created.
* **Categorical Encoding:**
    * Label Encoding was applied to: `vendor_id`, `item_id`, `location_type`, `address_type`, `payment_type`.
    * One-Hot Encoding (using `pd.get_dummies`) was applied to: `location_id`, `vendor_category`, `cust_id`, `gender`, `device_type`.
* **Numerical Scaling:** `StandardScaler` was applied to numerical features before model training.

## Modeling and Evaluation
Multiple regression models were implemented and their performance compared. The Random Forest Regressor underwent hyperparameter tuning using `RandomizedSearchCV` with parameters like `n_estimators`, `max_features`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.

## Results
The Random Forest Regressor emerged as the best-performing model:
* **Mean Absolute Error (MAE):** 0.17 days
* **Mean Squared Error (MSE):** 0.06
* **Root Mean Squared Error (RMSE):** 0.25 days
* **R² Score:** 0.95

This indicates that the Random Forest model can predict the delivery time with high accuracy, explaining 95% of the variance in the data.

## Setup and Installation
1.  **Prerequisites:**
    * Python 3.x
    * Jupyter Notebook or an IDE that supports `.ipynb` files (like VS Code, Google Colab).
2.  **Libraries:**
    You can install the necessary libraries using pip. It's recommended to use a virtual environment.
    ```bash
    pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost
    ```
    Alternatively, create a `requirements.txt` file with the libraries and versions used in the notebook and install using:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run
1.  **Clone the repository (if applicable) or download the notebook and CSV file.**
2.  **Ensure all dependencies are installed** (see Setup and Installation).
3.  **Place the `amazon_delivery.csv` file** in the same directory as the notebook, or update the file path in the notebook accordingly.
4.  **Open and run the `Bhavesh_Project_3_Amazon.ipynb` notebook cells sequentially.**
    * The notebook will load the data, perform EDA, preprocess features, train models, and evaluate them.

## Files in this Repository
* `Bhavesh_Project_3_Amazon.ipynb`: The Jupyter Notebook containing all the code for analysis, preprocessing, modeling, and evaluation.
* `amazon_delivery.csv`: The dataset used for the project.
* `Amazon Deliver time Prediction.docx`: A document providing an overview and report of the project.
* `README.md`: This file.
* *(Optionally, if a model was saved: `best_model.pkl` or similar)*

## Future Work
* **Advanced Feature Engineering:** Explore interactions between features or more complex time-based features.
* **Deep Learning Models:** Implement and evaluate Neural Network regressors more thoroughly, potentially using TensorFlow/Keras or PyTorch.
* **More Extensive Hyperparameter Tuning:** Use GridSearchCV or more sophisticated Bayesian optimization techniques.
* **Handling Outliers:** Investigate the impact of outliers in delivery times and apply robust scaling or outlier removal techniques if beneficial.
* **Model Deployment:** Save the best model using `pickle` or `joblib` and create a simple API (e.g., using Flask or FastAPI) for real-time predictions.
* **Real-time Data:** Integrate with a mock real-time data stream to simulate a production environment.
