# Customer Review Sentiment Analysis System

An NLP-powered sentiment analysis project that classifies women’s clothing e-commerce reviews into **positive**, **neutral**, and **negative** sentiment using review text and rating-based label mapping. The project also includes interactive analytics and a Streamlit app for exploring customer feedback patterns across departments, products, and customer segments.

\---

## Overview

Understanding customer sentiment at scale helps businesses identify what customers like, where dissatisfaction occurs, and which product areas may need attention.

This project builds an end-to-end customer review analysis workflow that:

* cleans and prepares raw e-commerce review data
* combines review titles and review text for richer NLP input
* maps star ratings into 3-class sentiment labels
* trains and compares multiple machine learning text classifiers
* evaluates model performance using macro F1, precision, recall, and confusion matrices
* analyses low-confidence and misclassified reviews
* surfaces actionable business insights through an interactive Streamlit dashboard

The dataset used is the **Women’s Clothing E-Commerce Reviews** dataset.

\---

## Problem Statement

Customer reviews contain valuable signals about satisfaction, quality, fit, and product experience, but manually reviewing thousands of comments is slow and inconsistent.

The goal of this project is to develop a sentiment analysis system that can automatically classify customer reviews and support insight generation from review data.

\---

## Objectives

* build a reliable 3-class sentiment classifier
* compare multiple baseline NLP models
* identify which reviews are high-confidence or uncertain
* uncover patterns in sentiment by department, class, age group, recommendation status, and feedback count
* create a deployable review intelligence app using Streamlit

\---

## Dataset

The dataset contains women’s clothing e-commerce reviews and related metadata, including:

* `Title`
* `Review Text`
* `Rating`
* `Recommended IND`
* `Positive Feedback Count`
* `Age`
* `Division Name`
* `Department Name`
* `Class Name`

### Sentiment label mapping

The original review ratings are converted into sentiment classes as follows:

* **1–2** → Negative
* **3** → Neutral
* **4–5** → Positive

\---

## Project Structure

```bash
customer-review-sentiment-analysis-system/
│
├── data/
│   ├── raw/
│   │   └── Womens Clothing E-Commerce Reviews.csv
│   └── processed/
│       ├── cleaned\_reviews.csv
│       └── reviews\_with\_predictions.csv
│
├── notebooks/
│   ├── 01\_data\_cleaning\_eda.ipynb
│   ├── 02\_model\_training.ipynb
│   └── 03\_error\_analysis\_and\_insights.ipynb
│
├── models/
│   ├── best\_sentiment\_pipeline.joblib
│   └── model\_metadata.json
│
├── outputs/
│   ├── figures/
│   └── reports/
│
├── app.py
├── requirements.txt
└── README.md
```

\---

## Workflow

### 1\. Data Cleaning and Preprocessing

* renamed and standardised dataset columns
* removed missing and blank review text
* filled missing titles where needed
* combined `Title` and `Review Text` into one text field
* applied light text cleaning
* removed duplicate reviews
* created sentiment labels from ratings

### 2\. Exploratory Data Analysis

* sentiment class distribution
* rating distribution
* review length analysis
* sentiment by department
* recommendation indicator vs sentiment
* age-group and feedback-based trends

### 3\. Text Vectorisation and Model Training

Reviews were transformed into machine-learning-ready features using **TF-IDF**.

The following models were trained and compared:

* Logistic Regression
* Multinomial Naive Bayes
* Linear SVM

### 4\. Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* Macro F1-score
* Confusion Matrix
* Classification Report

### 5\. Error Analysis and Insights

* full-dataset prediction scoring
* confidence estimation
* low-confidence review analysis
* misclassification inspection
* business insight summaries across customer and product dimensions

### 6\. Deployment

A **Streamlit app** was built to support:

* single review prediction
* batch CSV sentiment analysis
* interactive dashboard filtering and review insight exploration

\---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

\---

## Model Inputs and Target

### Input Features

* combined customer review text (`Title + Review Text`)

### Target

* `sentiment`

\---

## Key Features

* 3-class sentiment classification
* TF-IDF text vectorisation
* model comparison across multiple classifiers
* confidence-aware review scoring
* dashboard filters by department, class, age, and sentiment
* batch CSV scoring for new review datasets
* business-facing insight generation from customer feedback

\---

## Streamlit App Features

The deployed app includes three main sections:

### Single Review Prediction

* enter a review manually
* get predicted sentiment
* view confidence score and model output

### Batch Review Analysis

* upload a CSV of reviews
* generate sentiment predictions in bulk
* download scored results

### Review Intelligence Dashboard

* explore sentiment distribution
* filter by department, class, and age group
* inspect recommendation trends
* view data-driven insights and model-driven insights

\---

## How to Run the Project

### 1\. Clone the repository

```bash
git clone https://github.com/owendiche-sys/customer-review-sentiment-analysis-system.git
cd customer-review-sentiment-analysis-system
```

### 2\. Install dependencies

```bash
pip install -r requirements.txt
```

### 3\. Run the notebooks

Run the notebooks in this order:

1. `01\_data\_cleaning\_eda.ipynb`
2. `02\_model\_training.ipynb`
3. `03\_error\_analysis\_and\_insights.ipynb`

### 4\. Run the Streamlit app

```bash
streamlit run app.py
```

\---

## Expected Outputs

Running the notebooks will generate:

* `data/processed/cleaned\_reviews.csv`
* `models/best\_sentiment\_pipeline.joblib`
* `models/model\_metadata.json`
* `data/processed/reviews\_with\_predictions.csv`

\---

## Why This Project Matters

This project goes beyond basic sentiment classification by combining NLP, model evaluation, error analysis, and business intelligence in one workflow.

It demonstrates the ability to:

* build end-to-end NLP pipelines
* prepare messy real-world review data
* evaluate classification systems responsibly
* translate model outputs into useful business insights
* deploy machine learning workflows in an interactive application

\---

## Future Improvements

* add n-gram optimisation and hyperparameter tuning
* test transformer-based sentiment models
* add topic extraction for common complaints and praise themes
* support live API-based prediction
* expand dashboard visualisations and export options

\---

## Author

**Owen Nda Diche**  


