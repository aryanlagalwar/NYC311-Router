# 311-NYC-Analysis

A machine learning project that classifies NYC 311 complaints and predicts the correct agency to route the request to — helping improve service efficiency and reduce misdirected cases. Built using real 2024 data from NYC Open Data and trained with multi-class classification models.

---

## 📍 Project Overview

311 is New York City's non-emergency request and complaint system. Unlike 911, 311 covers a wide range of issues from noise complaints to housing violations and sanitation problems — each handled by a different agency. The goal of this project is to develop a machine learning model that accurately predicts **which agency** should handle a complaint based on features like location, complaint type, and time.

This is a **multi-class classification problem** using over **3 million complaints filed in 2024**, with agencies like NYPD, HPD, DOT, and DSNY represented.

---

## 🔍 Problem Motivation

Routing errors in 311 can delay issue resolution, waste resources, and reduce citizen satisfaction.  
We aimed to:
- Automatically classify incoming complaints
- Improve routing efficiency
- Reduce agency-level misdirection

Given the large **class imbalance** (e.g., NYPD handles 46% of requests, HPD 21%), traditional accuracy isn't enough. So we use **Macro F1-Score** to ensure fair evaluation across all agencies.

---

## 🧠 Features & Preprocessing

### Final Features Used:
- Borough
- Complaint Type (grouped to 9 main categories)
- Location Type (grouped to 6 main categories)
- Incident ZIP Code
- Time of Day (Morning, Afternoon, Evening, Night)

All features were **categorical** and one-hot encoded, resulting in a reduced, optimized feature set. The target variable (Agency) was label-encoded for model compatibility.

---
## 🔎 Chi-Squared Feature Selection

We ran a **Chi-Squared test** between all categorical features and the target agency.  
All features had near-zero p-values — meaning they’re statistically related to the target.  
Top features (strongest correlation):
- **Complaint Type**
- **Location Type**
- **Time of Day**
- **Borough**

ZIP Code, Day of Week, and Month showed weaker but still relevant associations.

---
## 🔬 Models Tested

We trained and evaluated the following models:

| Model              | Highlights |
|-------------------|------------|
| **Logistic Regression** (SGDClassifier) | Fast and simple, but less accurate on rare agencies |
| **Decision Tree** | Strong interpretability and good precision |
| **Random Forest** | Slower and underperformed due to class imbalance |
| **XGBoost**        | Fastest and most accurate using GPU acceleration |

**Best Model:** XGBoost — it delivered the highest macro F1-Score and the lowest training time (~2.78 minutes with GPU).

---

## 📊 Evaluation Metrics

- **Macro F1-Score** (Primary): Treats all agencies equally, regardless of frequency
- **Precision & Recall**: Used to understand agency-specific performance
- **Training Time**: Considered for model scalability

---
