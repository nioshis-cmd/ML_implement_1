# ML_implement_1
🐧 Penguin Species Classification
Overview
This project implements a multi-class classification model to predict penguin species using physical measurements from the Palmer Penguins dataset.

The model classifies penguins into:
Adelie
Chinstrap
Gentoo

Features Used
The following numerical features were used:
bill_length_mm
bill_depth_mm
flipper_length_mm
body_mass_g

Target variable:
species
Machine Learning Pipeline
Load dataset using seaborn
Remove missing values
Select numerical features
Train-test split (80% / 20%)
Feature scaling using StandardScaler
Train Logistic Regression model

Evaluate using:
Accuracy
Confusion Matrix
Classification Report
Custom prediction using user input

Model
Algorithm: Logistic Regression
Scaling: StandardScaler
Multi-class handling: One-vs-Rest (default in sklearn)

Results
The model achieved high accuracy on the test set with strong precision, recall, and F1-scores across all three species.
