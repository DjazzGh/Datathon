import pandas as pd

# Load the datasets
train_data = pd.read_csv('/content/churn-detection/train.csv')
test_data = pd.read_csv('/content/churn-detection/test.csv')

# Convert TotalCharges to numeric, coercing errors to NaN
train_data['TotalCharges'] = pd.to_numeric(train_data['TotalCharges'], errors='coerce')
test_data['TotalCharges'] = pd.to_numeric(test_data['TotalCharges'], errors='coerce')

# Set TotalCharges to 0 where tenure is 0
train_data.loc[train_data['tenure'] == 0, 'TotalCharges'] = 0
test_data.loc[test_data['tenure'] == 0, 'TotalCharges'] = 0

# Impute remaining missing TotalCharges with mean from training data
total_charges_mean = train_data['TotalCharges'].mean()
train_data['TotalCharges'].fillna(total_charges_mean, inplace=True)
test_data['TotalCharges'].fillna(total_charges_mean, inplace=True)

# List of categorical columns (adjust based on your dataset)
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod']

#Convert categorical columns to category type
for col in categorical_cols:
    train_data[col] = train_data[col].astype('category')
    test_data[col] = test_data[col].astype('category')

# Impute missing categorical values by creating a new category 'Missing'
for col in categorical_cols:
    train_data[col] = train_data[col].cat.add_categories('Missing').fillna('Missing')
    test_data[col] = test_data[col].cat.add_categories('Missing').fillna('Missing')

#Convert categorical columns to category type for efficiency
for col in categorical_cols:
    train_data[col] = train_data[col].astype('category')
    test_data[col] = test_data[col].astype('category')

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    train_data[col] = train_data[col].map({'Yes': 1, 'No': 0})
    test_data[col] = test_data[col].map({'Yes': 1, 'No': 0})

# Encode the target variable 'Churn' in train_df (if present)
if 'Churn' in train_data.columns:
    train_data['Churn'] = train_data['Churn'].map({'Yes': 1, 'No': 0})

nominal_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaymentMethod']

train_data = pd.get_dummies(train_data, columns=nominal_cols) # Removed drop_first=True
test_data = pd.get_dummies(test_data, columns=nominal_cols) # Removed drop_first=True

if 'id' in train_data.columns:
    train_data.drop('id', axis=1, inplace=True)
if 'id' in test_data.columns:
    test_data.drop('id', axis=1, inplace=True)

test_data = test_data.reindex(columns=train_data.columns.drop('Churn', errors='ignore'), fill_value=0)
train_data.to_csv('cleaned_train.csv', index=False)
test_data.to_csv('cleaned_test.csv', index=False)
import pandas as pd
import numpy as np

# Load the cleaned datasets
train_df = pd.read_csv('/content/cleaned_train.csv')
test_df = pd.read_csv('/content/cleaned_test.csv')

# Feature 1: Average Monthly Charges (AvgMonthlyCharges)
# TotalCharges / tenure if tenure > 0, else MonthlyCharges
train_df['AvgMonthlyCharges'] = np.where(train_df['tenure'] > 0,
                                         train_df['TotalCharges'] / train_df['tenure'],
                                         train_df['MonthlyCharges'])
test_df['AvgMonthlyCharges'] = np.where(test_df['tenure'] > 0,
                                        test_df['TotalCharges'] / test_df['tenure'],
                                        test_df['MonthlyCharges'])

# Feature 2: Has Both Services (HasBothServices)
# 1 if customer has both phone and internet services, else 0
train_df['HasBothServices'] = ((train_df['PhoneService'] == 1) & (train_df['InternetService_No'] == 0)).astype(int)
test_df['HasBothServices'] = ((test_df['PhoneService'] == 1) & (test_df['InternetService_No'] == 0)).astype(int)

# Feature 3: Number of Additional Services (NumAdditionalServices)
# Sum of additional services the customer has subscribed to
additional_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
yes_columns = [f'{service}_Yes' for service in additional_services]
train_df['NumAdditionalServices'] = train_df[yes_columns].sum(axis=1)
test_df['NumAdditionalServices'] = test_df[yes_columns].sum(axis=1)

# Feature 4: Long-Term Contract (LongTermContract)
# 1 if customer is on a one-year or two-year contract, else 0
train_df['LongTermContract'] = train_df[['Contract_One year', 'Contract_Two year']].max(axis=1)
test_df['LongTermContract'] = test_df[['Contract_One year', 'Contract_Two year']].max(axis=1)

# Verify the new features by displaying the first few rows
print("Training data with new features:")
print(train_df.head())

# Save the enhanced datasets
train_df.to_csv('enhanced_train.csv', index=False)
test_df.to_csv('enhanced_test.csv', index=False)

print("Feature engineering completed. Enhanced datasets saved as 'enhanced_train.csv' and 'enhanced_test.csv'.")

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer

# Load the enhanced datasets
train_df = pd.read_csv('/content/enhanced_train.csv')
test_df = pd.read_csv('/content/enhanced_test.csv')

# Separate features and target
X_train = train_df.drop('Churn', axis=1)
y_train = train_df['Churn']

# Calculate scale_pos_weight to handle class imbalance
num_negative = (y_train == 0).sum()
num_positive = (y_train == 1).sum()
scale_pos_weight = num_negative / num_positive

# Initialize XGBoost classifier
clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False
)

# Define hyperparameter search space
param_dist = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 400, 500]
}

# Set up Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Set up Randomized Search
random_search = RandomizedSearchCV(
    estimator=clf,
    param_distributions=param_dist,
    n_iter=20,
    scoring=make_scorer(f1_score),
    cv=skf,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit the random search
random_search.fit(X_train, y_train)

# Print best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best F1 score:", random_search.best_score_)

# Get the best model
best_model = random_search.best_estimator_

# Load original test data for submission IDs
original_test = pd.read_csv('/content/churn-detection/test.csv')

# Make predictions on test set
X_test = test_df
y_pred = best_model.predict(X_test)

# Convert predictions back to Yes/No labels
y_pred_labels = ['Yes' if pred == 1 else 'No' for pred in y_pred]

# Create submission file
submission = pd.DataFrame({
    'id': original_test['id'],
    'Churn': y_pred_labels
})
submission.to_csv('submission.csv', index=False)

print("Predictions completed and saved to 'submission.csv'")