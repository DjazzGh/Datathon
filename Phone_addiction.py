import kagglehub
kagglehub.login()

phone_addiction_challenge_path = kagglehub.competition_download('phone-addiction-challenge')

print('Data source import complete.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('/kaggle/input/phone-addiction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/phone-addiction-challenge/test.csv')
train.head()

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nColumns in train:\n", train.columns)

train.isnull().sum()

train = train.drop(['id', 'Name'], axis=1)
test_ids = test['id']
test.drop(['id', 'Name'], axis=1, inplace=True)

train['School_Grade'] = train['School_Grade'].astype(str).str.extract('(\d+)').astype(int)
le = LabelEncoder()
train['Gender'] = le.fit_transform(train['Gender'])
train = pd.get_dummies(train, columns=['Phone_Usage_Purpose'], drop_first=True)

test['School_Grade'] = test['School_Grade'].astype(str).str.extract('(\d+)').astype(int)
le = LabelEncoder()
test['Gender'] = le.fit_transform(test['Gender'])
test = pd.get_dummies(test, columns=['Phone_Usage_Purpose'], drop_first=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# Initialize the new column with NaNs
train['Location_encoded'] = np.nan

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(train):
    # Split data
    train_fold = train.iloc[train_idx]
    val_fold = train.iloc[val_idx]

    # Compute mean Addiction_Level per wilaya on the train fold
    means = train_fold.groupby('Location')['Addiction_Level'].mean()

    # Map the means to the validation fold
    train.loc[val_fold.index, 'Location_encoded'] = val_fold['Location'].map(means)

# If any wilaya was unseen in a fold, fill its encoding with the global mean
train['Location_encoded'] = train['Location_encoded'].fillna(train['Addiction_Level'].mean())

# Final mapping from full training data
final_means = train.groupby('Location')['Addiction_Level'].mean()

# Map to test set
test['Location_encoded'] = test['Location'].map(final_means)

# Handle unseen wilayas in test
mean_addiction_level = train['Addiction_Level'].mean()
test['Location_encoded'] = test['Location_encoded'].fillna(mean_addiction_level)

# Drop original Location column
train = train.drop(['Location'], axis=1)
test = test.drop(['Location'], axis=1)

X = train.drop("Addiction_Level", axis=1)
y = train["Addiction_Level"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train = X
y_train = y
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print("MAE:", mean_absolute_error(y_val, y_pred))
print("RMSE:", mean_squared_error(y_val, y_pred, squared=False))
print("R²:", r2_score(y_val, y_pred))
# train
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)

# predict
y_pred = gbr.predict(X_val)

# evaluate
mse = mean_squared_error(y_val, y_pred)
print("MSE:", mse)
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print("XGBoost MSE:", mse)

top_features = [
    'Daily_Usage_Hours',
    'Time_on_Social_Media',
    'Time_on_Gaming',
    'Apps_Used_Daily',
    'Phone_Checks_Per_Day',
    'Sleep_Hours',
]
X_train_top = X_train[top_features]
X_val_top = X_val[top_features]
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

model_top = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.08,
    random_state=42
)

model_top.fit(X_train_top, y_train)
y_pred_top = model_top.predict(X_val_top)
mse_top = mean_squared_error(y_val, y_pred_top)
print("Reduced Feature Set MSE:", mse_top)

from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.08)
xgb_model.fit(X_train_top, y_train)
xgb_pred = xgb_model.predict(X_val_top)
print("XGBoost MSE:", mean_squared_error(y_val, xgb_pred))

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Lasso

stack = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('gbr', GradientBoostingRegressor()),
        ('lasso', Lasso(alpha=0.001))
    ],
    final_estimator=XGBRegressor()
)
stack.fit(X_train, y_train)
y_pred = stack.predict(X_val)
print("Stacked MSE:", mean_squared_error(y_val, y_pred))

import joblib
joblib.dump(stack, "final_model.pkl")

predictions = stack.predict(test)
# SUBMISSION FILE
submission = pd.DataFrame({
    'id':               test_ids,
    'Addiction_Level':  predictions
})
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
display(submission.head())
