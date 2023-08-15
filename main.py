# Importing modules
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

import joblib

# Import Data
customers_data = pd.read_csv("data/Passanger_booking_data.csv")

# To save memory I will reduce the datatypes to int8, int16 and float16, and convert object to category.
customers_data['num_passengers'] = customers_data['num_passengers'].astype('int8')
customers_data['sales_channel'] = customers_data['sales_channel'].astype('category')
customers_data['trip_type'] = customers_data['trip_type'].astype('category')
customers_data['purchase_lead'] = customers_data['purchase_lead'].astype('int16')
customers_data['length_of_stay'] = customers_data['length_of_stay'].astype('int16')
customers_data['flight_hour'] = customers_data['flight_hour'].astype('int8')
customers_data['flight_day'] = customers_data['flight_day'].astype('category')
customers_data['route'] = customers_data['route'].astype('category')
customers_data['booking_origin'] = customers_data['booking_origin'].astype('category')
customers_data['wants_extra_baggage'] = customers_data['wants_extra_baggage'].astype('int8')
customers_data['wants_preferred_seat'] = customers_data['wants_preferred_seat'].astype('int8')
customers_data['wants_in_flight_meals'] = customers_data['wants_in_flight_meals'].astype('int8')
customers_data['flight_duration'] = customers_data['flight_duration'].astype('float16')
customers_data['booking_complete'] = customers_data['booking_complete'].astype('int8')

# Handling Duplicate Rows
customers_data = customers_data.drop_duplicates()

# Since 99% of customers are willing to book round trip, I am going to drop other rows.
customers_data = customers_data[customers_data['trip_type'] == 'RoundTrip']

# There are almost 104 countries and very few passengers are there from these countries.
# So, I am going to aggregate countries who have less than 1,000 passengers to 'Other'.
freq = customers_data['booking_origin'].value_counts()
smallCategories = freq[freq < 1000].index
customers_data['booking_origin'] = customers_data['booking_origin'].replace(smallCategories.tolist(), "Other")

# I will not use 'trip_type' as all are 'RoundTrip' and I will not use 'route' as it has 799 unique values.
x = customers_data.drop(columns=['booking_complete', 'trip_type', 'route'])
y = customers_data['booking_complete']

# Split Dataset into train and test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Handling Categorical Data
transform1 = ColumnTransformer([("YeoJohnsonDist", PowerTransformer(), [0, 2, 3, 4, 10])], remainder="passthrough")

# Handling Categorical Data
transform2 = ColumnTransformer([("OneHotEncoding", OneHotEncoder(), [5, 6, 7])], remainder="passthrough")

# Creating a Pipeline
pipe = Pipeline([
    ("Transform1", transform1),
    ("Transform2", transform2)
])

# Transforming/ Performing Feature Engineering on x_train and x_test
x_train = pipe.fit_transform(x_train)
x_test = pipe.transform(x_test)

# Feature Extraction
pca = PCA(n_components=13)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

start = time.time()
estimators = [
    ('svc', SVC()),
    ('sgd', SGDClassifier()),
    ('logistic', LogisticRegression()),
    ('adaboost', AdaBoostClassifier())
]
streg = StackingClassifier(estimators=estimators,
                           final_estimator=GradientBoostingClassifier(),
                           cv=5)
streg.fit(x_train, y_train)
y_pred = streg.predict(x_test)

print("\nTime Taken for Stacking is", np.round(time.time() - start, 2), "seconds")
print("Accuracy Score:", np.round(accuracy_score(y_test, y_pred) * 100, 2))
print("ROC AUC Score:", np.round(roc_auc_score(y_test, y_pred) * 100, 2))
print("Classification Report:", classification_report(y_test, y_pred))

# Saving the model
joblib.dump(streg, 'airlines_booking_model')
joblib.dump(pipe, 'pipe')
joblib.dump(pca, 'pca')
