import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
data = pd.read_csv(url)

data = data.dropna()
le = LabelEncoder()

categorical_columns = ['origin', 'destination', 'train_type', 'fare']
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])
    
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
data['price_category'] = discretizer.fit_transform(data[['price']])

X = data[['origin', 'destination', 'train_type', 'fare']]
y = data['price_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі: {accuracy:.2f}")
