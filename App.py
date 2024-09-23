import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

st.title("Credit Card Fraud Detection")

@st.cache_data
def load_data():
    data = pd.read_csv(r"https://drive.google.com/file/d/1JsdrNqQr5nQszcGQ7QqRYgFGllSESd_N/view?usp=drive_link")
    return data

data = load_data()

st.write("### Dataset Information")
st.write(data.info())
st.write("### Missing Values")
st.write(data.isnull().sum())


st.write("### Transaction Class Distribution")
LABELS = ["Normal", "Fraud"]
count_class = pd.value_counts(data['Class'], sort=True)
fig, ax = plt.subplots()
count_class.plot(kind='bar', rot=0, ax=ax)
ax.set_title("Transaction Class Distribution")
ax.set_xticks(range(2))
ax.set_xticklabels(LABELS)
ax.set_xlabel("Class")
ax.set_ylabel("Frequency")
st.pyplot(fig)

fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]
st.write("### Fraud Transaction Description")
st.write(fraud.Amount.describe())
st.write("### Normal Transaction Description")
st.write(normal.Amount.describe())


st.write("### Amount per Transaction by Class")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.suptitle("Amount per Transaction by Class")
bins = 50
ax1.hist(fraud.Amount, bins=bins)
ax1.set_title("Fraud")
ax2.hist(normal.Amount, bins=bins)
ax2.set_title("Normal")
plt.xlabel("Amount ($)")
plt.ylabel("Number of transactions")
plt.yscale('log')
st.pyplot(fig)

st.write("### Time of Transaction vs Amount by Class")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.suptitle("Time of Transaction vs Amount by Class")
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title("Fraud")
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title("Normal")
plt.xlabel("Time (In Seconds)")
plt.ylabel("Amount")
st.pyplot(fig)


st.write("### Data Preparation")
legit = data[data["Class"] == 0]
fraud = data[data["Class"] == 1]

legit_sample = legit.sample(n=492) 
new_dataset = pd.concat([legit_sample, fraud], axis=0)

X = new_dataset.drop(columns="Class", axis=1)
Y = new_dataset["Class"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
st.write("Training and testing dataset shapes:")
st.write(f"X_train: {X_train.shape}, X_test: {X_test.shape}, Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")

model_type = st.selectbox("Select a Model", ("Logistic Regression", "Random Forest", "K-Nearest Neighbors"))

if model_type == "Logistic Regression":
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, Y_train)
elif model_type == "Random Forest":
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_dist = {
        'n_estimators': [100, 200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, scoring='roc_auc', cv=3, verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(X_train, Y_train)
    model = random_search.best_estimator_
elif model_type == "K-Nearest Neighbors":
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn = KNeighborsClassifier()
    param_dist = {
        'n_neighbors': list(range(1, 31)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    random_search = RandomizedSearchCV(estimator=knn, param_distributions=param_dist, n_iter=100, scoring='roc_auc', cv=3, verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(X_train_scaled, Y_train)
    model = random_search.best_estimator_

# Predictions and Results
st.write("### Model Results")
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
st.write(f"Accuracy on training data: {training_data_accuracy}")

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
st.write(f"Accuracy on test data: {test_data_accuracy}")

Y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
roc_auc = roc_auc_score(Y_test, Y_prob) if Y_prob is not None else None
st.write(f"AUC-ROC Score: {roc_auc}")

st.write("Classification Report:")
st.text(classification_report(Y_test, X_test_prediction))

st.write("Confusion Matrix:")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(Y_test, X_test_prediction), annot=True, fmt="d", cmap="Blues", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("True")
st.pyplot(fig)
