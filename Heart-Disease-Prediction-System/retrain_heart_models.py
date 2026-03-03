"""
Retrain heart disease models with current sklearn version.
Fixes:
  1. sklearn version mismatch (old pickles unusable)
  2. 10,973 duplicate rows causing 100% fake accuracy and data leakage
  3. Inverted target labels (in this dataset target=0 has disease markers,
     target=1 has healthy markers — opposite of standard convention)
After retraining: model predicts 0=HEALTHY, 1=AT RISK (standard convention)
"""
import os, pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

CSV_PATH = os.path.join(os.path.dirname(__file__), 'Machine_Learning', 'heart.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')

df = pd.read_csv(CSV_PATH)
print(f"Raw dataset: {df.shape[0]} rows")

# Step 1: Remove duplicates (10,973 of 11,275 rows are duplicates)
df = df.drop_duplicates()
print(f"After dedup: {df.shape[0]} unique rows")

# Step 2: Fix inverted labels
# In this dataset target=0 has disease markers (high ca, low thalach, high exang)
# target=1 has healthy markers — opposite of standard convention (0=healthy,1=sick)
# Flip so that: 0=HEALTHY, 1=AT RISK (standard)
df['target'] = 1 - df['target']
print(f"Labels corrected — HEALTHY(0): {(df.target==0).sum()}, AT RISK(1): {(df.target==1).sum()}")

FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
X = df[FEATURES]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    'logistic_regression': LogisticRegression(max_iter=2000, C=1.0, random_state=42),
    'random_forest':       RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=8, random_state=42),
    'decision_tree':       DecisionTreeClassifier(max_depth=5, min_samples_split=8, random_state=42),
    'knn':                 KNeighborsClassifier(n_neighbors=7),
    'naive_bayes':         GaussianNB(),
}

model_info = {}
print(f"\n{'Model':<25} {'Train':>8} {'Test':>8} {'CV-5':>8}")
print('-' * 55)

for key, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
    test_acc  = accuracy_score(y_test,  model.predict(X_test))  * 100
    cv_acc    = cross_val_score(model, X, y, cv=5).mean() * 100
    model_info[key] = {'train_accuracy': train_acc, 'test_accuracy': test_acc, 'cv_accuracy': cv_acc}
    print(f"{key:<25} {train_acc:>7.1f}% {test_acc:>7.1f}% {cv_acc:>7.1f}%")
    with open(os.path.join(MODELS_DIR, f'{key}.pkl'), 'wb') as f:
        pickle.dump(model, f)

# Use CV accuracy as the honest accuracy metric (avoids train/test luck)
for key in model_info:
    model_info[key]['test_accuracy'] = model_info[key]['cv_accuracy']

with open(os.path.join(MODELS_DIR, 'model_info.pkl'), 'wb') as f:
    pickle.dump(model_info, f)

best = max(model_info, key=lambda k: model_info[k]['cv_accuracy'])
print(f"\nBest model by CV: {best} ({model_info[best]['cv_accuracy']:.1f}%)")
print("All models saved.")

# Sanity check with medically known profiles
print("\n--- Sanity check (expect HEALTHY/AT RISK as labeled) ---")
best_model_path = os.path.join(MODELS_DIR, f'{best}.pkl')
with open(best_model_path, 'rb') as f:
    m = pickle.load(f)

tests = [
    # (label, expected, values)
    # Healthy profiles: young, good HR, no angina, no blocked vessels
    ("HEALTHY - young male, good vitals",   "HEALTHY", [29, 1, 2, 120, 180, 0, 2, 210, 0, 0.0, 2, 0, 2]),
    ("HEALTHY - young female, no symptoms", "HEALTHY", [35, 0, 2, 120, 180, 0, 0, 190, 0, 0.0, 2, 0, 2]),
    ("HEALTHY - mid-age, low risk",         "HEALTHY", [45, 1, 2, 128, 210, 0, 0, 180, 0, 0.2, 2, 0, 2]),
    # AT RISK: elderly, low HR, exercise angina, blocked vessels, ST depression
    ("AT RISK  - elderly male, high BP",    "AT RISK",  [63, 1, 0, 145, 233, 1, 0, 150, 0, 2.3, 0, 2, 3]),
    ("AT RISK  - old female, exang+ca",     "AT RISK",  [67, 0, 0, 160, 286, 0, 0, 108, 1, 1.5, 1, 3, 3]),
    ("AT RISK  - exang, ST depression",     "AT RISK",  [57, 1, 0, 150, 276, 0, 0, 112, 1, 3.5, 0, 2, 3]),
]
cols = FEATURES
all_ok = True
for label, expected, vals in tests:
    sample = pd.DataFrame([vals], columns=cols)
    p = m.predict(sample)[0]
    got = "HEALTHY" if p == 0 else "AT RISK"
    ok = "OK" if got == expected else "WRONG"
    if ok == "WRONG":
        all_ok = False
    print(f"  [{ok}] {label:<38} expected={expected:<8} got={got}")

print()
if all_ok:
    print("All sanity checks passed!")
else:
    print("Some checks failed — review label encoding.")
