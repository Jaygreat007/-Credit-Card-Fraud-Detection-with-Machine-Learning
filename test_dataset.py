import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# Load the dataset
# ===============================
df = pd.read_csv("creditcard.csv")

# Print shape and first few rows
print("Shape of the dataset:", df.shape)
print(df.head())

# ===============================
# Check for missing values
# ===============================
print("\nMissing values in each column:")
print(df.isnull().sum())

# ===============================
# Check original class distribution
# ===============================
print("\nClass distribution:")
print(df['Class'].value_counts())

print("\nPercentage of Fraudulent Transactions:")
print(df['Class'].value_counts(normalize=True) * 100)

# ===============================
# Balance the dataset using undersampling
# ===============================
df_majority = df[df['Class'] == 0]
df_minority = df[df['Class'] == 1]

df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=len(df_minority),
    random_state=42
)

df_balanced = pd.concat([df_majority_downsampled, df_minority])
df_balanced = df_balanced.sample(frac=1, random_state=42)

print("\nBalanced Class Distribution:")
print(df_balanced['Class'].value_counts())

# ===============================
# Split features and labels
# ===============================
X = df_balanced.drop('Class', axis=1)
y = df_balanced['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ===============================
# Train a Random Forest model
# ===============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===============================
# Make predictions and evaluate
# ===============================
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
