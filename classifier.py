import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from basic_info import *

from sklearn.svm import SVC # Support Vector Classifier
from sklearn.metrics import accuracy_score


def rates(y_true, y_pred):
    classes = classes = ['Negative', 'Neutral', 'Positive']
    results = {}
    
    for cls in classes:
        TP = np.sum((y_true == cls) & (y_pred == cls))
        TN = np.sum((y_true != cls) & (y_pred != cls))
        FP = np.sum((y_true != cls) & (y_pred == cls))
        FN = np.sum((y_true == cls) & (y_pred != cls))
        P = TP + FN
        N = TN + FP
        
        # Calculate derived metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN) 
        precision = TP / (TP + FP) 
        recall = TP / (TP + FN) 
        f1 = 2 * (precision * recall) / (precision + recall)
        specificity = TN / N
        sensitivity = TP / P
        
        results[cls] = {
            'P' : P,
            'N' : N,
            'TP': TP, 
            'TN': TN, 
            'FP': FP, 
            'FN': FN,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1-Score': f1,
            'sensitivity' : sensitivity,
            'specificity' : specificity
        }
    
    return results

# Load YOUR dataset
df = pd.read_csv('Dataset_BicycleUse.csv')
df = fix_weekend(df)

print(f"Target variable distribution:\n{df['User_Experience'].value_counts()}\n")

X = df.drop('User_Experience', axis=1)  # All columns EXCEPT User_Experience
y = df['User_Experience']  # Only User_Experience



# Convert categorical text to numbers so classifier understands them
label_encoders = {}
categorical_columns = ['Bike_Type', 'Gender', 'Occupation', 'Weather', 
                      'Day_of_Week', 'Time_of_Day', 'Purpose_of_Ride', 'Road_Condition']

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))  # Convert to string first to handle any issues
    label_encoders[col] = le
    print(f"Encoded {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")

# Convert boolean columns to integers (1 = True / 0 = False)
bool_columns = ['Is_Holiday', 'Is_Weekend', 'Helmet_Used']
315
for col in bool_columns:
    X[col] = X[col].astype(int)
    print(f"Converted {col} to integers")

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,  # For reproducible results
    stratify=y  # Keeps same class distribution in both sets
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training target distribution:\n{y_train.value_counts()}")
print(f"\nTest target distribution:\n{y_test.value_counts()}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nFeatures scaled successfully!")

# Train classifier
print("\nTraining Logistic Regression Classifier...")
classifier = LogisticRegression(random_state=42, max_iter=1000)
classifier.fit(X_train_scaled, y_train)

print("Training completed!")

# Make predictions and evaluate
y_pred = classifier.predict(X_test_scaled)
y_pred_probs = classifier.predict_proba(X_test_scaled) # probs for ROC

# Check performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nDetailed Performance:")
print(classification_report(y_test, y_pred))

training_columns = list(X_test.columns)
results_df = pd.DataFrame({
    'Actual_User_Experience': y_test.values,
    'Predicted_User_Experience': y_pred,
    'Correct': (y_test.values == y_pred)
}, index=X_test.index)

print(f"Predictions\n\n{list(results_df.columns)}\n\n{results_df}")

# test predictiopns
print(f"Test predictions\n{y_pred[:10]}\n\nTest prediction probabilities\n{y_pred_probs[:10]}")


# Add probability columns for each class
for i, class_name in enumerate(classifier.classes_):
    results_df[f'Probability_{class_name}'] = y_pred_probs[:, i]


results = rates(y_test, y_pred)
print("\nRates:\n")
for cls, data in results.items():
    print(f"{cls}\n" + '-' * 30)
    for rate, value in data.items():
        print(f"{rate}:\t{value:.4f}")
    print()

print()


# Show summary statistics
print(f"\nPREDICTION SUMMARY:")
print(f"Total test samples: {len(results_df)}")
print(f"Correct predictions: {results_df['Correct'].sum()} ({results_df['Correct'].mean()*100:.2f}%)")
print(f"Incorrect predictions: {(~results_df['Correct']).sum()} ({(~results_df['Correct']).mean()*100:.2f}%)")

print(f"\nCONFUSION MATRIX (What was predicted vs actual):")
cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
cm_df = pd.DataFrame(cm, 
                     index=[f'Actual_{cls}' for cls in classifier.classes_], 
                     columns=[f'Predicted_{cls}' for cls in classifier.classes_])
print(cm_df)


print(f"\nFINAL ACCURACY: {accuracy_score(y_test, y_pred):.4f} ({accuracy_score(y_test, y_pred)*100:.2f}%)")
