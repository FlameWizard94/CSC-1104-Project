import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from basic_info import *

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def get_rates(y_true, predictions):
    classes = classes = ['Negative', 'Neutral', 'Positive']
    results = {}
    
    for cls in classes:
        TP = np.sum((y_true == cls) & (predictions == cls))
        TN = np.sum((y_true != cls) & (predictions != cls))
        FP = np.sum((y_true != cls) & (predictions == cls))
        FN = np.sum((y_true == cls) & (predictions != cls))
        P = TP + FN
        N = TN + FP
        
        # Calculate derived metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN) 
        precision = TP / (TP + FP) 
        recall = TP / (TP + FN) 
        sensitivity = TP / P
        f1 = 2 * (precision * recall) / (precision + recall)
        specificity = TN / N
        
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

def explain_rates(results):
    output = ''
    output += f"\nRates:\n\n"
    for cls, data in results.items():
        output += f"{cls}\n" + '-' * 30 + '\n'
        for rate, value in data.items():
            output += f"{rate}:\t{value:.4f}\n"
        output += '\n'

    return output

def CM(y_test, predictions, model):
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    cm_df = pd.DataFrame(cm, 
                        index=[f'Actual_{cls}' for cls in model.classes_], 
                        columns=[f'Predicted_{cls}' for cls in model.classes_])
    return cm_df

def Data_to_File(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    with open(f"model_info/{name}.txt", "w") as f:
        accuracy = accuracy_score(y_test, predictions)
        f.write(f"Accuracy:\t{accuracy}\n\n")

        cm = CM(y_test, predictions, model)
        f.write(f'Confusion Matrix\n\n{cm}\n')

        rates = get_rates(y_test, predictions)
        f.write(f"\n{explain_rates(rates)}")

        f.write(f"\nClassification Report\n{classification_report(y_test, predictions)}")
    

def prediction_table(name, model, X_test, y_test, predictions):
    if name != 'SVM':
        X_test_scaled = scaler.transform(X_test)
        prediction_probabilities = model.predict_proba(X_test_scaled)
        results_df = pd.DataFrame({
            'Actual_User_Experience': y_test.values,
            'Predicted_User_Experience': predictions,
            'Correct': (y_test.values == predictions)
        }, index=X_test.index)

        for i, class_name in enumerate(model.classes_):
            results_df[f'Probability_{class_name}'] = prediction_probabilities[:, i]

        filename = f'model_info/{name}.csv'
        results_df.to_csv(filename, index=True)


models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=50,        # 50 sequential trees
        learning_rate=0.1,      # How much each tree contributes
        max_depth=3,            # Smaller trees
        random_state=42
    ),
    'Ada Boosting' : AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    ),
    "Bagging" : BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),  
        n_estimators=50,        # Number of base estimators
        max_samples=0.8,        # Use 80% of data for each tree
        max_features=0.8,       # Use 80% of features for each tree
        bootstrap=True,         # Sample with replacement
        bootstrap_features=False, # Don't bootstrap features
        random_state=42,
        n_jobs=-1               # Use all available cores
    )
}

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
    X[col] = le.fit_transform(X[col].astype(str))  # Convert to string first
    label_encoders[col] = le
    print(f"Encoded {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")

# Convert boolean columns to integers (1 = True / 0 = False)
bool_columns = ['Is_Holiday', 'Is_Weekend', 'Helmet_Used']

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


print("Model Comparison:")
print("=" * 50)

for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"{name:25} Accuracy: {accuracy:.4f}")
    Data_to_File(name, model, X_train_scaled, X_test_scaled, y_train, y_test)
    prediction_table(name, model, X_test, y_test, predictions)
