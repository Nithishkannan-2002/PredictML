# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the Titanic dataset (Make sure to download 'train.csv' from Kaggle)
df = pd.read_csv("C:/Users/nithi/Downloads/archive/Titanic-Dataset.csv")

# Display the first few rows
print(df.head())

# Check basic information about the dataset
print(df.info())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualizing the distribution of target variable 'Survived'
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

# Visualizing the distribution of passengers by class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()

# Visualizing Age distribution
sns.histplot(df['Age'].dropna(), kde=True, bins=30)
plt.title('Age Distribution')
plt.show()

#Summary  Statistics
print(df.describe())
# Check for missing values
print(df.isnull().sum())

# Visualizing the distribution of target variable 'Survived'
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()


# Visualizing the distribution of passengers by class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()

# Visualizing Age distribution
sns.histplot(df['Age'].dropna(), kde=True, bins=30)
plt.title('Age Distribution')
plt.show()



# Handle missing values
# Fill missing 'Age' with median age and 'Embarked' with the mode
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop the 'Name', 'Ticket', and 'Cabin' columns as they are not useful for this model
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Encode categorical variables
# 'Sex' is a binary categorical variable (male=0, female=1)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# 'Embarked' has three categories (C, Q, S), so we can use get_dummies to create dummy variables
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Feature selection: Choose features and the target variable
X = df.drop('Survived', axis=1)  # Features (input variables)
y = df['Survived']  # Target variable (output)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
