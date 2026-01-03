export const labs = [
  {
    id: "lab1",
    title: "Lab 1: Distributions & Outliers",
    aim: "Develop a program to create histograms for all numerical features and analyze the distribution of each feature. Generate box plots for all numerical features and identify any outliers. Use California Housing dataset.",
    algorithm: [
      "1. Load the California Housing dataset using sklearn.",
      "2. Select only the numerical features from the dataframe.",
      "3. Iterate through each feature and plot a Histogram to visualize the distribution.",
      "4. Iterate through each feature and plot a Box Plot to visualize the spread.",
      "5. Calculate the IQR (Inter-Quartile Range) for each feature.",
      "6. Identify and count outliers that fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR."
    ],
    code: `import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Step 1: Load the California Housing dataset
data = fetch_california_housing(as_frame=True)
housing_df = data.frame
numerical_features = housing_df.select_dtypes(include=[np.number])

print("First 5 rows of numerical features:")
print(numerical_features.head())

# Step 2: Plot Histogram
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features.columns):
    plt.subplot(3, 3, i + 1)
    sns.histplot(data=housing_df[feature], kde=True, bins=45, color='blue')
    plt.title(f'Histogram of {feature}')
plt.tight_layout()
plt.show()

# Step 3: Plot Boxplot
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features.columns):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(data=housing_df[feature], color='orange')
    plt.title(f'Box Plot of {feature}')
plt.tight_layout()
plt.show()

# Step 4: Identify Outliers
print("\\nDescription of Outliers:")
outliers_summary = {}
for feature in numerical_features.columns:
    Q1 = housing_df[feature].quantile(0.25)
    Q3 = housing_df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = housing_df[(housing_df[feature] < lower_bound) | (housing_df[feature] > upper_bound)]
    outliers_summary[feature] = len(outliers)

print(outliers_summary)
print("\\nDataset Statistics:")
print(housing_df.describe())`
  },
  {
    id: "lab2",
    title: "Lab 2: Correlation Analysis",
    aim: "Develop a program to Compute the correlation matrix to understand the relationships between pairs of features. Visualize the correlation matrix using a heatmap to know which variables have strong positive/negative correlations. Create a pair plot to visualize pairwise relationships.",
    algorithm: [
      "1. Load the California Housing dataset.",
      "2. Compute the Correlation Matrix using the .corr() method.",
      "3. Visualize the matrix using a Seaborn Heatmap with annotations.",
      "4. Generate a Pair Plot to visualize relationships between all pairs of features."
    ],
    code: `import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Step 1: Load Data
california_data = fetch_california_housing(as_frame=True)
data = california_data.frame

# Step 2: Compute Correlation
correlation_matrix = data.corr()

# Step 3: Heatmap Visualization
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='BuPu', fmt='.2f', linewidths=1.5)
plt.title('Correlation Matrix of California Housing Features')
plt.show()

# Step 4: Pair Plot Visualization
sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Pair Plot of California Housing Features', y=1.02)
plt.show()`
  },
  {
    id: "lab3",
    title: "Lab 3: PCA on Iris Dataset",
    aim: "Develop a program to implement Principal Component Analysis (PCA) for reducing the dimensionality of the Iris dataset from 4 features to 2.",
    algorithm: [
      "1. Load the Iris dataset.",
      "2. Standardize the data (optional but recommended) or load directly into a DataFrame.",
      "3. Initialize the PCA model with n_components=2.",
      "4. Fit and transform the data to reduce dimensions.",
      "5. Create a scatter plot of PC1 vs PC2, coloring points by their target class."
    ],
    code: `import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load Data
iris = load_iris()
data = iris.data
target = iris.target
label_names = iris.target_names
iris_df = pd.DataFrame(data, columns=iris.feature_names)

# Step 2: Apply PCA
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

# Step 3: Create DataFrame for reduced data
reduced_df = pd.DataFrame(data_reduced, columns=['PC1', 'PC2'])
reduced_df['target'] = target

# Step 4: Visualize
plt.figure(figsize=(10, 8))
colors = ['r', 'g', 'b']
for i, target_name in enumerate(np.unique(target)):
    plt.scatter(
        reduced_df[reduced_df['target'] == target_name]['PC1'],
        reduced_df[reduced_df['target'] == target_name]['PC2'],
        c=colors[i],
        label=label_names[i]
    )

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Iris Dataset')
plt.legend()
plt.show()`
  },
  {
    id: "lab4",
    title: "Lab 4: Find-S Algorithm",
    aim: "For a given set of training data examples stored in a .CSV file, implement and demonstrate the Find-S algorithm to output a description of the set of all hypotheses consistent with the training examples.",
    algorithm: [
      "1. Load the dataset (enjoysport.csv).",
      "2. Initialize specific_h to the first positive instance.",
      "3. Iterate through all training examples.",
      "4. If an example is positive, compare it with specific_h.",
      "5. If a value differs, replace it with '?' (most general).",
      "6. Print the final specific hypothesis."
    ],
    code: `import pandas as pd
import numpy as np

# Note: Ensure 'enjoysport.csv' is in the same directory
# Or update the path below
data = pd.read_csv('enjoysport.csv')
attribute = np.array(data)[:, :-1]
target = np.array(data)[:, -1]

print("Training data:\\n")
print(data)

def train(att, tar):
    specific_h = None
    
    # Initialize with first positive example
    for i, val in enumerate(tar):
        if val == 'yes':
            specific_h = att[i].copy()
            break
            
    # Update hypothesis
    for i, val in enumerate(att):
        if tar[i] == 'yes':
            for x in range(len(specific_h)):
                if val[x] != specific_h[x]:
                    specific_h[x] = '?'
                else:
                    pass
    return specific_h

print("\\nFinal Hypothesis:")
print(train(attribute, target))`
  },
  {
    id: "lab5",
    title: "Lab 5: K-Nearest Neighbours (KNN)",
    aim: "Develop a program to implement k-Nearest Neighbour algorithm to classify randomly generated values. Label points based on a threshold and classify new test points for various values of k.",
    algorithm: [
      "1. Generate 100 random values in range [0,1].",
      "2. Assign classes 'Class1' if x <= 0.5 else 'Class2' for the first 50 points (Training Set).",
      "3. Use the remaining 50 points as the Test Set.",
      "4. Apply k-NN Classifier for k = 1, 2, 3, 4, 5, 20, 30.",
      "5. Predict and print results for each k."
    ],
    code: `import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Generate 100 random values in range [0,1]
data = np.random.rand(100).reshape(-1, 1)

# Label first 50 points based on the given condition
labels = np.array(["Class1" if x <= 0.5 else "Class2" for x in data[:50]])

# Split data into training (first 50) and testing (remaining 50)
train_data = data[:50]
train_labels = labels
test_data = data[50:]

# Apply k-NN for given k values
k_values = [1, 2, 3, 4, 5, 20, 30]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    predictions = knn.predict(test_data)
    
    print(f"Results for k={k}:")
    print(predictions[:10], "...") # Printing first 10 for brevity
    print("\\n")`
  },
  {
    id: "lab6",
    title: "Lab 6: Locally Weighted Regression",
    aim: "Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points.",
    algorithm: [
      "1. Generate synthetic data (sine wave with noise).",
      "2. Define the Gaussian Kernel function.",
      "3. Define the Locally Weighted Regression function.",
      "4. For each test point, calculate weights based on distance to training points.",
      "5. Solve the weighted linear regression equation.",
      "6. Plot the original data vs the LWR fit."
    ],
    code: `import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(x, xi, tau):
    return np.exp(-np.sum((x - xi) ** 2) / (2 * tau ** 2))

def locally_weighted_regression(x, X, y, tau):
    m = X.shape[0]
    weights = np.array([gaussian_kernel(x, X[i], tau) for i in range(m)])
    W = np.diag(weights)
    X_transpose_W = X.T @ W
    theta = np.linalg.inv(X_transpose_W @ X) @ X_transpose_W @ y
    return x @ theta

np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + 0.1 * np.random.randn(100)

X_bias = np.c_[np.ones(X.shape), X]
x_test = np.linspace(0, 2 * np.pi, 200)
x_test_bias = np.c_[np.ones(x_test.shape), x_test]

tau = 0.5
y_pred = np.array([locally_weighted_regression(xi, X_bias, y, tau) for xi in x_test_bias])

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Training Data', alpha=0.7)
plt.plot(x_test, y_pred, color='blue', label=f'LWR Fit (tau={tau})', linewidth=2)
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Locally Weighted Regression', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()`
  },
  {
    id: "lab7",
    title: "Lab 7: Linear & Polynomial Regression",
    aim: "Develop a program to demonstrate the working of Linear Regression and Polynomial Regression using Boston Housing and Auto MPG datasets.",
    algorithm: [
      "1. Linear Regression: Load Boston Housing dataset.",
      "2. Select 'rm' (average rooms) as feature and 'medv' (price) as target.",
      "3. Train a Linear Regression model and calculate MSE/R2 score.",
      "4. Polynomial Regression: Load Auto MPG dataset.",
      "5. Use 'displacement' to predict 'mpg' using Polynomial Features (degree=2).",
      "6. Visualize both regressions with scatter plots and trend lines."
    ],
    code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression_boston():
    print("--- Linear Regression (Boston Housing) ---")
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)
    
    # Selecting 'rm' (average number of rooms per dwelling)
    X = df[["rm"]].values
    # 'medv' is the median house price
    y = df["medv"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.plot(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("Average number of rooms (RM)")
    plt.ylabel("Median value of homes ($1000)")
    plt.title("Linear Regression - Boston Housing Dataset")
    plt.legend()
    plt.show()
    
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))
    print("\\n")

def polynomial_regression_auto_mpg():
    print("--- Polynomial Regression (Auto MPG) ---")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]
    
    data = pd.read_csv(url, sep='\\s+', names=column_names, na_values="?")
    data = data.dropna()
    
    X = data["displacement"].values.reshape(-1, 1)
    y = data["mpg"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    poly_model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LinearRegression())
    poly_model.fit(X_train, y_train)
    y_pred = poly_model.predict(X_test)
    
    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.scatter(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("Displacement")
    plt.ylabel("Miles per gallon (mpg)")
    plt.title("Polynomial Regression - Auto MPG Dataset")
    plt.legend()
    plt.show()
    
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

if __name__ == "__main__":
    linear_regression_boston()
    polynomial_regression_auto_mpg()`
  },
  {
    id: "lab8",
    title: "Lab 8: Decision Tree Classifier",
    aim: "Develop a program to demonstrate the working of the decision tree algorithm. Use Breast Cancer Data set.",
    algorithm: ["To be updated."],
    code: "# Pending Code"
  },
  {
    id: "lab9",
    title: "Lab 9: Naive Bayesian Classifier",
    aim: "Develop a program to implement the Naive Bayesian classifier considering Olivetti Face Data set.",
    algorithm: ["To be updated."],
    code: `import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Load Olivetti Faces dataset
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = data.data
y = data.target

# Step 2: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 3: Train Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Step 4: Predict
y_pred = gnb.predict(X_test)

# Step 5: Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("\\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 6: Visualize
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for ax, image, label, prediction in zip(axes.ravel(), X_test, y_test, y_pred):
    ax.imshow(image.reshape(64, 64), cmap=plt.cm.gray)
    ax.set_title(f"True: {label}   Pred: {prediction}", fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()`
  
  },
  {
    id: "lab10",
    title: "Lab 10: K-Means Clustering",
    aim: "Develop a program to implement k-means clustering using Wisconsin Breast Cancer data set.",
    algorithm: [
      "1. Load the Wisconsin Breast Cancer dataset.",
      "2. Standardize the features using StandardScaler.",
      "3. Apply K-Means clustering with k=2.",
      "4. Visualize results using PCA (2 components) to plot clusters.",
      "5. Compare predicted clusters with true labels using Confusion Matrix and Classification Report.",
      "6. Plot Centroids on the PCA scatter plot."
    ],
    code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

# Step 1: Load Data
data = load_breast_cancer()
X = data.data
y = data.target

# Step 2: Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Step 4: Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y, y_kmeans))
print("\\nClassification Report:")
print(classification_report(y, y_kmeans))

# Step 5: PCA for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df['Cluster'] = y_kmeans
df['True Label'] = y

# Plot 1: K-Means Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=100, edgecolor='black', alpha=0.7)
plt.title('K-Means Clustering of Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title="Cluster")
plt.show()

# Plot 2: True Labels
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='True Label', palette='coolwarm', s=100, edgecolor='black', alpha=0.7)
plt.title('True Labels of Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title="True Label")
plt.show()

# Plot 3: Centroids
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=100, edgecolor='black', alpha=0.7)
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering with Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title="Cluster")
plt.show()`
  }
];