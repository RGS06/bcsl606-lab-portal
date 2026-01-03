export const manuals: any = {
  "lab1": {
    title: "Data Visualization & Outlier Detection",
    aim: "Develop a program to create histograms for all numerical features and analyze the distribution of each feature. Generate box plots for all numerical features and identify any outliers using the California Housing dataset.",
    dataset: "california_housing (Built-in)", 
    datasetUrl: null, 
    steps: [
      {
        title: "1. Import Libraries & Load Data",
        explanation: "We import the necessary libraries for data manipulation (pandas, numpy) and visualization (seaborn, matplotlib). We then fetch the California Housing dataset directly from sklearn.",
        code: `import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Step 1: Load the California Housing dataset
data = fetch_california_housing(as_frame=True)
housing_df = data.frame

# Display first few rows to verify
print(housing_df.head())`
      },
      {
        title: "2. Create Histograms",
        explanation: "We visualize the distribution of all numerical features using histograms. The Kernel Density Estimate (KDE) line helps us see the shape of the distribution.",
        code: `# Step 2: Create histograms for numerical features
numerical_features = housing_df.select_dtypes(include=[np.number]).columns

# Plot histograms
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(3, 3, i + 1)
    sns.histplot(housing_df[feature], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()`
      },
      {
        title: "3. Generate Box Plots",
        explanation: "Box plots are crucial for identifying outliers visually. The 'whiskers' extend to the rest of the distribution, while points outside are potential outliers.",
        code: `# Step 3: Generate box plots for numerical features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x=housing_df[feature], color='orange')
    plt.title(f'Box Plot of {feature}')
plt.tight_layout()
plt.show()`
      },
      {
        title: "4. Detect Outliers (IQR Method)",
        explanation: "We use the Inter-Quartile Range (IQR) method to mathematically identify outliers. Any data point falling below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is flagged.",
        code: `# Step 4: Identify outliers using the IQR method
print("Outliers Detection:")
outliers_summary = {}
for feature in numerical_features:
    Q1 = housing_df[feature].quantile(0.25)
    Q3 = housing_df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = housing_df[(housing_df[feature] < lower_bound) | (housing_df[feature] > upper_bound)]
    outliers_summary[feature] = len(outliers)
    print(f"{feature}: {len(outliers)} outliers")`
      }
    ]
  },
  "lab2": {
    title: "Correlation Matrix & Pair Plots",
    aim: "Develop a program to Compute the correlation matrix to understand the relationships between pairs of features. Visualize the correlation matrix using a heatmap and create a pair plot to visualize pairwise relationships.",
    dataset: "california_housing (Built-in)", 
    datasetUrl: null,
    steps: [
      {
        title: "1. Load Dataset",
        explanation: "We load the California Housing dataset as a pandas DataFrame. This dataset is built into Scikit-Learn, so no external CSV download is required.",
        code: `import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Step 1: Load the California Housing Dataset
california_data = fetch_california_housing(as_frame=True)
data = california_data.frame

print("Dataset Loaded Successfully")
print(data.head())`
      },
      {
        title: "2. Compute Correlation Matrix",
        explanation: "We calculate the correlation coefficients between all pairs of numerical features. A value close to 1 implies strong positive correlation, while -1 implies strong negative correlation.",
        code: `# Step 2: Compute the correlation matrix
correlation_matrix = data.corr()

print("Correlation Matrix Computed")
print(correlation_matrix.head())`
      },
      {
        title: "3. Heatmap Visualization",
        explanation: "A heatmap visually represents the correlation matrix. 'Coolwarm' colors help distinguish between positive (red) and negative (blue) correlations.",
        code: `# Step 3: Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of California Housing Features')
plt.show()`
      },
      {
        title: "4. Pair Plot Visualization",
        explanation: "A pair plot creates a grid of scatter plots for every pair of features, with histograms or KDEs on the diagonal. This helps identify linear or non-linear relationships quickly.",
        code: `# Step 4: Create a pair plot to visualize pairwise relationships
sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Pair Plot of California Housing Features', y=1.02)
plt.show()`
      }
    ]
  },
  "lab3": {
    title: "Principal Component Analysis (PCA)",
    aim: "Develop a program to implement Principal Component Analysis (PCA) for reducing the dimensionality of the Iris dataset from 4 features to 2.",
    dataset: "iris (Built-in)",
    datasetUrl: null,
    steps: [
      {
        title: "1. Load Data & Libraries",
        explanation: "We import the Iris dataset from sklearn and load the feature data and target labels. Converting it to a Pandas DataFrame helps in easier manipulation.",
        code: `import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
data = iris.data
labels = iris.target
label_names = iris.target_names

# Convert to a DataFrame for better visualization
iris_df = pd.DataFrame(data, columns=iris.feature_names)`
      },
      {
        title: "2. Apply PCA",
        explanation: "We initialize the PCA object with n_components=2 to reduce the dimensions from 4 to 2. The fit_transform method calculates the principal components and applies the transformation.",
        code: `# Perform PCA to reduce dimensionality to 2
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

# Create a DataFrame for the reduced data
reduced_df = pd.DataFrame(data_reduced, columns=['Principal Component 1', 'Principal Component 2'])
reduced_df['Label'] = labels`
      },
      {
        title: "3. Visualize Reduced Data",
        explanation: "We plot the two new Principal Components on a 2D Scatter Plot. Each color represents a different species of Iris, showing how well PCA separated the classes.",
        code: `# Plot the reduced data
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for i, label in enumerate(np.unique(labels)):
    plt.scatter(
        reduced_df[reduced_df['Label'] == label]['Principal Component 1'],
        reduced_df[reduced_df['Label'] == label]['Principal Component 2'],
        label=label_names[label],
        color=colors[i]
    )

plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show()`
      }
    ]
  },
  "lab4": {
    title: "Find-S Algorithm",
    aim: "For a given set of training data examples stored in a .CSV file, implement and demonstrate the Find-S algorithm to output a description of the set of all hypotheses consistent with the training examples.",
    dataset: "enjoysport.csv",
    datasetUrl: "/datasets/enjoysport.csv",
    steps: [
      {
        title: "1. Load Data & Prepare",
        explanation: "We load the training data from 'enjoysport.csv'. We separate the attributes (features) from the target variable (yes/no).",
        code: `import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('enjoysport.csv') # Ensure file is in local directory

attribute = np.array(data)[:,:-1]
target = np.array(data)[:,-1]

print("Training data:\\n")
print(data)`
      },
      {
        title: "2. Implement Find-S Logic",
        explanation: "We initialize the hypothesis with the first positive example. Then, we iterate through all positive examples, generalizing the hypothesis by replacing mismatching attributes with '?'.",
        code: `def train(att,tar):
  specific_h = None
  
  # Initialize with first positive example
  for i,val in enumerate(tar):
    if val=='yes':
      specific_h=att[i].copy()
      break
      
  # Update hypothesis for subsequent positive examples
  for i,val in enumerate(att):
    if tar[i]=='yes':
      for x in range(len(specific_h)):
        if val[x]!=specific_h[x]:
          specific_h[x]='?'
        else:
          pass
  return specific_h`
      },
      {
        title: "3. Output Final Hypothesis",
        explanation: "We print the final specific hypothesis that is consistent with all positive training examples.",
        code: `print("\\nFinal Specific Hypothesis:")
print(train(attribute,target))`
      }
    ]
  },
  "lab5": {
    title: "k-Nearest Neighbour (KNN)",
    aim: "Develop a program to implement k-Nearest Neighbour algorithm to classify randomly generated values. Label the first 50 points based on a threshold and classify the remaining 50 using KNN with various k values.",
    dataset: "Generated (Synthetic)",
    datasetUrl: null,
    steps: [
      {
        title: "1. Generate Data & Labels",
        explanation: "We create 100 random values. The first 50 are labeled manually: 'Class1' if x <= 0.5, else 'Class2'. The remaining 50 are used as test data.",
        code: `import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Generate 100 random values in range [0,1]
data = np.random.rand(100).reshape(-1, 1)

# Label first 50 points based on the given condition
labels = np.array(["Class1" if x <= 0.5 else "Class2" for x in data[:50]])

# Split data into training (first 50) and testing (remaining 50)
train_data, train_labels = data[:50], labels
test_data = data[50:]`
      },
      {
        title: "2. Train & Test KNN Models",
        explanation: "We iterate through a list of 'k' values. For each k, we train a new KNeighborsClassifier on the first 50 points and predict classes for the next 50.",
        code: `# Apply k-NN for given k values
k_values = [1, 2, 3, 4, 5, 20, 30]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    predictions = knn.predict(test_data)

    print(f"Results for k={k}:")
    print(predictions, "\\n")`
      }
    ]
  },
  "lab6": {
    title: "Locally Weighted Regression",
    aim: "Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs.",
    dataset: "Generated (Synthetic)",
    datasetUrl: null,
    steps: [
      {
        title: "1. Define Kernel & Regression Functions",
        explanation: "We define the Gaussian Kernel to calculate weights based on distance. The locally_weighted_regression function computes theta using the weighted normal equation.",
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
    return x @ theta`
      },
      {
        title: "2. Generate Noisy Sine Data",
        explanation: "We generate synthetic data following a sine wave pattern and add random noise to test the regression's ability to fit non-linear data.",
        code: `np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + 0.1 * np.random.randn(100)
X_bias = np.c_[np.ones(X.shape), X]

x_test = np.linspace(0, 2 * np.pi, 200)
x_test_bias = np.c_[np.ones(x_test.shape), x_test]`
      },
      {
        title: "3. Predict & Plot",
        explanation: "We predict values for the test set using tau=0.5 and visualize the fitted curve against the noisy training data.",
        code: `tau = 0.5
y_pred = np.array([locally_weighted_regression(xi, X_bias, y, tau) for xi in x_test_bias])

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Training Data', alpha=0.7)
plt.plot(x_test, y_pred, color='blue', label=f'LWR Fit (tau={tau})', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Locally Weighted Regression')
plt.legend()
plt.grid(alpha=0.3)
plt.show()`
      }
    ]
  },
  "lab7": {
    title: "Linear & Polynomial Regression",
    aim: "Develop a program to demonstrate the working of Linear Regression (Boston Housing) and Polynomial Regression (Auto MPG).",
    dataset: "Boston Housing & Auto MPG",
    datasetUrl: null,
    steps: [
      {
        title: "1. Linear Regression (Boston Housing)",
        explanation: "We load the Boston Housing dataset, split it, and train a simple Linear Regression model to predict house value (medv) based on rooms (rm).",
        code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression_boston():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)

    X = df[["rm"]].values 
    y = df["medv"].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Plotting code in next step...
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))`
      },
      {
        title: "2. Visualize Linear Model",
        explanation: "We visualize the linear fit (Red Line) against the actual test data (Blue Dots).",
        code: `    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.plot(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("Average number of rooms (RM)")
    plt.ylabel("Median value of homes ($1000)")
    plt.title("Linear Regression - Boston Housing")
    plt.legend()
    plt.show()`
      },
      {
        title: "3. Polynomial Regression (Auto MPG)",
        explanation: "We load the Auto MPG dataset, clean it, and train a Polynomial Regression model (degree=2) to predict MPG based on Displacement.",
        code: `def polynomial_regression_auto_mpg():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]

    data = pd.read_csv(url, sep='\\s+', names=column_names, na_values="?")
    data = data.dropna()

    X = data["displacement"].values.reshape(-1, 1)
    y = data["mpg"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly_model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LinearRegression())
    poly_model.fit(X_train, y_train)
    y_pred = poly_model.predict(X_test)`
      },
      {
        title: "4. Visualize Polynomial Model",
        explanation: "We visualize the polynomial curve fit (Red Dots) against the actual data (Blue Dots), showing how it captures the non-linear relationship.",
        code: `    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.scatter(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("Displacement")
    plt.ylabel("Miles per gallon (mpg)")
    plt.title("Polynomial Regression - Auto MPG")
    plt.legend()
    plt.show()`
      }
    ]
  },
  "lab8": {
    title: "Decision Tree Classifier",
    aim: "Develop a program to demonstrate the working of the decision tree algorithm using Breast Cancer dataset and apply it to classify a new custom sample.",
    dataset: "Breast Cancer (Built-in)",
    datasetUrl: null,
    steps: [
      {
        title: "1. Train on Breast Cancer Data",
        explanation: "We load the Breast Cancer dataset from sklearn and train a Decision Tree Classifier with a max depth of 4.",
        code: `import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Step 1: Load and train on Breast Cancer dataset
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

bc_model = DecisionTreeClassifier(max_depth=4, random_state=42)
bc_model.fit(X_cancer, y_cancer)
print("Breast Cancer model trained successfully.")`
      },
      {
        title: "2. Prepare Custom Dataset",
        explanation: "We create a custom pandas DataFrame representing candidate profiles (cgpa, interactiveness, etc.) and encode categorical values to numbers.",
        code: `# Step 2: Custom sample dataset
sample_data = pd.DataFrame({
    'cgpa': [9.2, 8.5, 9.0, 7.5, 8.2, 9.1, 7.8, 9.3, 8.4, 8.6],
    'interactiveness': ['yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'yes'],
    'practical_knowledge': ['verygood', 'good', 'average', 'average', 'good', 'good', 'good', 'verygood', 'good', 'average'],
    'communication': ['good', 'moderate', 'poor', 'good', 'moderate', 'moderate', 'poor', 'good', 'good', 'good'],
    'job_offer': ['yes', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes']
})

# Encode categorical features
label_encoders = {}
for column in ['interactiveness', 'practical_knowledge', 'communication', 'job_offer']:
    le = LabelEncoder()
    sample_data[column] = le.fit_transform(sample_data[column])
    label_encoders[column] = le

X_sample = sample_data.drop('job_offer', axis=1)
y_sample = sample_data['job_offer']`
      },
      {
        title: "3. Visualize Decision Tree",
        explanation: "We train a new model on the custom data and visualize the decision tree structure to understand the splitting logic.",
        code: `# Step 3 & 4: Train and Plot
sample_model = DecisionTreeClassifier(max_depth=4, random_state=42)
sample_model.fit(X_sample, y_sample)

plt.figure(figsize=(12, 6))
plot_tree(sample_model, feature_names=X_sample.columns,
          class_names=label_encoders['job_offer'].classes_,
          filled=True, rounded=True)
plt.title("Decision Tree for Job Offer Prediction")
plt.show()`
      },
      {
        title: "4. Predict New Sample",
        explanation: "We define a new candidate profile, encode it using the same encoders, and predict whether they will get a job offer.",
        code: `# Step 5: Test the model with a new sample input
test_sample = pd.DataFrame([{
    'cgpa': 6.5,
    'interactiveness': 'yes',
    'practical_knowledge': 'good',
    'communication': 'good'
}])

for column in ['interactiveness', 'practical_knowledge', 'communication']:
    test_sample[column] = label_encoders[column].transform(test_sample[column])

prediction = sample_model.predict(test_sample)
predicted_label = label_encoders['job_offer'].inverse_transform(prediction)
print("Predicted Job Offer for test sample:", predicted_label[0])`
      }
    ]
  },
  "lab9": {
    title: "Naive Bayesian Classifier",
    aim: "Develop a program to implement the Naive Bayesian classifier considering Olivetti Face Data set for training. Compute the accuracy of the classifier and visualize predictions.",
    dataset: "Olivetti Faces (Built-in)",
    datasetUrl: null,
    steps: [
      {
        title: "1. Load Olivetti Faces",
        explanation: "We fetch the Olivetti faces dataset (400 images of 40 distinct subjects) and split it into training and testing sets (70/30 split).",
        code: `import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load Olivetti Faces dataset
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)`
      },
      {
        title: "2. Train GaussianNB",
        explanation: "We initialize and train the Gaussian Naive Bayes classifier. We then predict labels for the test set and calculate accuracy.",
        code: `gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("\\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))`
      },
      {
        title: "3. Visualize Predictions",
        explanation: "We plot a grid of test images, displaying both the True Label (actual person) and Predicted Label to visually assess model performance.",
        code: `fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for ax, image, label, prediction in zip(axes.ravel(), X_test, y_test, y_pred):
    ax.imshow(image.reshape(64, 64), cmap=plt.cm.gray)
    ax.set_title(f"True: {label}    Pred: {prediction}")
    ax.axis('off')

plt.tight_layout()
plt.show()`
      }
    ]
  },
  "lab10": {
    title: "k-Means Clustering",
    aim: "Develop a program to implement k-means clustering using Wisconsin Breast Cancer data set and visualize the clustering result.",
    dataset: "Breast Cancer (Built-in)",
    datasetUrl: null,
    steps: [
      {
        title: "1. Load & Scale Data",
        explanation: "We load the Wisconsin Breast Cancer dataset and standardize the features using StandardScaler, which is crucial for distance-based algorithms like K-Means.",
        code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)`
      },
      {
        title: "2. Apply K-Means & PCA",
        explanation: "We perform K-Means clustering with 2 clusters. We also use PCA to reduce the 30 dimensions down to 2 principal components for visualization purposes.",
        code: `kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df['Cluster'] = y_kmeans
df['True Label'] = y`
      },
      {
        title: "3. Visualize Clusters",
        explanation: "We plot the data points in the 2D PCA space, coloring them by their K-Means cluster assignment.",
        code: `plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=100, edgecolor='black', alpha=0.7)
plt.title('K-Means Clustering of Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title="Cluster")
plt.grid(True)
plt.show()`
      },
      {
        title: "4. Visualize Centroids",
        explanation: "We overlay the Cluster Centroids (marked with red X) on the plot to see the center of each identified group.",
        code: `plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=100, edgecolor='black', alpha=0.7)

# Transform centroids back to PCA space for plotting
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', label='Centroids')

plt.title('K-Means Clustering with Centroids')
plt.legend(title="Cluster")
plt.grid(True)
plt.show()`
      }
    ]
  }
};