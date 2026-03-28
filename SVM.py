from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
cancer = load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Kernels
kernels = {
    "Linear": SVC(kernel="linear", C=1),
    "Polynomial": SVC(kernel="poly", degree=3, C=1),
    "RBF": SVC(kernel="rbf", gamma="scale", C=1),
    "Sigmoid": SVC(kernel="sigmoid", C=1)
}

# Create subplots properly
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()  # flatten

for i, (name, model) in enumerate(kernels.items()):
    
    ax = axes[i]   #  assign axis
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Decision boundary (IMPORTANT: pass ax)
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        response_method="predict",
        alpha=0.5,
        ax=ax   #  fix alignment issue
    )
    
    # Scatter plot
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=20)
    
    # Title
    ax.set_title(f"{name} Kernel\nAccuracy = {acc:.3f}")

# Adjust layout
plt.tight_layout()
plt.show()
