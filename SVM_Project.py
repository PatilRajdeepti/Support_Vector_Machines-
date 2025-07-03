import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score

# Step 1: Load the dataset
data = load_breast_cancer()
X_full = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Step 2: Select 2 features for 2D visualization
X = X_full[["mean radius", "mean texture"]]

# Step 3: Train SVM with linear kernel
svc_linear = SVC(kernel='linear', C=1.0)
svc_linear.fit(X, y)

# Step 4: Train SVM with RBF kernel
svc_rbf = SVC(kernel='rbf', gamma=0.1, C=1.0)
svc_rbf.fit(X, y)

# Step 5: Function to plot decision boundary
def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title(title)
    plt.show()

# Step 6: Visualize decision boundaries
plot_decision_boundary(svc_linear, X, y, "SVM with Linear Kernel")
plot_decision_boundary(svc_rbf, X, y, "SVM with RBF Kernel")

# Step 7: Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X, y)

print("🔍 Best Parameters:", grid.best_params_)
print("✅ Best Cross-Validation Score:", grid.best_score_)

# Step 8: Evaluate with cross-validation
best_model = grid.best_estimator_
scores = cross_val_score(best_model, X, y, cv=5)

print("📊 Cross-validation scores:", scores)
print("📈 Average accuracy:", scores.mean())
