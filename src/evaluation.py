import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from anomaly_detector import AnomalyDetector

# -- Загрузка данных --
df = pd.read_csv("data/raw/SWAT_Dataset.csv")
features = df.drop(columns=["Normal/Attack"])
labels = (df["Normal/Attack"] == "Attack").astype(int).values 

X_normal = features[labels == 0].values
X_attack = features[labels == 1].values

# -- Train/Val split нормальных данных --
# Часть нормальных идёт в обучение, часть в тест вместе с аномалиями
X_train, X_val_normal = train_test_split(X_normal, test_size=0.2, random_state=53)

# Тестовые данные = нормальные + аномалии
X_test = np.vstack([X_val_normal, X_attack])  # type: ignore
y_test = np.hstack([np.zeros(len(X_val_normal)), np.ones(len(X_attack))])

# -- Обучаем детектор --
detector = AnomalyDetector(n_components=1)
detector.fit(X_train)

# -- Anomaly scores --
scores = detector.anomaly_score(X_test)
preds = detector.predict(X_test)

# -- ROC-кривая --
fpr, tpr, _ = roc_curve(y_test, scores)
auc = roc_auc_score(y_test, scores)

# -- Графики --
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Левый: ROC
ax = axes[0]
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"PCA Detector (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
ax.fill_between(fpr, tpr, alpha=0.1, color="darkorange")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Anomaly Detection")
ax.legend()
ax.grid(True, alpha=0.3)

# Правый: распределение anomaly scores
ax = axes[1]
ax.hist(scores[y_test == 0], bins=40, alpha=0.6, color="steelblue", label="Normal")
ax.hist(scores[y_test == 1], bins=40, alpha=0.6, color="red", label="Attack")
ax.axvline(
    detector.threshold,
    color="black",
    linestyle="--",
    label=f"Threshold = {detector.threshold:.4f}",
)
ax.set_xlabel("Reconstruction Error (anomaly score)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Anomaly Scores")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/processed/roc_curve.png", dpi=150)
plt.show()

# ── Метрики ──
print(f"ROC-AUC: {auc:.4f}")
print(f"Threshold: {detector.threshold:.6f}")
print()
print(classification_report(y_test, preds, target_names=["Normal", "Attack"]))
