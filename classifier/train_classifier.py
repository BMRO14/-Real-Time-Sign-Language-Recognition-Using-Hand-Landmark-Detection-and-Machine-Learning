import pickle
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DATA_PICKLE_PATH = 'C:/Users/parav/Desktop/NN_Project/data'

data_dict = pickle.load(open(DATA_PICKLE_PATH, "rb"))

data_raw = data_dict["data"]
labels_raw = data_dict["labels"]

lengths = [len(s) for s in data_raw]
length_counts = Counter(lengths)
print("Feature lengths and counts:", length_counts)

target_len = length_counts.most_common(1)[0][0]
print("Using feature length:", target_len)

data_filtered = []
labels_filtered = []
for s, lbl in zip(data_raw, labels_raw):
    if len(s) == target_len:
        data_filtered.append(s)
        labels_filtered.append(lbl)

data = np.asarray(data_filtered, dtype=float)
labels = np.asarray(labels_filtered)

print("Final data shape:", data.shape)
print("Number of labels:", len(labels))

x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels,
    random_state=42,
)

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f"{score * 100:.2f}% of samples were classified correctly!")

classes = np.unique(np.concatenate([y_test, y_predict]))
cm = confusion_matrix(y_test, y_predict, labels=classes) 

print("Confusion matrix (rows=true, cols=pred):\n", cm)  

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(
    ax=ax,
    cmap="Blues",
    include_values=True,
    values_format="d",
    xticks_rotation="vertical",
    text_kw={"fontsize": 8},
)
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.show()  

col_sums = cm.sum(axis=0)
per_class_precision = np.divide(
    np.diag(cm),
    col_sums,
    out=np.zeros_like(col_sums, dtype=float),
    where=col_sums != 0,
)

plt.figure(figsize=(10, 6))
plt.barh([str(c) for c in classes], per_class_precision)  
plt.xlim(0, 1.0)
plt.xlabel("Per-class accuracy (precision = correct-in-predicted / total-predicted)")
plt.ylabel("Class")
plt.title("Per-class Precision")
plt.tight_layout()
plt.show()

with open("model.p", "wb") as f:
    pickle.dump({"model": model}, f)

print("Model saved to model.p")
