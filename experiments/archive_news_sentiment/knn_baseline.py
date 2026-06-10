"""KNN 분류 진단: 헤드라인 -> D+1 가격방향(up/down)
TF-IDF 임베딩 + KNN, 시간순 분할 (최근 15%를 테스트셋)
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('output/finbert_training_data.csv').sort_values('date').reset_index(drop=True)
n = len(df)
split = int(n * 0.85)
train, test = df.iloc[:split], df.iloc[split:]
print(f"train={len(train)} ({train['date'].min()}~{train['date'].max()}), "
      f"test={len(test)} ({test['date'].min()}~{test['date'].max()})")
print("test label dist:", test['label'].value_counts().to_dict())

vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english')
X_train = vec.fit_transform(train['title'])
X_test = vec.transform(test['title'])
y_train, y_test = train['label'], test['label']

majority = y_train.value_counts().idxmax()
maj_acc = (y_test == majority).mean()
print(f"\nMajority baseline ({majority}): {maj_acc:.3f}")

for k in [3, 5, 9, 15, 25]:
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"KNN k={k:>2}: acc={acc:.3f}")
