import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca dataset dari file CSV
def load_data(filepath):
    df = pd.read_csv(filepath)
    if 'text' not in df.columns or 'labels' not in df.columns:
        raise ValueError("File CSV harus memiliki kolom 'text' dan 'labels'")
    return df['text'], df['labels']

# Path ke file CSV (ganti dengan path file Anda)
filepath = 'Pembacaan Tema/datasheet artikel/bbc_text_cls.csv'
texts, labels = load_data(filepath)

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Membuat pipeline untuk vektorisasi dan klasifikasi
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Melatih model dengan data latih
model.fit(X_train, y_train)

# Evaluasi model pada data uji
y_pred = model.predict(X_test)
accuracy_train = model.score(X_train, y_train)  # Mengukur akurasi pada data latih
accuracy_test = model.score(X_test, y_test)    # Mengukur akurasi pada data uji
accuracy = accuracy_score(y_test, y_pred)

print(f"Akurasi pada data latih: {accuracy_train * 100:.2f}%")
print(f"Akurasi pada data uji: {accuracy_test * 100:.2f}%")
print(f"Akurasi prediksi model: {accuracy * 100:.2f}%")

# Membuat plot untuk keakurasi model
plt.figure(figsize=(8, 5))
plt.plot(['Training', 'Testing'], [accuracy_train, accuracy_test], marker='o', color='g', linewidth=2)
plt.title('Keakurasi Model (Training vs Testing)')
plt.xlabel('Dataset')
plt.ylabel('Akurasi')
plt.grid(True)
plt.show()

# Fungsi untuk memprediksi tema artikel
def prediksi_tema(artikel):
    tema = model.predict([artikel])
    return tema[0]

# Contoh penggunaan
artikel_baru = "Scientists are developing a new method to store energy efficiently."
tema = prediksi_tema(artikel_baru)
print(f"Tema artikel: {tema}")

# Plot confusion matrix untuk melihat performa model
conf_matrix = confusion_matrix(y_test, y_pred, labels=model.named_steps['multinomialnb'].classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
           xticklabels=model.named_steps['multinomialnb'].classes_,
           yticklabels=model.named_steps['multinomialnb'].classes_)
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Label Asli')
plt.show()
