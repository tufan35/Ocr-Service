import os
import json
from PIL import Image
import pytesseract
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Resim ve JSON dosyalarının bulunduğu klasör
data_folder = "Data_Object"

# Tüm resim ve etiket dosyalarının isimlerini listeleme
imageFiles = [f for f in os.listdir(data_folder) if f.endswith(".jpg")]
jsonFiles = [f for f in os.listdir(data_folder) if f.endswith(".json")]

imageFile = []
ticketObject = []

# OCR ile metin çıkartma
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text.strip()

# Tüm JSON dosyalarını okuma ve resimlerle eşleştirme

for json_file in jsonFiles:
    with open(os.path.join(data_folder, json_file), "r", encoding="utf-8") as file:
        json_data = json.load(file)
        print(json_data);
        # JSON dosyasındaki etiketleri al
        etiket = json_data["label"]
        
        # Eşleşen resim dosyasını bul
        corresponding_image = next((img for img in image_files if img.startswith(json_file[:-5])), None)
        
        if corresponding_image:
            # Resim dosya yolu ve etiketi listelere ekle
            imageFile.append(os.path.join(data_folder, corresponding_image))
            ticketObject.append(etiket)

# OCR ile metin çıkartma
fatura_metinleri = [extract_text_from_image(image) for image in imageFile]

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(fatura_metinleri, ticketObject, test_size=0.2, random_state=42)

# Metin verilerini sayısal vektörlere dönüştürme
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Karar ağacı sınıflandırıcı modeli oluşturma
model = DecisionTreeClassifier()
model.fit(X_train_vectorized, y_train)

# Test seti üzerinde modelin performansını değerlendirme
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Belirli bir resim için fatura tipini tahmin etme
new_invoice_image_path = "yeni_fatura.jpg"  # Değiştirilmesi gereken
new_invoice_text = extract_text_from_image(new_invoice_image_path)
new_invoice_vectorized = vectorizer.transform([new_invoice_text])
predicted_label = model.predict(new_invoice_vectorized)[0]
print("Tahmin Edilen Fatura Tipi:", predicted_label)
