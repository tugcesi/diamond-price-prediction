# 💎 Diamond Price Prediction

Bu proje, **makine öğrenmesi** ve **derin öğrenme** algoritmaları kullanarak elmas fiyatlarını tahmin eden bir Python uygulamasıdır. Çeşitli ML modelleri karşılaştırılmış ve en iyi performans gösteren model bir Streamlit web uygulamasına entegre edilmiştir.

## 📁 Proje Yapısı

```
diamond-price-prediction/
├── app.py                                          # Streamlit web uygulaması
├── DiomandPricePredictionwithMachineLearning.ipynb # Model eğitimi ve analiz
├── diamond_model.joblib                            # Eğitilmiş ML modeli
├── diamond_bundle.pkl                              # Model bundle (scaler + model)
├── diamond_price.h5                                # Derin öğrenme modeli (Keras)
├── diamonds.csv                                    # Elmas veri seti (53.940 kayıt)
├── requirements.txt                                # Python bağımlılıkları
└── README.md                                       # Proje dokümantasyonu
```

## 🚀 Kurulum

### 1. Repoyu klonlayın
```bash
git clone https://github.com/tugcesi/diamond-price-prediction.git
cd diamond-price-prediction
```

### 2. Bağımlılıkları yükleyin
```bash
pip install -r requirements.txt
```

## ▶️ Kullanım

### Streamlit Uygulaması
```bash
streamlit run app.py
```
Tarayıcınızda `http://localhost:8501` adresine gidin.

### Jupyter Notebook
```bash
jupyter notebook DiomandPricePredictionwithMachineLearning.ipynb
```

## 🛠️ Özellikler

- 💎 Elmas özelliklerini girerek anlık fiyat tahmini
- 🎛️ Carat, Cut, Color, Clarity, Depth, Table ve boyut parametreleri
- 📊 Gauge chart ile görsel fiyat aralığı gösterimi
- 💡 Elmas seçimine yönelik akıllı öneriler
- 📐 Fiziksel boyutlar ve kalite skoru hesaplama

## 📊 Veri Seti

`diamonds.csv` veri seti **53.940 elmas** kaydı içermekte olup aşağıdaki özellikleri kapsamaktadır:

| Özellik | Açıklama |
|---------|----------|
| `carat` | Elmas ağırlığı (karat) |
| `cut` | Kesim kalitesi (Fair → Ideal) |
| `color` | Renk kalitesi (D=En iyi → J=En kötü) |
| `clarity` | Berraklık (IF=En iyi → I1=En kötü) |
| `depth` | Derinlik yüzdesi |
| `table` | Tablo yüzdesi |
| `x`, `y`, `z` | Fiziksel boyutlar (mm) |
| `price` | Fiyat (USD) - Hedef değişken |

## 🤖 Kullanılan Modeller

- ✅ **Random Forest Regressor** (Ana model - `diamond_model.joblib`)
- ✅ **Gradient Boosting / XGBoost**
- ✅ **Linear Regression**
- ✅ **Deep Learning - Neural Network** (`diamond_price.h5`)

## 📦 Gereksinimler

- Python 3.8+
- scikit-learn
- TensorFlow / Keras
- Streamlit
- Pandas
- NumPy
- Plotly
- Joblib

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.