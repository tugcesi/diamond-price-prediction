import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# PAGE CONFIGURATION
st.set_page_config(
    page_title="💎 Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .result-price {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .result-label {
        font-size: 1.1em;
        opacity: 0.9;
    }
    
    .sidebar-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 15px;
        text-align: center;
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        font-size: 1.1em;
        width: 100%;
        height: 50px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    """Model'i yükle"""
    try:
        model = joblib.load('diamond_model.joblib')
        return model
    except FileNotFoundError:
        st.error("❌ diamond_model.joblib bulunamadı!")
        st.info("Lütfen convert_pkl_to_joblib.py çalıştır")
        return None
    except Exception as e:
        st.error(f"❌ Model yükleme hatası: {str(e)}")
        return None

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_price_range_interpretation(predicted_price):
    """Fiyat aralığı yorumlama"""
    if predicted_price < 500:
        return "💎 Bütçe Dostu", "Ekonomik segment elmas"
    elif predicted_price < 2000:
        return "✨ İyi Fiyat", "Orta segment elmas"
    elif predicted_price < 5000:
        return "👑 Premium", "Yüksek kalite elmas"
    else:
        return "💰 Lüks", "Çok yüksek fiyatlı elmas"

def get_quality_interpretation(color):
    """Renk kalitesi yorumlama"""
    color_rank = {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1}
    rank = color_rank.get(color, 0)
    
    if rank >= 6:
        return "Mükemmel Renk 🌟"
    elif rank >= 4:
        return "İyi Renk ⭐"
    else:
        return "Orta Renk 👍"

def predict_price(model, carat, color, table, x, y, z, cut_encoded, clarity_encoded):
    """Fiyat tahmini yap"""
    try:
        # Model'in beklediği feature'lar
        # ['carat', 'color', 'table', 'x', 'y', 'z', 'quality_score', 'volume']
        
        # color_encoded'i oluştur (renk sayısal değer)
        color_mapping = {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1}
        color_encoded = color_mapping.get(color, 0)
        
        # quality_score hesapla
        quality_score = (cut_encoded + color_encoded + clarity_encoded) / 3
        
        # volume hesapla
        volume = x * y * z
        
        # DataFrame oluştur (model'in beklediği sırada)
        df = pd.DataFrame([[
            carat,
            color_encoded,  # color sayısal
            table,
            x,
            y,
            z,
            quality_score,
            volume
        ]], columns=['carat', 'color', 'table', 'x', 'y', 'z', 'quality_score', 'volume'])
        
        # Tahmin yap
        prediction = model.predict(df)
        
        # Array'den scalar çıkar
        if isinstance(prediction, np.ndarray):
            prediction = prediction.item()  # numpy array'den Python scalar'a
        else:
            prediction = float(prediction)
        
        prediction = max(prediction, 0)
        
        return prediction, volume, quality_score
        
    except Exception as e:
        st.error(f"❌ Tahmin hatası: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>💎 Diamond Price Predictor</h1>
        <p>🔮 Elmas özelliklerini girerek fiyat tahmini yapın</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model yükle
    model = load_model()
    if model is None:
        st.stop()
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-title">📊 Elmas Özelliklerini Girin</div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Temel özellikler
        col1, col2 = st.columns(2)
        with col1:
            carat = st.slider("⚖️ Carat (Ağırlık)", 0.2, 5.0, 1.0, 0.01,
                            help="1 karat = 200mg")
        with col2:
            cut = st.selectbox("✂️ Cut (Kesim)", 
                             ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
                             help="Kesim kalitesi")
        
        col1, col2 = st.columns(2)
        with col1:
            color = st.selectbox("🎨 Color (Renk)", 
                               ['D', 'E', 'F', 'G', 'H', 'I', 'J'],
                               help="D=Mükemmel, J=En kötü")
        with col2:
            clarity = st.selectbox("💫 Clarity (Berraklık)", 
                                  ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'],
                                  help="IF=Mükemmel, I1=En kötü")
        
        st.divider()
        
        # Yüzde değerleri
        col1, col2 = st.columns(2)
        with col1:
            depth = st.slider("📏 Depth (%)", 40.0, 80.0, 62.0, 0.1,
                            help="Derinlik yüzdesi")
        with col2:
            table = st.slider("📐 Table (%)", 40.0, 95.0, 57.0, 0.1,
                            help="Tablo yüzdesi")
        
        st.divider()
        
        # Fiziksel boyutlar
        st.markdown("**📐 Fiziksel Boyutlar (mm)**")
        col1, col2, col3 = st.columns(3)
        with col1:
            x = st.number_input("X (Uzunluk)", 0.0, 15.0, 4.0, 0.01)
        with col2:
            y = st.number_input("Y (Genişlik)", 0.0, 15.0, 4.0, 0.01)
        with col3:
            z = st.number_input("Z (Derinlik)", 0.0, 10.0, 2.5, 0.01)
        
        # Mappings
        cut_mapping = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
        clarity_mapping = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}
        
        st.divider()
        predict_button = st.button("🔮 Fiyat Tahminini Yap", use_container_width=True)
    
    # MAIN AREA
    if predict_button:
        with st.spinner('⏳ Tahmin yapılıyor...'):
            predicted_price, volume, quality_score = predict_price(
                model,
                carat=carat,
                color=color,
                table=table,
                x=x,
                y=y,
                z=z,
                cut_encoded=cut_mapping[cut],
                clarity_encoded=clarity_mapping[clarity]
            )
            
            if predicted_price is not None:
                price_category, price_desc = get_price_range_interpretation(predicted_price)
                color_quality = get_quality_interpretation(color)
                
                # Sonuç kutusu
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">💰 Tahmini Fiyat</div>
                    <div class="result-price">${predicted_price:,.2f}</div>
                    <div class="result-label">{price_category} - {price_desc}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("✓ Tahmin başarıyla tamamlandı!")
                
                st.divider()
                
                # Metrikleri göster
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("⚖️ Ağırlık", f"{carat} ct")
                with col2:
                    st.metric("✂️ Kesim", cut)
                with col3:
                    st.metric("🎨 Renk", f"{color} - {color_quality}")
                with col4:
                    st.metric("💫 Berraklık", clarity)
                
                st.divider()
                
                # Detaylı Bilgiler
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Temel Özellikler")
                    features_df = pd.DataFrame({
                        'Özellik': ['Ağırlık (Carat)', 'Kesim', 'Renk', 'Berraklık', 'Kalite Skoru'],
                        'Değer': [f"{carat} ct", cut, color, clarity, f"{quality_score:.2f}"]
                    })
                    st.table(features_df)
                
                with col2:
                    st.subheader("📐 Fiziksel Bilgiler")
                    physical_df = pd.DataFrame({
                        'Ölçü': ['Uzunluk (X)', 'Genişlik (Y)', 'Derinlik (Z)', 'Derinlik %', 'Tablo %', 'Hacim'],
                        'Değer': [f"{x} mm", f"{y} mm", f"{z} mm", f"{depth:.1f}%", f"{table:.1f}%", f"{volume:.2f} mm³"]
                    })
                    st.table(physical_df)
                
                st.divider()
                
                # Gauge Chart
                st.subheader("📊 Fiyat Aralığı")
                
                fig = go.Figure(data=[go.Indicator(
                    mode="gauge+number",
                    value=predicted_price,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Tahmini Fiyat ($)"},
                    gauge={
                        'axis': {'range': [0, max(10000, predicted_price * 1.2)]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 500], 'color': "#e8f4f8"},
                            {'range': [500, 2000], 'color': "#d0e8f2"},
                            {'range': [2000, 5000], 'color': "#b8dce8"},
                            {'range': [5000, max(10000, predicted_price * 1.2)], 'color': "#a0d0de"}
                        ]
                    }
                )])
                
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # Öneriler
                st.subheader("💡 Öneriler")
                
                tips = []
                
                if carat < 0.5:
                    tips.append("💎 Düşük Karat - Bütçe dostu seçim")
                elif carat < 1:
                    tips.append("💎 Standart Karat - Popüler seçim")
                else:
                    tips.append("💎 Yüksek Karat - Premium seçim")
                
                if cut == 'Ideal':
                    tips.append("✂️ İdeal Kesim - En iyi ışın dağılımı")
                elif cut == 'Premium':
                    tips.append("✂️ Premium Kesim - Çok iyi görünüm")
                
                if color in ['D', 'E', 'F']:
                    tips.append("🎨 Üstün Renk - Mükemmel seçim")
                elif color in ['H', 'I', 'J']:
                    tips.append("🎨 Tasarımla Gizlenebilen Renk")
                
                if clarity in ['IF', 'VVS1', 'VVS2']:
                    tips.append("💫 Mükemmel Berraklık - Hiç kusur yok")
                elif clarity in ['VS1', 'VS2']:
                    tips.append("💫 İyi Berraklık - İyi kalite-fiyat dengesi")
                
                for tip in tips:
                    st.info(tip)

if __name__ == "__main__":
    main()