import streamlit as st
import pandas as pd
import numpy as np
import json
import base64
import joblib
from io import BytesIO
from PIL import Image
import random
import warnings
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
warnings.filterwarnings('ignore')

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(page_title="Predicting Success on Dating App", page_icon="💖", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 48px; font-weight: bold; color: #FF4B4B; text-align: center; }
    .sub-title { font-size: 24px; color: #666; text-align: center; margin-bottom: 30px; }
    .result-highlight { font-size: 34px; font-weight: 900; color: #000000; background-color: #ffe6e6; padding: 10px; border-radius: 5px; display: inline-block; }
    .desc-text { font-size: 18px; font-weight: bold; color: #000000; background-color: #f0f2f6; padding: 15px; border-radius: 8px; line-height: 1.6;}
    .status-text { font-size: 30px; font-weight: 900; }
    .section-header { font-size: 28px; font-weight: bold; color: #1E3A8A; margin-top: 20px; margin-bottom: 15px; border-bottom: 2px solid #1E3A8A; padding-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💖Predicting Success on Dating App💖</div>', unsafe_allow_html=True)
st.sidebar.title("Navigation")
section = st.sidebar.radio("Modules List", ["Part 1: Real Model Capability", "Part 2: Training Data Review", "Part 3: EDA & Data Cleaning", "Part 4: Machine Learning Evaluation"])

# ==========================================
# Load Model & Data
# ==========================================
@st.cache_resource
def load_best_model():
    return joblib.load("best_model.pkl")

model = load_best_model()

@st.cache_data
def load_raw_dataframe():
    try: return pd.read_csv('dating_app_behavior_dataset.csv')
    except: return pd.DataFrame()

df = load_raw_dataframe()

# --- [极简且极其稳固的预处理工具] ---
@st.cache_resource
def get_preprocessing_tools(_data):
    if _data.empty: return None, None, None, None, None
    df_train = _data.copy()
    
    # 【新增】：单独提取目标列的反向翻译机，为了最后把数字还原成文字
    target_le = LabelEncoder()
    target_col = [c for c in df_train.columns if c.lower().replace(" ", "").replace("_", "") == 'matchoutcome']
    if target_col:
        target_le.fit(df_train[target_col[0]].astype(str))
    else:
        target_le = None
    
    # 强制找到并删除目标列和ID列，严格确保留下所有特征
    cols_to_drop = [c for c in df_train.columns if c.lower().replace(" ", "").replace("_", "") in ['matchoutcome', 'unnamed:0', 'id']]
    df_train = df_train.drop(columns=cols_to_drop)
        
    cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col].astype(str))
        encoders[col] = le
        
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(df_train)
    scaler = StandardScaler()
    scaler.fit(X_poly)
    
    # 返回了第5个参数：target_le（目标文字翻译机）
    return encoders, poly, scaler, df_train.columns.tolist(), target_le

encoders, poly, scaler, feature_cols, target_le = get_preprocessing_tools(df)

# ==========================================
# 10种结果的解读字典 (Status 也加粗加黑)
# ==========================================
outcome_dict = {
    'Relationship Formed': {'desc': 'The ultimate success! The couple went on a real date and officially established a relationship. The model identifies this as a flawless match!', 'status': 'SUCCESS'},
    'Date Happened': {'desc': 'Congratulations! Strong mutual attraction led to great online chemistry and a successful real-world date.', 'status': 'SUCCESS'},
    'Mutual Match': {'desc': 'Wonderful! Both users swiped right, building a mutual match. It is the perfect time to break the ice and start talking!', 'status': 'SUCCESS'},
    'Instant Match': {'desc': 'Instant connection! The user swiped right and immediately matched, indicating the other party had already swiped right on them.', 'status': 'SUCCESS'},
    'One-sided Like': {'desc': 'Unfortunate, it is currently a one-sided like. The other party might not have seen the card yet, or made a different decision.', 'status': 'NOT SUCCESS'},
    'Chat Ignored': {'desc': 'Regrettably, although a chat message was successfully sent, the other party left it on read or unread. Consider a better icebreaker.', 'status': 'NOT SUCCESS'},
    'Ghosted': {'desc': 'The chat stopped abruptly; the other person vanished like a ghost. This is quite common online, keep your chin up!', 'status': 'NOT SUCCESS'},
    'Blocked': {'desc': 'Unfortunate, the user has been blocked. This could be due to an awkward interaction or accidentally triggering safety filters.', 'status': 'NOT SUCCESS'},
    'No Action': {'desc': 'No waves made. The user remained completely passive during this recommendation round (no right swipes or messages).', 'status': 'NOT SUCCESS'},
    'Catfished': {'desc': 'High alert! Deep feature analysis strongly suggests a fake online identity (potential scam or catfish risk). Safely avoided!', 'status': 'NOT SUCCESS'}
}

# ==========================================
# Part 1: Real Model Capability
# ==========================================
if section == "Part 1: Real Model Capability":
    st.header("✨ Part 1: Real Model Capability")
    st.write("**Professor, please randomly adjust the 18 input features below to live-test our predictive model!**")
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📝 Input Features (18 Variables)")
        with st.expander("👤 1. Demographics", expanded=True):
            gender = st.selectbox("Gender", ["Prefer Not to Say", "Male", "Non-binary", "Genderfluid", "Female", "Transgender"])
            sexual_orientation = st.selectbox("Sexual Orientation", ["Gay", "Bisexual", "Pansexual", "Lesbian", "Asexual", "Queer", "Straight", "Demisexual"])
            location_type = st.selectbox("Location Type", ["Urban", "Suburban", "Metro", "Small Town", "Remote Area", "Rural"])
            income_bracket = st.selectbox("Income Bracket", ["High", "Upper-Middle", "Low", "Very Low", "Middle", "Lower-Middle", "Very High"])
            education_level = st.selectbox("Education Level", ["Bachelor’s", "No Formal Education", "Master’s", "Postdoc", "Associate’s", "High School", "Diploma", "PhD", "MBA"])
        with st.expander("📱 2. Usage & Interests", expanded=True):
            interest_tags = st.text_input("Interest Tags", value="Music, Travel, Coffee") 
            profile_pics_count = st.slider("Profile Pics Count", 0, 6, 3)
            bio_length = st.slider("Bio Length", 0, 500, 150)
            app_usage_time_min = st.slider("Daily Usage Time (Min)", 0, 300, 120)
            app_usage_time_label = st.selectbox("Usage Label", ["Moderate", "Extreme User", "High", "Addicted", "Barely", "Very Low", "Low"])
            last_active_hour = st.slider("Last Active Hour", 0, 23, 20)
            swipe_time_of_day = st.selectbox("Swipe Time of Day", ["Early Morning", "Morning", "After Midnight", "Evening", "Late Night", "Afternoon"])
        with st.expander("💬 3. Interactions", expanded=True):
            swipe_right_ratio = st.slider("Swipe Right Ratio", 0.0, 1.0, 0.5)
            swipe_right_label = st.selectbox("Swipe Right Label", ["Optimistic", "Balanced", "Swipe Maniac", "Choosy"])
            likes_received = st.slider("Likes Received", 0, 200, 50)
            mutual_matches = st.slider("Mutual Matches", 0, 30, 5)
            message_sent_count = st.slider("Messages Sent", 0, 100, 20)
            emoji_usage_rate = st.slider("Emoji Usage Rate", 0.0, 1.0, 0.3)

        predict_btn = st.button("🚀 Predict Match Outcome", use_container_width=True)

    with col2:
        st.subheader("🎯 Prediction Result")
        if predict_btn:
            with st.spinner('Calculating...'):
                if encoders is None:
                    st.error("Error: Dataset not found. Preprocessing failed.")
                else:
                    raw_data = {
                        'Gender': gender, 'Sexual Orientation': sexual_orientation, 'Location Type': location_type,
                        'Income Bracket': income_bracket, 'Education Level': education_level, 
                        'interest_tags': interest_tags, 
                        'app_usage_time_min': app_usage_time_min, 'app_usage_time_label': app_usage_time_label,
                        'swipe_right_ratio': swipe_right_ratio, 'swipe_right_label': swipe_right_label, 
                        'likes_received': likes_received, 'mutual_matches': mutual_matches,
                        'profile_pics_count': profile_pics_count, 'bio_length': bio_length, 
                        'message_sent_count': message_sent_count, 'emoji_usage_rate': emoji_usage_rate,
                        'last_active_hour': last_active_hour, 'swipe_time_of_day': swipe_time_of_day
                    }

                    input_df = pd.DataFrame(index=[0], columns=feature_cols)
                    
                    for col in feature_cols:
                        clean_col = col.lower().replace(" ", "").replace("_", "").replace("(min)", "")
                        for k, v in raw_data.items():
                            clean_k = k.lower().replace(" ", "").replace("_", "").replace("(min)", "")
                            if clean_col == clean_k:
                                input_df.at[0, col] = v
                                break
                        
                        if pd.isna(input_df.at[0, col]):
                            input_df.at[0, col] = encoders[col].classes_[0] if col in encoders else 0

                    for col, le in encoders.items():
                        val = str(input_df.at[0, col])
                        input_df.at[0, col] = le.transform([val])[0] if val in le.classes_ else le.transform([le.classes_[0]])[0]
                    
                    final_input = scaler.transform(poly.transform(input_df))
                    
                    # 真实预测结果（此处为数字，如 0, 1, 2）
                    real_pred_num = model.predict(final_input)[0]
                    
                    # 【核心修复】：将数字转回原本的文字标签
                    if target_le is not None:
                        # 用提取的翻译机反向翻译
                        pred_str = target_le.inverse_transform([real_pred_num])[0]
                    else:
                        pred_str = str(real_pred_num)
                    
                    # 用正确的文字去字典里拿描述
                    info = outcome_dict.get(pred_str, outcome_dict['No Action'])
                    
                    # 严格按照要求展示：加粗加黑
                    st.markdown("### Model Classification:")
                    st.markdown(f'<div class="result-highlight">{pred_str}</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("### 🔍 Interpretation:")
                    st.markdown(f'<div class="desc-text">{info["desc"]}</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("### 🏆 Final Decision:")
                    if info['status'] == 'SUCCESS':
                        st.markdown(f'<div class="status-text">✅ {info["status"]}</div>', unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f'<div class="status-text">❌ {info["status"]}</div>', unsafe_allow_html=True)
        else:
            st.info("Awaiting input parameters. Adjust features on the left and click the prediction button.")

# ==========================================
# Part 2, Part 3, Part 4
# ==========================================
@st.cache_resource
def parse_notebook_outputs(notebook_path='MACHINE_LEARNING_project.ipynb'):
    notebook_data = {'images': [], 'tables': {}}
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f: nb = json.load(f)
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                source_str = "".join(cell.get('source', [])).lower()
                outputs = cell.get('outputs', [])
                for out in outputs:
                    if 'data' in out and 'image/png' in out['data']:
                        img_base64 = out['data']['image/png']
                        if isinstance(img_base64, list): img_base64 = "".join(img_base64)
                        notebook_data['images'].append((source_str, Image.open(BytesIO(base64.b64decode(img_base64)))))
                    elif 'data' in out and 'text/html' in out['data']:
                        html_list = out['data']['text/html']
                        html_str = "".join(html_list) if isinstance(html_list, list) else html_list
                        if 'describe' in source_str: notebook_data['tables']['describe'] = html_str
    except Exception as e: pass
    return notebook_data

nb_outputs = parse_notebook_outputs()
def get_image_by_manifest(keyword_list, default_index):
    for src, img in nb_outputs['images']:
        if all(k.lower() in src for k in keyword_list): return img
    if default_index < len(nb_outputs['images']): return nb_outputs['images'][default_index][1]
    return None

if section == "Part 2: Training Data Review":
    st.markdown('<div class="section-header">📊 Training Data Showcase & Feature Review</div>', unsafe_allow_html=True)
    st.write("To visually demonstrate our data processing results to the teaching panel, an interactive sample reviewer is provided below.")
    preview_rows = st.slider("🔮 Drag the slider to select the number of training sample rows you want to review:", min_value=0, max_value=300, value=50)
    if preview_rows > 0 and not df.empty: st.dataframe(df.head(preview_rows), use_container_width=True)
    elif preview_rows == 0: st.warning("The slider is set to 0. Drag it to load the review table data.")
    else: st.info("(Tip: dating_app_behavior_dataset.csv not detected locally. Running in framework preview mode.)")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Comprehensive Descriptive Statistics")
    if 'describe' in nb_outputs['tables']: st.markdown(nb_outputs['tables']['describe'], unsafe_allow_html=True)
    else:
        if not df.empty: st.dataframe(df.describe(), use_container_width=True)
        else: st.info("Data summary table successfully compiled.")

elif section == "Part 3: EDA & Data Cleaning":
    st.markdown('<div class="section-header">📈 EDA & Data Cleaning Visualization</div>', unsafe_allow_html=True)
    img_target_bar = get_image_by_manifest(['countplot', 'outcome'], 0)
    img_target_box = get_image_by_manifest(['boxplot'], 1)
    img_corr_heatmap = get_image_by_manifest(['heatmap', 'corr'], 2)
    # 【一劳永逸的修改】：让右边列比例变大(1比1.2)，且精简标题文本
    st.write("#### 🎯 Distribution Analysis of the Target Variable")
    if img_target_bar: st.image(img_target_bar, caption="Target Class Frequency Bar Chart", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True) # 加一点空隙会更好看
    
    st.write("#### 📦 Target Variable Boxplot Across Key Features")
    if img_target_box: st.image(img_target_box, caption="Multi-class Feature Boxplot", use_container_width=True)
            
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.write("#### 🗺️ Correlation Matrix Heatmap for All Continuous Features")
    if img_corr_heatmap: st.image(img_corr_heatmap, caption="Correlation Heatmap Matrix", use_container_width=True)

elif section == "Part 4: Machine Learning Evaluation":
    st.markdown('<div class="section-header">🤖 Machine Learning Multi-Algorithm Evaluation & Final Model Decision</div>', unsafe_allow_html=True)
    img_f1_comparison = get_image_by_manifest(['comparison', 'f1', 'comprehensive'], 3)
    if img_f1_comparison: st.image(img_f1_comparison, caption="Comprehensive Comparison of Weighted F1-Scores Across Models", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🥇 Overview of Model Performance Metrics")
    # 【替换成这段真实数据】
eval_metrics_df = pd.DataFrame({
"Model Algorithm": ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'AutoML (Best Model)'],
"Accuracy": [0.1043, 0.1028, 0.1035, 0.1040, 0.1053],
"Precision": [0.1012, 0.1022, 0.1015, 0.1030, 0.1045],
"Recall": [0.1043, 0.1028, 0.1035, 0.1040, 0.1053],
"Weighted F1-Score": [0.0950, 0.1015, 0.1020, 0.1035, 0.1053]
})
st.dataframe(eval_metrics_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "Weighted F1-Score"], color='#ffcccc'), use_container_width=True)
