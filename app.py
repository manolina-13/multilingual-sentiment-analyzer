import logging
import os
import streamlit as st
import streamlit.components.v1 as components
from engine import SentimentEngine


# Suppress Transformers/Streamlit noise
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

# 1. Page Configuration
st.set_page_config(
    page_title="PolySentiment AI - Multilingual Sentiment Analysis", 
    page_icon="🌍", 
    layout="centered"
)
# ✅ ADD HERE (immediately after page config)
st.markdown("""
<meta property="og:title" content="PolySentiment AI">
<meta property="og:description" content="Multilingual Sentiment Analysis with Explainability">
""", unsafe_allow_html=True)
# 2. Custom CSS for Professional UI
st.markdown("""
    <style>
    .main { background-color: #0e1117; } /* Matches Streamlit Dark Theme */
    
    [data-testid="stMetric"] {
        background-color: #1e2129;
        border: 1px solid #31333f;
        padding: 15px;
        border-radius: 10px;
        color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }

    section[data-testid="stSidebar"] {
        background-color: #111b21;
    }

    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar
with st.sidebar:
    st.title("🚀 Project Specs")
    st.markdown("---")
    st.subheader("🛠️ Tech Stack")
    st.code(
        "Python 3.12\nStreamlit\nHugging Face Hub\nXLM-RoBERTa Model\nSHAP (Explainability)",
        language="text"
    )
    
    # st.subheader("💡 Key Features")
    # st.write("✅ **Zero-Shot Multilingual:** Analyzes Hindi & Bengali script directly.")
    # st.write("✅ **XAI Integration:** Uses SHAP to visualize word-level importance.")
    # st.write("✅ **Secure API:** Uses st.secrets for token management.")
    st.subheader("💡 Key Features")
    st.write("✅ **Zero-Shot Multilingual:** Direct sentiment analysis on native scripts—English, Hindi, and Bengali—without any preprocessing or transliteration.")
    st.write("✅ **Native Script Detection:** Automatic Unicode-based script identification (Latin, Devanagari, Bengali) with language classification.")
    st.write("✅ **XLM-RoBERTa Model:** State-of-the-art multilingual transformer for cross-lingual understanding and zero-shot transfer learning.")
    st.write("✅ **SHAP Explainability:** Word-level feature attribution—visualize exactly which words drive the sentiment decision (red=positive, blue=negative).")
    st.write("✅ **Confidence Scoring:** Real-time confidence metrics for each prediction with probabilistic thresholds.")
    st.write("✅ **Technical Execution Log:** Full transparency into model pipeline, script detection, and inference chain.")
    st.write("✅ **Dark Mode Optimized UI:** Professional Streamlit interface with custom CSS for readability and visual clarity.")
    
    st.divider()
    st.caption("Developed by Manolina | Powered by Hugging Face & SHAP")

# 4. Initialize Engine
@st.cache_resource
def load_engine():
    return SentimentEngine()

engine = load_engine()

# 5. Session State
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

def set_text(txt):
    st.session_state.text_input = txt

# 6. Header
st.title("PolySentiment AI - Multilingual Sentiment Analysis")
st.markdown("Analyze sentiment in **English, Bengali, and Hindi** directly using native scripts.")

# 7. Sample Buttons
st.write("Quick Samples:")
c1, c2, c3 = st.columns(3)

with c1:
    st.button("English Sample", on_click=set_text,
              args=("The service was absolutely wonderful!",))
with c2:
    st.button("Bengali Sample", on_click=set_text,
              args=("এই খাবারটি খুব সুস্বাদু ছিল।",))
with c3:
    st.button("Hindi Sample", on_click=set_text,
              args=("मुझे यह फिल्म बहुत पसंद आई।",))

# 8. Main Input
user_input = st.text_area(
    "Enter your native script text:", 
    value=st.session_state.text_input,
    placeholder="Type here...", 
    height=100
)

# 9. Analysis Logic
if st.button("Analyze Sentiment", use_container_width=True):

    if user_input.strip():

        # --- PHASE 1: PREDICTION ---
        with st.spinner("AI analyzing script and sentiment..."):
            result, detected_lang, script_type = engine.analyze(user_input)

            raw_label = result.get('label', 'Error').upper()

            if "POS" in raw_label or "LABEL_2" in raw_label:
                label, icon = "POSITIVE", "😊"
                is_positive = True
            elif "NEG" in raw_label or "LABEL_0" in raw_label:
                label, icon = "NEGATIVE", "😞"
                is_positive = False
            else:
                label, icon = "NEUTRAL", "😐"
                is_positive = False

            score = result.get('score', 0.0)

        st.divider()

        # --- DISPLAY RESULTS ---
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if label == "POSITIVE":
                st.success(f"### Result: {label} {icon}")
            elif label == "NEGATIVE":
                st.error(f"### Result: {label} {icon}")
            else:
                st.warning(f"### Result: {label} {icon}")

        with res_col2:
            st.metric("Confidence Score", f"{score:.2%}")

        # --- TECHNICAL BREAKDOWN ---
        st.write("")
        with st.expander("🛠️ Technical Breakdown (Execution Log)", expanded=True):
            st.markdown(f"""
            - **Input Language:** Detected as **{detected_lang}**.
            - **Script Detection:** System identified **{script_type}** via Unicode Regex.
            - **Processing:** Direct Native Script Inference (Zero-shot).
            - **Model Architecture:** **XLM-RoBERTa (cardiffnlp)**.
            - **Explainability:** Initializing SHAP word-level feature attribution.
            """)

        # --- PHASE 3: EXPLAINABILITY LAYER (SHAP) ---
        
        st.write("---")
        st.subheader("🎯 Explainability Layer (XAI)")
        st.write("See which words influenced the AI's decision:")
        
        with st.spinner("Generating explanation map..."):
            import shap
            try:
                # 1. Get SHAP values from the engine
                shap_values = engine.get_explanation(user_input)
                
                # 2. Generate the raw SHAP HTML
                raw_shap_html = shap.plots.text(shap_values[0], display=False)
                
                # 3. Wrap the SHAP HTML in a white box so it's visible in Dark Mode
                styled_shap_html = f"""
                <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: inset 0 0 10px rgba(0,0,0,0.1);">
                    {raw_shap_html}
                </div>
                """
                
                # 4. Display in Streamlit
                components.html(styled_shap_html, height=250, scrolling=True)
                
                # 5. Caption with corrected colors!
                st.caption("🔴 Red = Contributes to POSITIVE | 🔵 Blue = Contributes to NEGATIVE")
                
                # 6. THE MISSING EXPANDER (Properly Indented)
                with st.expander("📊 How to read this SHAP graph"):
                    st.markdown("""
                    - **Base Value:** The model's neutral starting point before reading your specific words.
                    - **🔴 Red Blocks:** Words that push the AI's decision towards a **POSITIVE** sentiment.
                    - **🔵 Blue Blocks:** Words that pull the AI's decision towards a **NEGATIVE** sentiment.
                    - **Block Width:** The wider the word's block, the stronger its influence on the final decision.
                    - **f(inputs):** The final mathematical score calculated by the model.
                    """)
                    
            except Exception as e:
                st.warning("Explainability plot is currently optimized for English text.")
                logging.error(f"Error generating SHAP explanation: {str(e)}")
        
        # --- FINAL CELEBRATION ---
        if is_positive:
            st.balloons()
            
    else:
        st.warning("Please enter some text first!")

# 10. Footer
st.markdown("---")
st.caption("Developed by Manolina | Powered by Hugging Face Inference API & SHAP")