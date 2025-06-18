import streamlit as st
from keybert import KeyBERT

# 使用輕量多語模型，免除日文字典相依
kw_model = KeyBERT(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def extract_keywords(text: str, top_n: int = 10):
    """抽取語意關鍵詞片語"""
    return kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words=None,
        top_n=top_n,
    )

# ──────────────── Streamlit UI ────────────────
st.title("📘 日文語意關鍵詞抽取工具（文字版本）")

text = st.text_area("請輸入日文段落：", height=250)

if st.button("開始抽取關鍵詞") and text.strip():
    with st.spinner("模型分析中，請稍候…"):
        keywords = extract_keywords(text.strip(), top_n=10)

    st.subheader("🔍 抽取結果")
    for kw, score in keywords:
        st.write(f"🔹 **{kw}**　（重要性：{score:.2f}）")
else:
    st.caption("⬆️ 請先輸入文字，再點擊「開始抽取關鍵詞」")
