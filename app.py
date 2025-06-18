import streamlit as st
from keybert import KeyBERT
import pdfplumber

# 初始化 KeyBERT 模型
kw_model = KeyBERT(model='cl-tohoku/bert-base-japanese')

# 關鍵詞抽取函數
def extract_keywords(text):
    return kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words=None,
        top_n=10
    )

# Streamlit UI
st.title("📘 日文語意關鍵詞抽取工具")

option = st.radio("📂 請選擇輸入方式", ["直接輸入", "上傳 PDF"])

text = ""

if option == "直接輸入":
    text = st.text_area("請輸入日文段落：", height=200)

elif option == "上傳 PDF":
    uploaded = st.file_uploader("請上傳 PDF 檔案", type=["pdf"])
    if uploaded:
        with pdfplumber.open(uploaded) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"

# 抽取按鈕
if text and st.button("開始抽取關鍵詞"):
    keywords = extract_keywords(text)
    st.subheader("📌 抽取結果")
    for kw, score in keywords:
        st.write(f"🔹 {kw}（重要性: {score:.2f}）")
