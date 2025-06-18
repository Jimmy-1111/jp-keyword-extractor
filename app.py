import streamlit as st
from keybert import KeyBERT
import pdfplumber

# ✅ 輕量且支援多語（含日文）的語意模型，免安裝 unidic
kw_model = KeyBERT(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def extract_keywords(text: str, top_n: int = 10):
    """
    使用 KeyBERT 抽取語意關鍵詞片語。
    """
    return kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words=None,
        top_n=top_n,
    )

# ───────────────────────── UI ──────────────────────────
st.title("📘 日文語意關鍵詞抽取工具")

input_mode = st.radio("📂 請選擇輸入方式", ["直接輸入", "上傳 PDF"])

text = ""

if input_mode == "直接輸入":
    text = st.text_area("請輸入日文段落：", height=200)

elif input_mode == "上傳 PDF":
    pdf_file = st.file_uploader("請上傳 PDF 檔案", type=["pdf"])
    if pdf_file is not None:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        st.success("✅ PDF 內容已載入！")

if st.button("開始抽取關鍵詞") and text.strip():
    with st.spinner("模型分析中，請稍候…"):
        keywords = extract_keywords(text.strip(), top_n=10)

    st.subheader("🔍 抽取結果")
    for kw, score in keywords:
        st.write(f"🔹 **{kw}** 　（重要性：{score:.2f}）")
else:
    st.caption("⬆️ 請先輸入文字或上傳 PDF，再點擊「開始抽取關鍵詞」")
