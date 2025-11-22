modules = ['streamlit','faiss','sentence_transformers','whisper','torch']
for m in modules:
    try:
        __import__(m)
        print(f"{m} OK")
    except Exception as e:
        print(f"{m} FAIL -> {type(e).__name__}: {e}")
