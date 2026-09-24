# -*- coding: utf-8 -*-
"""把 EBMC 论文 PDF 抽成文本，供分析使用。"""
import sys, os, re

SRC = r"C:\Users\Administrator\.dsh\attachments\v1\files\43\432fe7fc07a76677910ea8dd63e3927f8f8450d047ef346aa65d2a7cb8bd9d88\He 等 - 2026 - Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis.pdf"
OUT = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\deepseek_work\papers\EBMC_2026.txt"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

text = None
try:
    import pdfplumber
    parts = []
    with pdfplumber.open(SRC) as pdf:
        print("pages:", len(pdf.pages))
        for i, page in enumerate(pdf.pages):
            t = page.extract_text() or ""
            parts.append("\n\n===== PAGE %d =====\n" % (i + 1) + t)
    text = "".join(parts)
    print("pdfplumber ok")
except Exception as e:
    print("pdfplumber failed:", e)
    try:
        from pypdf import PdfReader
        r = PdfReader(SRC)
        text = "\n\n".join("===== PAGE %d =====\n" % (i + 1) + (p.extract_text() or "")
                           for i, p in enumerate(r.pages))
        print("pypdf ok")
    except Exception as e2:
        print("pypdf failed:", e2)

if text:
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print("chars:", len(text), "->", OUT)
    print(text[:1200])
