import os
import sys

BASE = r"e:\招商银行RAG项目"
sys.path.insert(0, BASE)

from pypdf import PdfReader

pdf_path = os.path.join(BASE, "data", "规章制度", "上市公司信息披露管理办法.pdf")
reader = PdfReader(pdf_path)
print(f"Pages: {len(reader.pages)}")

text = ""
for i, page in enumerate(reader.pages[:10]):
    t = page.extract_text() or ""
    text += t
    print(f"Page {i}: {len(t)} chars")

print(f"\nFirst 1000 chars:\n{text[:1000]}")
