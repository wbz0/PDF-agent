import os
import hashlib
import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import numpy as np
import shutil


# 用于存储解析过的 PDF 结果
pdf_cache = {}

def compute_pdf_hash(uploaded_file):
    """计算 PDF 文件的哈希值，用于检查是否已解析过"""
    file_content = uploaded_file.getvalue()
    return hashlib.md5(file_content).hexdigest()  # 计算 MD5 哈希值

@st.cache_resource
def get_ocr_instance():
    """缓存 OCR 实例，避免重复初始化"""
    return PaddleOCR(use_angle_cls=True, lang="ch")

def process_uploaded_file(uploaded_file, temp_dir="../temp_files"):
    """
    处理上传的 PDF 文件，提取文本或 OCR 解析扫描版 PDF。

    返回：
        docs (list): 包含 PDF 每页文本的列表。
    """
    pdf_hash = compute_pdf_hash(uploaded_file)

    # 如果 PDF 解析过，直接返回缓存的结果
    if pdf_hash in pdf_cache:
        return pdf_cache[pdf_hash]

    # 创建临时目录
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{pdf_hash}.pdf")

    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    reader = PdfReader(temp_file_path)
    docs = []
    is_scanned = False

    # 遍历 PDF 页数，尝试提取文本
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            docs.append(text.strip())  # 直接存储文本
        else:
            is_scanned = True  # 标记为扫描版 PDF

    # 如果 PDF 是扫描版，使用 OCR 解析
    if is_scanned:
        if shutil.which("pdftoppm") is None:
            raise RuntimeError("错误：Poppler 未安装，请参考 https://github.com/Belval/pdf2image#installation 安装 Poppler 并配置环境变量。")

        images = convert_from_path(temp_file_path)  # 将 PDF 转换为图片
        ocr = get_ocr_instance()  # 获取缓存的 OCR 实例

        for img in images:
            img_np = np.array(img)
            result = ocr.ocr(img_np, cls=True)
            extracted_text = "\n".join([word_info[1][0] for line in result for word_info in line]).strip()
            docs.append(extracted_text)

    # 缓存解析结果
    pdf_cache[pdf_hash] = docs
    return docs