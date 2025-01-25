import os
from PyPDF2 import PdfReader


def process_uploaded_file(uploaded_file, temp_dir="../temp_files"):
    """
    处理上传的 PDF 文件，将其保存并提取文本内容。

    参数：
        uploaded_file (file-like object): 上传的 PDF 文件对象。
        temp_dir (str): 临时文件存放目录，默认为 "../temp_files"。

    返回：
        docs (list): 包含 PDF 每页文本的列表。
    """
    # 确保临时目录存在
    os.makedirs(temp_dir, exist_ok=True)

    # 临时保存 PDF 文件
    temp_file_path = os.path.join(temp_dir, "temp.pdf")
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    # 解析 PDF 文件
    docs = []
    reader = PdfReader(temp_file_path)
    for page in reader.pages:
        text = page.extract_text()
        if text:  # 如果提取到了文本
            docs.append(text.strip())

    return docs