import re

def split_documents(docs, chunk_size=1000, chunk_overlap=50, separators=None):
    """
    按指定规则将文档内容切分为较小的文本块。

    参数：
        docs (list): 文档内容列表，每个元素是文档的一页文本。
        chunk_size (int): 每个文本块的最大字符数。
        chunk_overlap (int): 相邻文本块之间的重叠字符数。
        separators (list): 自定义分隔符列表（按优先级排序）。

    返回：
        texts (list): 切分后的文本块列表。
    """
    if separators is None:
        separators = ["\n", "。", "！", "？", "，", "、", " "]

    texts = []  # 存储分割后的文本块

    for doc in docs:
        # 按优先级从大到小依次尝试分割
        for sep in separators:
            if sep in doc:
                chunks = doc.split(sep)
                break
        else:
            chunks = [doc]  # 如果没有匹配的分隔符，视为一个整体

        # 合并分割后的段落为指定大小的块
        current_chunk = []
        current_length = 0
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:  # 跳过空段
                continue

            if current_length + len(chunk) <= chunk_size:
                current_chunk.append(chunk)
                current_length += len(chunk)
            else:
                # 当前块已满，保存并重置
                if current_chunk:
                    texts.append("".join(current_chunk))
                # 添加重叠部分
                overlap_start = max(0, len(current_chunk) - chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + [chunk]
                current_length = sum(len(c) for c in current_chunk)

        # 保存最后一个块
        if current_chunk:
            texts.append("".join(current_chunk))

    return texts