from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import sys
sys.stdout.reconfigure(encoding='utf-8')
class RAGSystem:
    def __init__(self, deepseek_api_key, base_url, texts, question, chat_history):
        """
        初始化 RAG 系统。

        参数：
            openai_api_key (str): Deepseek API 密钥。
            base_url (str): Deepseek API 的基本 URL。
            texts (list): 文本块列表，每个元素是一个字符串（文档片段）。
            question (str): 用户的问题。
            chat_history (list): 会话历史记录。
        """
        self.deepseek_api_key = deepseek_api_key
        self.base_url = base_url  # 新增 base_url 参数
        self.texts = texts
        self.question = question
        self.chat_history = chat_history


    # def get_embeddings(self, content):
    #     """
    #     使用 sentence-transformers 计算文本嵌入向量。
    #
    #     参数：
    #         batch_size (int): 每次请求的文本数量。
    #
    #     返回：
    #         embeddings (np.array): 返回一个二维数组，数组每一行都是一个嵌入向量。
    #     """
    #     batch_size = 20
    #     model = SentenceTransformer("all-MiniLM-L6-v2")  # 加载模型
    #     all_embeddings = []
    #
    #     for i in range(0, len(content), batch_size):
    #         batch_texts = content[i:i + batch_size]
    #         batch_embeddings = model.encode(batch_texts)  # 计算嵌入
    #         all_embeddings.append(batch_embeddings)
    #
    #     return np.vstack(all_embeddings)  # 合并所有批次的结果

    def get_embeddings(self, content):
        """
        使用 OpenAI API 获取文本的嵌入向量。

        参数：
            texts (list): 文本块列表，每个元素是一个字符串（换句话说，是文档片段）。
            batch_size (int): 每次请求的文本数量，不超过 20。

        返回：
            embeddings (np.array): 返回一个二维数组，数组每一行都是一个嵌入向量
        """

        client = OpenAI(api_key=self.deepseek_api_key, base_url=self.base_url)
        all_embeddings = []
        batch_size = 20

        # 分批处理文本

        for i in range(0, len(content), batch_size):
            batch_texts = content[i:i + batch_size]
            response = client.embeddings.create(
                input=batch_texts,
                model="text-embedding-v3"
            )
            # 提取并保存嵌入向量
            embeddings = np.array([item.embedding for item in response.data])
            all_embeddings.extend(embeddings)

        index = np.array(all_embeddings)

        return index


    def retrieve(self, index, k=3):
        """
        基于查询检索相关文档。

        参数：
            question (str): 用户提问。
            index (np.array): 文本嵌入的索引。
            texts (list): 文本块列表，每个元素是一个字符串（文档片段）。
            k (int): 要返回的最相似文档的数量。

        返回：
            retrieved_docs (list): 返回最相似的 k 个文档。
        """
        # 使用 OpenAI 兼容 API 获取问题的嵌入向量
        question_embedding = self.get_embeddings([self.question])
        # 计算问题嵌入与文档嵌入的相似度
        similarities = cosine_similarity(question_embedding, index)
        top_k_indices = similarities.argsort()[0][-k:]  # 获取最相似的 k 个文档

        # 返回最相关的文档
        retrieved_docs = [self.texts[i] for i in top_k_indices]
        return retrieved_docs


    def generate_answer(self, retrieved_docs):
        """
        使用 OpenAI 兼容 API 生成答案。

        参数：
            question (str): 用户提问。
            retrieved_docs (list): 检索到的相关文档列表。
            chat_history (list, optional): 之前的对话历史记录。

        返回：
            answer (str): 生成的答案。
        """
        # 如果有 chat_history，进行检索增强
        if self.chat_history:
            # 获取 chat_history 中每条消息的嵌入向量并检索最相关的 3 条
            chat_history_texts = [f"{message['role']}: {message['content']}" for message in self.chat_history]
            chat_history_index = self.get_embeddings(chat_history_texts)
            retrieved_chat_history = self.retrieve(chat_history_index, k=3)  # 这里传入的是 chat_history 的嵌入索引

            # 将检索到的最相关的 chat_history 与 retrieved_docs 结合
            context = "\n".join(retrieved_chat_history + retrieved_docs)
        else:
            context = "\n".join(retrieved_docs)

        prompt = f"问题：{self.question}\n\n相关文档：\n{context}"

        try:
            # 调用 OpenAI 兼容 API 生成答案
            client = OpenAI(api_key=self.deepseek_api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model="qwen-max",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8192
            )
            if response:
                return {"answer": response.choices[0].message.content}
            else:
                return {"answer": "无法生成答案，请稍后再试。"}
        except Exception as e:
            # 处理异常并返回默认答案
            print(f"Error: {e}")  # 打印异常信息
            return {"answer": f"发生错误: {str(e)}，请稍后再试。"}


    def qa(self):
        """
        基于检索的问答系统。

        参数：
            question (str): 用户提问。
            texts (list): 文本块列表，每个元素是一个字符串（文档片段）。
            k (int): 要返回的最相似文档的数量。
            chat_history (list, optional): 之前的对话历史记录。

        返回：
            dict: 包含生成的答案和相关文档的字典。
        """
        # 1. 构建向量存储索引
        index = self.get_embeddings(self.texts)

        # 2. 基于查询检索相关文档
        retrieved_docs = self.retrieve(index)

        # 3. 生成答案
        answer = self.generate_answer(retrieved_docs)

        # 返回答案和相关文档
        return answer




