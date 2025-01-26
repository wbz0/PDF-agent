import openai
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RAGSystem:
    def __init__(self, openai_api_key, base_url, texts, question, chat_history, model="text-embedding-ada-002"):
        """
        初始化 RAG 系统。

        参数：
            openai_api_key (str): OpenAI API 密钥。
            base_url (str): OpenAI API 的基本 URL（如果是中转站购买的 API，用户需要提供）。
            texts (list): 文本块列表，每个元素是一个字符串（文档片段）。
            question (str): 用户的问题。
            chat_history (list): 会话历史记录。
            model (str): 使用的嵌入模型，默认为 "text-embedding-ada-002"。
        """
        self.openai_api_key = openai_api_key
        self.base_url = base_url  # 新增 base_url 参数
        self.texts = texts
        self.question = question
        self.chat_history = chat_history
        self.model = model

        # 配置 OpenAI API 客户端的 ，，密钥与base_url
        openai.api_key = self.openai_api_key
        openai.api_base = self.base_url

    def get_embeddings(self, texts = None):
        """
        使用 OpenAI API 获取文本的嵌入向量。

        参数：
            texts (list): 文本块列表，每个元素是一个字符串（换句话说，是文档片段）。

        返回：
            embeddings (np.array): 返回一个二维数组，数组每一行都是一个嵌入向量
        """

        #补充说明：调用 OpenAI API 批量生成嵌入向量
        """response 是一个字典，常见结构为：
        {
        "data": [
            {"embedding": [...], "index": 0},
            {"embedding": [...], "index": 1}
        ],
        "model": "text-embedding-ada-002",
        ...
        }
        """
        if texts is None:
            texts = self.texts
        response = openai.embeddings.create(
            input=texts,
            model=self.model
        )
        embeddings = np.array([item.embedding for item in response.data])

        return embeddings


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
        # 使用 OpenAI API 获取问题的嵌入向量
        question_embedding = self.get_embeddings([self.question])[0]

        # 计算问题嵌入与文档嵌入的相似度
        similarities = cosine_similarity([question_embedding], index)
        top_k_indices = similarities.argsort()[0][-k:]  # 获取最相似的 k 个文档

        # 返回最相关的文档
        retrieved_docs = [self.texts[i] for i in top_k_indices]
        return retrieved_docs

    def generate_answer(self, retrieved_docs):
        """
        使用 OpenAI API 生成答案。

        参数：
            question (str): 用户提问。
            retrieved_docs (list): 检索到的相关文档列表。
            chat_history (list, optional): 之前的对话历史记录。

        返回：
            answer (str): 生成的答案。
        """
        # 拼接文档和问题作为上下文
        context = "\n".join(retrieved_docs)
        prompt = f"问题：{self.question}\n\n相关文档：\n{context}\n\n答案："

        # 如果存在会话历史，添加到上下文
        if self.chat_history:
            history_context = "\n".join([f"{message['role']}: {message['content']}" for message in self.chat_history])
            prompt = f"会话历史:{history_context}\n\n{prompt}"

        try:
            # 调用 OpenAI Chat API 生成答案
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )
            if response:
                return {"answer": response.choices[0].message.content.strip()}
            else:
                return {"answer": "无法生成答案，请稍后再试。"}
        except Exception as e:
            # 处理异常并返回默认答案
            print(f"Error: {e}")  # 打印异常信息
            return {"answer": f"发生错误: {str(e)}，请稍后再试。"}

    def qa(self, k=3):
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
        index = self.get_embeddings()

        # 2. 基于查询检索相关文档
        retrieved_docs = self.retrieve(index)

        # 3. 生成答案
        answer = self.generate_answer(retrieved_docs)

        # 返回答案和相关文档
        return answer