import streamlit as st
from utils import qa_agent  # 使用封装好的模块函数


# 页面标题
st.title("📑 wbz的PDF助手")

# 侧边栏：输入 OpenAI API 密钥和 Base URL
with st.sidebar:
    openai_api_key = st.text_input("请输入 OpenAI API 密钥：", type = "password")
    base_url = st.text_input("请输入 API 基本 URL（仅在使用中转站购买API时输入，如果您是在官网购买的API，保留默认文本即可）：",
                             value="https://api.openai.com/v1")
    st.markdown("[获取 OpenAI API 密钥](https://platform.openai.com/account/api-keys)")

# 初始化会话历史
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # 使用普通列表存储历史记录

# 文件上传和问题输入
uploaded_file = st.file_uploader("上传你的 PDF 文件：", type="pdf")
question = st.text_input("对 PDF 的内容进行提问", disabled=not uploaded_file)

# 提示输入 API 密钥
if uploaded_file and question and not openai_api_key:
    st.info("请输入你的 OpenAI API 密钥")

# AI 问答逻辑
if uploaded_file and question and openai_api_key:
    with st.spinner("AI 正在思考中，请稍等..."):

        # 调用 utils.py 中的封装函数
        response = qa_agent(
            openai_api_key=openai_api_key,
            base_url=base_url,  # 传递 base_url
            uploaded_file=uploaded_file,
            chat_history=st.session_state["chat_history"],  # 会话历史
            question=question,
        )

        # 显示答案
        st.write("### 答案")
        st.write(response)  # 展示 AI 的回答

        # 更新会话历史
        st.session_state["chat_history"].append({"role": "user", "content": question})
        st.session_state["chat_history"].append({"role": "assistant", "content": response})

# 展示会话历史
if st.session_state["chat_history"]:
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i + 1]
            st.write(f"**你：** {human_message['content']}")
            st.write(f"**AI：** {ai_message['content']}")
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()