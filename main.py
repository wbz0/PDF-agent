import streamlit as st
from utils import qa_agent  # 使用封装好的模块函数
import requests

# 页面标题
st.title("📑 wbz的PDF助手")

# 使用表单输入 API 密钥和 Base URL
with st.sidebar.form(key="api_form"):
    temp_qwen_api_key = st.text_input("请输入您的通义千问 API 密钥：", type="password")

    submit_button = st.form_submit_button("提交")
    qwen_api_key = temp_qwen_api_key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"


# 初始化会话历史
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # 使用普通列表存储历史记录


uploaded_file = st.file_uploader("上传你的 PDF 文件：", type="pdf")
question = st.text_input("对 PDF 的内容进行提问", disabled=not uploaded_file)


# AI 问答逻辑
if uploaded_file and question :
    with st.spinner("AI 正在思考中，请稍等..."):

        # 调用 utils.py 中的封装函数
        response = qa_agent(
            qwen_api_key=qwen_api_key,
            base_url=base_url,
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

# 页面底部添加小字，确保它总是显示在页面底部且居中
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0px;  # 距离页面底部20px
        text-align: center;
        font-size: 17px;
        padding: 35px;
        background-color: white;
        margin-left: 175px;
    }
    </style>
    <div class="footer">
        本模型基于阿里通义千问大模型生成
    </div>
    """,
    unsafe_allow_html=True
)