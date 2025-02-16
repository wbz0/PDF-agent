import streamlit as st
from utils import qa_agent  # 封装的问答函数
import re
from streamlit import session_state
from openai import OpenAI

# Deepseek测试用API:sk-18f2f67107c34fc9a54b66d96f788002
# 页面标题
st.title("📑 wbz的PDF助手")

# 初始化 API Key 验证状态
if "api_valid" not in st.session_state:
    st.session_state.api_valid = None  # 初始状态：None（未验证）
if "api_error" not in st.session_state:
    st.session_state.api_error = ""  # 记录 API Key 错误信息

def validate_api_key(api_key):
    """检查 API Key 是否格式正确"""
    return bool(re.match(r"^sk-[a-zA-Z0-9-_]+$", api_key))  # 允许的格式

def check_api_key_validity(api_key, base_url):
    """调用 OpenAI API 进行 API Key 真实性验证"""
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        client.models.list()  # 测试 API Key 是否有效
        return True
    except Exception as e:
        return str(e)  # 返回错误信息

# 侧边栏：输入 API Key
with st.sidebar:
    deepseek_api_key = st.text_input("请输入您的 Deepseek API 密钥：", type="password")
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    if deepseek_api_key:
        if not validate_api_key(deepseek_api_key):
            st.session_state.api_valid = False
            st.session_state.api_error = "API Key 格式错误，请检查后重新输入！"
        else:
            validation_result = check_api_key_validity(deepseek_api_key, base_url)
            if validation_result is True:
                st.session_state.api_valid = True
                st.session_state.api_error = ""  # 清除错误信息
                st.success("API Key 验证成功！")
            else:
                st.session_state.api_valid = False
                st.session_state.api_error = f"API Key 无效：{validation_result}"

    # 显示 API Key 错误信息
    if st.session_state.api_error:
        st.error(st.session_state.api_error)

    st.markdown("[获取 Deepseek API key](https://cloud.siliconflow.cn/models)")
    st.write("###### 本模型基于 Deepseek-R1 大模型生成")

if "chat_history" not in session_state:
    chat_history = []
    session_state["chat_history"] = chat_history
# **上传 PDF，仅当 API Key 有效时可用**
uploaded_file = st.file_uploader("上传你的 PDF 文件：", type="pdf", disabled=not st.session_state.api_valid)

# **提问框，仅当 PDF 已上传时可用**
question = st.text_input("对 PDF 的内容进行提问", disabled=not uploaded_file)


def convert_latex_format(text):
    """ 将 LaTeX 公式格式转换为 Streamlit 兼容的格式，并修正特殊符号 """
    # 替换块级公式
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    # 替换行内公式
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text)

    # 替换数学符号
    replacements = {
        "∫": r"\int",
        "Ω": r"\Omega",
        "μ": r"\mu",
        "×": r"\times"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


# AI 问答逻辑
if uploaded_file and question:
    with st.spinner("AI 正在思考中，请稍等..."):
        # 调用 utils.py 中的封装函数
        response = qa_agent(
            deepseek_api_key=deepseek_api_key,
            base_url=base_url,
            uploaded_file=uploaded_file,
            chat_history=st.session_state["chat_history"],  # 会话历史
            question=question,
        )

        # 转换 LaTeX 公式格式
        formatted_answer = convert_latex_format(response)

        # 显示答案
        st.write("### 答案")

        # 分行处理 LaTeX 公式，确保复杂公式正确显示
        for line in formatted_answer.split("\n"):
            if line.startswith("$$") and line.endswith("$$"):  # 识别块级公式
                st.latex(line.strip("$$"))
            else:
                st.markdown(line, unsafe_allow_html=True)

        # 更新会话历史
        st.session_state["chat_history"].append({"role": "user", "content": question})
        st.session_state["chat_history"].append({"role": "assistant", "content": formatted_answer})

# 展示会话历史
if st.session_state["chat_history"]:
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i + 1]
            st.write(f"**你：** {human_message['content']}")
            st.markdown(f"**AI：** {ai_message['content']}")
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()

