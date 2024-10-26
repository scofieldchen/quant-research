import streamlit as st

# 定义页面
ta_page = st.Page("ta.py", title="技术指标", icon="📊")  # Chart icon
correlation_page = st.Page("correlation.py", title="相关性分析", icon="🔗")  # Link icon

# 使用 st.navigation 来定义导航
pg = st.navigation([ta_page, correlation_page])

# 设置页面配置
st.set_page_config(
    page_title="QuantResearch",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

# 运行选中的页面
pg.run()
