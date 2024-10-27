import streamlit as st

# 定义页面
ta_page = st.Page("ta.py", title="技术指标", icon="📊")  # Chart icon
rolling_corr_page = st.Page(
    "rolling_corr.py", title="滚动相关系数", icon="🔗"
)  # Link icon
corr_page = st.Page("corr.py", title="相关系数", icon="🔗")  # Link icon


# 使用 st.navigation 来定义导航
pg = st.navigation([ta_page, rolling_corr_page, corr_page])

# 设置页面配置
st.set_page_config(
    page_title="QuantResearch",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 运行选中的页面
pg.run()
