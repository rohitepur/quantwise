import streamlit as st

st.set_page_config(page_title="Rohit Epur – Portfolio", layout="wide")

# --- HEADER SECTION ---
st.title("Hello, I’m Rohit Epur")
st.markdown("##### Senior Technical Advisor | Cloud Strategist | Risk Leader")

st.markdown("---")

# --- HERO SECTION ---
col1, col2 = st.columns([1, 2])

with col1:
    st.image("rohit_profile2.jpg", width=240)
    st.write("📍 Philadelphia, PA")
    st.write("✉️ rohitepur@gmail.com")
    st.write("🔗 [LinkedIn](https://www.linkedin.com/in/rohit-epur-381273a/)")

with col2:
    st.subheader("About Me")
    st.markdown("""
I'm a seasoned technology architect and program leader with over 25 years of experience designing, implementing, and advising on enterprise-scale systems and cloud modernization at Vanguard.

I thrive at the intersection of **technical depth**, **risk management**, and **strategic vision**, shaping solutions that power investments, drive governance, and safeguard information at scale.

---
- 🏆 Award-winning Data Privacy Architect
- 🌐 Cloud Migration & Security Advisor
- 🧠 Mentor, Innovator, and Curious Technologist
""")

# --- CAREER SNAPSHOT ---
st.markdown("## 🔍 What I Do")

col3, col4, col5 = st.columns(3)

with col3:
    st.image("https://cdn-icons-png.flaticon.com/512/3050/3050155.png", width=60)
    st.markdown("**Cloud Strategy**")
    st.caption("Architect serverless, secure AWS and hybrid solutions at enterprise scale.")

with col4:
    st.image("https://cdn-icons-png.flaticon.com/512/2721/2721294.png", width=60)
    st.markdown("**Technical Risk Review**")
    st.caption("Conduct system-level risk assessments and influence governance decisions.")

with col5:
    st.image("https://cdn-icons-png.flaticon.com/512/1145/1145988.png", width=60)
    st.markdown("**Mentorship & Leadership**")
    st.caption("Guide engineers, build cross-functional alignment, and drive technical excellence.")

# --- PERSONAL SIDE ---
st.markdown("## 🌱 Beyond the Office")
st.markdown("""
- 🧩 I'm passionate about **STEM education** and mentoring aspiring engineers.
- 🧭 I enjoy exploring the balance between technology and ethics.
- 🏡 I value time with family and learning something new every day.

This site showcases my journey and current work. Explore the pages on the left to learn more!
""")
