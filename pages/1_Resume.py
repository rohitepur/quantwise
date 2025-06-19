import streamlit as st

st.set_page_config(layout="wide")
st.title("Rohit Epur - Resume")

# Top Section: Photo and Contact
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("Philadelphia, PA")
    st.markdown("rohitepur@gmail.com")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/rohit-epur-381273a/)")

with col2:
    st.header("Senior Architect | Cloud Strategist | Risk Advisor")
    st.write("""
Experienced technology leader with 25+ years at Vanguard, driving enterprise cloud adoption,
risk governance, and data privacy. Known for blending technical depth with strategic influence
to deliver secure and scalable solutions at scale.
""")

st.markdown("---")

# Education and Certifications
col3, col4 = st.columns(2)

with col3:
    st.subheader("🎓 Education")
    st.markdown("""
- MBA, Business Administration – *Strayer University*  
- B.S., Electronics & Comm. Engg – *GNDEC*  
- Leadership Impact Program – *UC Berkeley*
""")

with col4:
    st.subheader("📜 Certifications")
    st.markdown("""
- AWS Solutions Architect Associate  
- CISSP – Certified Information Systems Security Professional  
- CIPT – Certified Information Privacy Technologist
""")

st.markdown("---")

# Work Experience
st.subheader("💼 Work Experience")

with st.expander("📍 Vanguard (1999 – Present)"):
    st.markdown("""
**Senior Technical Risk Advisor (2021–Present)**  
Lead cloud migration risk reviews, multi-region architecture, and influence senior leaders.  

**Senior Solutions Architect (2018–2021)**  
Designed enterprise investment management solutions using Appian + AWS.  

**Data Governance Architect (2013–2018)**  
Led Collibra and IBM Infosphere implementations.  

**IT Project Manager & Technical Integrator (2005–2013)**  
Managed projects in retirement onboarding, DB2, and Siebel.  

**Software Engineer (1999–2005)**  
Built backend applications in COBOL, Oracle, and DB2.
""")

with st.expander("📍 GeBBS Consulting (1998 – 1999)"):
    st.write("Worked on telecom and finance systems for Bell Atlantic and Merrill Lynch.")

with st.expander("📍 Silverline Technologies (1996 – 1998)"):
    st.write("Built COBOL & DB2 systems for First Data Resources.")

st.markdown("---")

# Skills and Languages
col5, col6 = st.columns(2)

with col5:
    st.subheader("Technical Skills")
    st.markdown("""
- **Languages**: SQL, Python, COBOL, Shell scripting  
- **Platforms**: AWS, Appian, MQ, Oracle, DB2  
- **Security**: SAML, OAuth, risk frameworks
""")

with col6:
    st.subheader("Languages & Interests")
    st.markdown("""
- Languages: English, Telugu, Hindi  
- Interests: Mentoring, AI Ethics, STEM education
""")

st.markdown("---")
st.markdown("Download full resume [coming soon]")
