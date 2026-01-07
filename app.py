import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Paironix - AI Roommate Matching",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Get CSS based on theme
def get_theme_css(theme):
    if theme == 'light':
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            .stApp {
                font-family: 'Inter', sans-serif;
                background: #f8f9fa;
            }
            
            /* Theme Toggle Button */
            .theme-btn {
                position: fixed;
                top: 1rem;
                right: 1rem;
                z-index: 1000;
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 0.5rem 1rem;
                cursor: pointer;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                font-size: 0.9rem;
                font-weight: 600;
                color: #1e3a5f;
            }
            
            /* Header - Full Width with Image */
            .main-header {
                background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                margin: -1rem -1rem 2rem -1rem;
                position: relative;
                overflow: hidden;
                min-height: 400px;
            }
            
            .header-content {
                display: flex;
                align-items: center;
                height: 100%;
                min-height: 400px;
            }
            
            .header-text {
                flex: 1;
                padding: 3rem 4rem;
                z-index: 2;
            }
            
            .main-header h1 {
                color: white;
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 1rem;
            }
            
            .main-header p {
                color: rgba(255,255,255,0.9);
                font-size: 1.2rem;
                margin: 0;
                line-height: 1.6;
                max-width: 500px;
            }
            
            .header-image {
                flex: 1;
                height: 400px;
                position: relative;
            }
            
            .header-image img {
                width: 100%;
                height: 100%;
                object-fit: cover;
                object-position: center;
            }
            
            /* Cards */
            .info-card {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
            }
            
            .info-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            }
            
            .info-card h3 {
                color: #1e3a5f;
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            
            .info-card p {
                color: #64748b;
                font-size: 0.9rem;
                margin: 0;
            }
            
            /* Stats */
            .stat-card {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 1.5rem;
                text-align: center;
                transition: all 0.3s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            
            .stat-card .number {
                font-size: 2.5rem;
                font-weight: 700;
                color: #1e3a5f;
                line-height: 1;
            }
            
            .stat-card .label {
                color: #64748b;
                font-size: 0.85rem;
                margin-top: 0.5rem;
            }
            
            /* Room Cards */
            .room-card {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 1rem;
                margin-bottom: 0.75rem;
                transition: all 0.3s ease;
            }
            
            .room-card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                transform: translateX(4px);
            }
            
            .room-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.75rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid #f1f5f9;
            }
            
            .room-number {
                background: #1e3a5f;
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 4px;
                font-size: 0.8rem;
                font-weight: 600;
            }
            
            .room-type {
                color: #64748b;
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .room-members {
                list-style: none;
                padding: 0;
                margin: 0;
            }
            
            .room-members li {
                padding: 0.35rem 0;
                color: #334155;
                font-size: 0.9rem;
                border-bottom: 1px solid #f8fafc;
            }
            
            .room-members li:last-child {
                border-bottom: none;
            }
            
            /* Section Headers */
            .section-header {
                color: #1e3a5f;
                font-size: 1.5rem;
                font-weight: 600;
                margin: 2rem 0 1rem 0;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #e2e8f0;
            }
            
            /* Upload Area */
            .upload-area {
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                padding: 2rem;
                text-align: center;
                background: white;
                margin-bottom: 1.5rem;
            }
            
            .upload-area:hover {
                border-color: #1e3a5f;
                background: #f8f9fa;
            }
            
            /* Buttons */
            .stButton > button {
                background: #1e3a5f !important;
                color: white !important;
                border: none !important;
                padding: 0.75rem 2rem !important;
                font-weight: 600 !important;
                border-radius: 6px !important;
                transition: all 0.3s ease !important;
            }
            
            .stButton > button:hover {
                background: #2d5a87 !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 12px rgba(30, 58, 95, 0.3) !important;
            }
            
            /* Success Box */
            .success-box {
                background: #ecfdf5;
                border: 1px solid #a7f3d0;
                border-radius: 8px;
                padding: 1rem 1.5rem;
                color: #065f46;
                margin: 1rem 0;
            }
            
            /* Footer */
            .footer {
                background: #1e3a5f;
                color: white;
                padding: 2rem;
                margin: 3rem -1rem -1rem -1rem;
                text-align: center;
            }
            
            .footer p {
                color: rgba(255,255,255,0.7);
                margin: 0;
                font-size: 0.9rem;
            }
            
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            
            .stSlider label, .stSelectbox label {
                color: #334155;
                font-weight: 500;
            }
        </style>
        """
    else:  # dark theme
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            .stApp {
                font-family: 'Inter', sans-serif;
                background: #0f172a;
            }
            
            /* Theme Toggle Button */
            .theme-btn {
                position: fixed;
                top: 1rem;
                right: 1rem;
                z-index: 1000;
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 0.5rem 1rem;
                cursor: pointer;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                font-size: 0.9rem;
                font-weight: 600;
                color: #e2e8f0;
            }
            
            /* Header - Full Width with Image */
            .main-header {
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                margin: -1rem -1rem 2rem -1rem;
                position: relative;
                overflow: hidden;
                min-height: 400px;
            }
            
            .header-content {
                display: flex;
                align-items: center;
                height: 100%;
                min-height: 400px;
            }
            
            .header-text {
                flex: 1;
                padding: 3rem 4rem;
                z-index: 2;
            }
            
            .main-header h1 {
                color: white;
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 1rem;
            }
            
            .main-header p {
                color: rgba(255,255,255,0.9);
                font-size: 1.2rem;
                margin: 0;
                line-height: 1.6;
                max-width: 500px;
            }
            
            .header-image {
                flex: 1;
                height: 400px;
                position: relative;
            }
            
            .header-image img {
                width: 100%;
                height: 100%;
                object-fit: cover;
                object-position: center;
            }
            
            /* Cards */
            .info-card {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                transition: all 0.3s ease;
            }
            
            .info-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.4);
            }
            
            .info-card h3 {
                color: #e2e8f0;
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            
            .info-card p {
                color: #94a3b8;
                font-size: 0.9rem;
                margin: 0;
            }
            
            /* Stats */
            .stat-card {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 1.5rem;
                text-align: center;
                transition: all 0.3s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            }
            
            .stat-card .number {
                font-size: 2.5rem;
                font-weight: 700;
                color: #60a5fa;
                line-height: 1;
            }
            
            .stat-card .label {
                color: #94a3b8;
                font-size: 0.85rem;
                margin-top: 0.5rem;
            }
            
            /* Room Cards */
            .room-card {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 1rem;
                margin-bottom: 0.75rem;
                transition: all 0.3s ease;
            }
            
            .room-card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
                transform: translateX(4px);
            }
            
            .room-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.75rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid #334155;
            }
            
            .room-number {
                background: #3b82f6;
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 4px;
                font-size: 0.8rem;
                font-weight: 600;
            }
            
            .room-type {
                color: #94a3b8;
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .room-members {
                list-style: none;
                padding: 0;
                margin: 0;
            }
            
            .room-members li {
                padding: 0.35rem 0;
                color: #e2e8f0;
                font-size: 0.9rem;
                border-bottom: 1px solid #334155;
            }
            
            .room-members li:last-child {
                border-bottom: none;
            }
            
            /* Section Headers */
            .section-header {
                color: #e2e8f0;
                font-size: 1.5rem;
                font-weight: 600;
                margin: 2rem 0 1rem 0;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #334155;
            }
            
            /* Upload Area */
            .upload-area {
                border: 2px dashed #475569;
                border-radius: 8px;
                padding: 2rem;
                text-align: center;
                background: #1e293b;
                margin-bottom: 1.5rem;
            }
            
            .upload-area:hover {
                border-color: #60a5fa;
                background: #334155;
            }
            
            .upload-area h3 {
                color: #e2e8f0;
            }
            
            .upload-area p {
                color: #94a3b8;
            }
            
            /* Buttons */
            .stButton > button {
                background: #3b82f6 !important;
                color: white !important;
                border: none !important;
                padding: 0.75rem 2rem !important;
                font-weight: 600 !important;
                border-radius: 6px !important;
                transition: all 0.3s ease !important;
            }
            
            .stButton > button:hover {
                background: #2563eb !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
            }
            
            /* Success Box */
            .success-box {
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.3);
                border-radius: 8px;
                padding: 1rem 1.5rem;
                color: #6ee7b7;
                margin: 1rem 0;
            }
            
            /* Footer */
            .footer {
                background: #1e293b;
                color: white;
                padding: 2rem;
                margin: 3rem -1rem -1rem -1rem;
                text-align: center;
                border-top: 1px solid #334155;
            }
            
            .footer p {
                color: rgba(255,255,255,0.7);
                margin: 0;
                font-size: 0.9rem;
            }
            
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            
            .stSlider label, .stSelectbox label {
                color: #e2e8f0;
                font-weight: 500;
            }
        </style>
        """

# Apply theme CSS
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# Theme Toggle in Sidebar
with st.sidebar:
    theme_label = "🌙 Dark Mode" if st.session_state.theme == 'light' else "☀️ Light Mode"
    if st.button(theme_label, use_container_width=True):
        toggle_theme()
        st.rerun()

# Header with Full-Screen Vector Image
st.markdown("""
<div class="main-header">
    <div class="header-content">
        <div class="header-text">
            <h1>Paironix</h1>
            <p>AI-Powered Roommate Matching Platform for Universities</p>
        </div>
        <div class="header-image">
            <img src="https://img.freepik.com/premium-vector/flat-vector-illustration-girls-living-dorm-room-with-bunk-bed_108061-2387.jpg" 
                 alt="Students in dorm room" />
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Introduction cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>AI-Powered Analysis</h3>
        <p>Advanced TF-IDF vectorization and cosine similarity for optimal matching</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>Flexible Room Types</h3>
        <p>Configure 4-share and 2-share rooms based on your housing capacity</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>Export Results</h3>
        <p>Download complete room assignments as CSV for easy integration</p>
    </div>
    """, unsafe_allow_html=True)

# File upload section
st.markdown('<h2 class="section-header">Step 1: Upload Student Data</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload CSV file with student names and personality descriptions",
    type="csv",
    help="CSV should contain columns for student name and personality/interest description"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    total_students = len(df)
    
    st.markdown(f"""
    <div class="success-box">
        <strong>File loaded successfully</strong> - {total_students} students found
    </div>
    """, unsafe_allow_html=True)
    
    # Show data preview
    with st.expander("Preview Data", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Column selection
    st.markdown('<h2 class="section-header">Step 2: Configure Matching</h2>', unsafe_allow_html=True)
    
    name_cols = [col for col in df.columns if any(x in col.lower() for x in ['name', 'student'])]
    desc_cols = [col for col in df.columns if df[col].dtype == 'object']
    
    col1, col2 = st.columns(2)
    
    with col1:
        name_col = st.selectbox(
            "Student Name Column",
            name_cols if name_cols else df.columns.tolist(),
            index=0,
            help="Select the column containing student names"
        )
    
    with col2:
        desc_col = st.selectbox(
            "Personality/Interest Column",
            desc_cols if desc_cols else df.columns.tolist(),
            index=min(1, len(desc_cols)-1) if desc_cols else 0,
            help="Select the column containing personality descriptions or interests"
        )
    
    # Room configuration
    st.markdown("### Room Allocation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_4rooms = st.slider(
            "Number of 4-Share Rooms",
            min_value=0,
            max_value=50,
            value=25,
            help=f"Each 4-share room accommodates 4 students"
        )
        st.caption(f"Capacity: {num_4rooms * 4} students")
    
    with col2:
        num_2rooms = st.slider(
            "Number of 2-Share Rooms",
            min_value=0,
            max_value=50,
            value=20,
            help=f"Each 2-share room accommodates 2 students"
        )
        st.caption(f"Capacity: {num_2rooms * 2} students")
    
    total_capacity = num_4rooms * 4 + num_2rooms * 2
    st.info(f"Total room capacity: {total_capacity} students | Students to assign: {total_students}")
    
    # Match button
    st.markdown('<h2 class="section-header">Step 3: Run Matching Algorithm</h2>', unsafe_allow_html=True)
    
    if st.button(f"Match All {total_students} Students", type="primary", use_container_width=True):
        with st.spinner("Running AI matching algorithm..."):
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            X = vectorizer.fit_transform(df[desc_col].astype(str).fillna('unknown')).toarray()
            sim = cosine_similarity(X)
            
            used, rooms4, rooms2 = set(), [], []
            
            # Allocate 4-share rooms
            for r in range(num_4rooms):
                avail = [i for i in range(total_students) if i not in used]
                if len(avail) < 4:
                    break
                    
                # Find student with highest average similarity
                avgs = [np.mean(sim[i][avail]) for i in avail]
                starter = avail[np.argmax(avgs)]
                room = [starter]
                used.add(starter)
                
                # Find top 3 most similar
                avail2 = [i for i in avail if i != starter]
                scores = [(i, sim[starter][i]) for i in avail2]
                scores.sort(key=lambda x: x[1], reverse=True)
                
                for i, _ in scores[:3]:
                    room.append(i)
                    used.add(i)
                
                rooms4.append([str(df.iloc[i][name_col])[:25].strip() for i in room])
            
            # Allocate 2-share rooms
            avail = [i for i in range(total_students) if i not in used]
            
            for r in range(num_2rooms):
                if len(avail) < 2:
                    break
                    
                best_score, best_i, best_j = -1, 0, 1
                
                for a in range(len(avail)):
                    for b in range(a + 1, len(avail)):
                        score = sim[avail[a]][avail[b]]
                        if score > best_score:
                            best_score = score
                            best_i, best_j = a, b
                
                i, j = avail[best_i], avail[best_j]
                rooms2.append([
                    str(df.iloc[i][name_col])[:25].strip(),
                    str(df.iloc[j][name_col])[:25].strip()
                ])
                
                # Remove assigned students
                for idx in sorted([best_i, best_j], reverse=True):
                    avail.pop(idx)
            
            assigned = len(used)
        
        # Results section
        st.markdown('<h2 class="section-header">Matching Results</h2>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="success-box">
            <strong>Matching complete</strong> - {assigned} of {total_students} students assigned to {len(rooms4) + len(rooms2)} rooms
        </div>
        """, unsafe_allow_html=True)
        
        # Stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="number">{assigned}</div>
                <div class="label">Students Assigned</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="number">{len(rooms4)}</div>
                <div class="label">4-Share Rooms</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="number">{len(rooms2)}</div>
                <div class="label">2-Share Rooms</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Room assignments
        st.markdown("### Room Assignments")
        
        # 4-Share Rooms
        if rooms4:
            st.markdown("#### 4-Share Rooms")
            cols = st.columns(4)
            for i, room in enumerate(rooms4):
                with cols[i % 4]:
                    members_html = "".join([f"<li>{m}</li>" for m in room])
                    st.markdown(f"""
                    <div class="room-card">
                        <div class="room-header">
                            <span class="room-number">Room {i + 1}</span>
                            <span class="room-type">4-Share</span>
                        </div>
                        <ul class="room-members">{members_html}</ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        # 2-Share Rooms
        if rooms2:
            st.markdown("#### 2-Share Rooms")
            cols = st.columns(4)
            for i, room in enumerate(rooms2):
                with cols[i % 4]:
                    members_html = "".join([f"<li>{m}</li>" for m in room])
                    st.markdown(f"""
                    <div class="room-card">
                        <div class="room-header">
                            <span class="room-number">Room {len(rooms4) + i + 1}</span>
                            <span class="room-type">2-Share</span>
                        </div>
                        <ul class="room-members">{members_html}</ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Download button
        st.markdown("### Export Results")
        
        # Create export dataframe
        all_rooms = []
        for i, room in enumerate(rooms4):
            all_rooms.append({
                'Room Number': i + 1,
                'Room Type': '4-Share',
                'Member 1': room[0] if len(room) > 0 else '',
                'Member 2': room[1] if len(room) > 1 else '',
                'Member 3': room[2] if len(room) > 2 else '',
                'Member 4': room[3] if len(room) > 3 else ''
            })
        
        for i, room in enumerate(rooms2):
            all_rooms.append({
                'Room Number': len(rooms4) + i + 1,
                'Room Type': '2-Share',
                'Member 1': room[0] if len(room) > 0 else '',
                'Member 2': room[1] if len(room) > 1 else '',
                'Member 3': '',
                'Member 4': ''
            })
        
        export_df = pd.DataFrame(all_rooms)
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="Download Room Assignments (CSV)",
            data=csv,
            file_name="room_assignments.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Compatibility Matrix
        st.markdown("### Compatibility Matrix")
        st.caption("Visual representation of student compatibility scores (showing first 30 students)")
        
        display_size = min(30, len(sim))
        fig = px.imshow(
            sim[:display_size, :display_size],
            labels=dict(x="Student", y="Student", color="Compatibility"),
            color_continuous_scale='RdYlGn',
            aspect='equal'
        )
        fig.update_layout(
            title=f"Student Compatibility Matrix ({assigned} students matched)",
            width=700,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    # Show instructions when no file is uploaded
    st.markdown("""
    <div class="upload-area">
        <h3 style="color: #334155; margin-bottom: 0.5rem;">Upload Your Student Data</h3>
        <p style="color: #64748b;">Drag and drop a CSV file or click to browse</p>
        <p style="color: #94a3b8; font-size: 0.85rem;">CSV should include student names and personality/interest descriptions</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p><strong>Paironix</strong> - Intelligent Student Housing Solutions</p>
    <p style="margin-top: 0.5rem;">Powered by TF-IDF Vectorization and Cosine Similarity Analysis</p>
</div>
""", unsafe_allow_html=True)