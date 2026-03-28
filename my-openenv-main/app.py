"""
app.py — Modern Streamlit Dashboard for the AI Digital Life Simulator.

Features:
- Dark / Light mode toggle
- Glassmorphism card design
- Animated progress bars for all life metrics
- Timeline charts (Plotly)
- Event notification feed
- Action history panel
- AI decision explanation
- Live score display
- Simulation playback controls
"""

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from pathlib import Path

from env import LifeSimulatorEnv
from models import Action, Personality, Difficulty, VALID_ACTIONS, TaskType
from grader import grade_agent, grade_label
from agent import BaselineAgent


# ═══════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title="AI Digital Life Simulator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════
#  LOAD CSS
# ═══════════════════════════════════════════════

def load_css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
    else:
        css = ""

    # Get theme from session state
    theme = st.session_state.get("theme", "dark")

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * {{ font-family: 'Inter', sans-serif; }}

    {css}

    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{background: transparent !important;}}

    /* Streamlit element overrides */
    .stApp {{
        background: {"#0a0a1a" if theme == "dark" else "#f8f9fc"} !important;
    }}

    /* Force text color on all generic text elements */
    .stMarkdown, .stText, p, span, label, h1, h2, h3, h4, h5, h6, li, .st-emotion-cache-16idsys p {{
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
    }}

    /* Fix selectbox styling */
    .stSelectbox > div > div {{
        background: {"rgba(25,25,50,0.65)" if theme == "dark" else "rgba(255,255,255,0.55)"} !important;
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
        border-color: {"rgba(100,100,200,0.2)" if theme == "dark" else "rgba(0,0,0,0.1)"} !important;
        border-radius: 12px !important;
    }}

    div[data-baseweb="select"] > div {{
        background: {"rgba(25,25,50,0.65)" if theme == "dark" else "rgba(255,255,255,0.55)"} !important;
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
        border-radius: 12px !important;
    }}

    /* Fix dropdown options menu visibility */
    div[data-baseweb="popover"], div[data-baseweb="popover"] *, ul[role="listbox"], ul[role="listbox"] * {{
        background-color: {"#141428" if theme == "dark" else "#ffffff"} !important;
        color: {"#e8e8f0" if theme == "dark" else "#1e1e3f"} !important;
    }}
    
    /* Ensure hover states are visible */
    ul[role="listbox"] li:hover, div[data-baseweb="popover"] li:hover {{
        background-color: {"#252545" if theme == "dark" else "#f0f2f6"} !important;
    }}

    /* Fix slider text (min/max/current values) */
    div[data-baseweb="slider"] div {{
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
    }}

    .stButton > button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }}

    .stSlider > div > div > div {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }}

    .stSidebar {{
        background: {"rgba(15,15,30,0.95)" if theme == "dark" else "rgba(240,242,248,0.95)"} !important;
        backdrop-filter: blur(20px) !important;
    }}

    .stSidebar .stMarkdown, .stSidebar p, .stSidebar span, .stSidebar label {{
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
    }}

    div[data-testid="stExpander"] {{
        background: {"rgba(25,25,50,0.4)" if theme == "dark" else "rgba(255,255,255,0.4)"} !important;
        border-radius: 12px !important;
        border: 1px solid {"rgba(100,100,200,0.15)" if theme == "dark" else "rgba(0,0,0,0.08)"} !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: {"rgba(25,25,50,0.3)" if theme == "dark" else "rgba(255,255,255,0.3)"} !important;
        border-radius: 10px !important;
        color: {"#a8a8c8" if theme == "dark" else "#4a4a6a"} !important;
        border: none !important;
        padding: 8px 20px !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }}

    /* ─── Text Input / Number Input ─── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stNumberInput input,
    .stTextInput input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    input[type="number"],
    input[type="text"],
    div[data-baseweb="input"] input {{
        background: {"rgba(25,25,50,0.65)" if theme == "dark" else "rgba(255,255,255,0.95)"} !important;
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
        border: 1px solid {"rgba(100,100,200,0.2)" if theme == "dark" else "rgba(0,0,0,0.15)"} !important;
        border-radius: 12px !important;
        -webkit-text-fill-color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
    }}
    /* Number input wrapper background */
    .stNumberInput > div > div,
    div[data-testid="stNumberInput"] > div > div,
    div[data-baseweb="input"] {{
        background: {"rgba(25,25,50,0.65)" if theme == "dark" else "rgba(255,255,255,0.95)"} !important;
        border-radius: 12px !important;
    }}
    .stTextInput > div > div > input::placeholder {{
        color: {"#6868a8" if theme == "dark" else "#9999bb"} !important;
    }}

    /* ─── Radio Button Labels ─── */
    .stRadio > div > div > label,
    .stRadio > div > div > label > div,
    .stRadio > div > div > label > div > p,
    .stRadio > div > div > label span {{
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
    }}
    .stRadio > div {{
        background: {"rgba(25,25,50,0.3)" if theme == "dark" else "rgba(255,255,255,0.5)"} !important;
        border-radius: 12px !important;
        padding: 4px 8px !important;
    }}

    /* ─── Checkbox Labels ─── */
    .stCheckbox > label > span {{
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
    }}

    /* ─── Selectbox: selected value text inside the widget ─── */
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div[class*="valueContainer"] *,
    div[data-baseweb="select"] div[class*="singleValue"],
    .stSelectbox div[data-baseweb="select"] * {{
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
    }}

    /* ─── Selectbox: dropdown arrow icon ─── */
    div[data-baseweb="select"] svg {{
        fill: {"#a8a8c8" if theme == "dark" else "#4a4a6a"} !important;
    }}

    /* ─── Number input +/- buttons ─── */
    .stNumberInput > div > div > div > button,
    .stNumberInput button,
    div[data-testid="stNumberInput"] button,
    div[data-baseweb="input"] button,
    .stNumberInput [data-testid="stNumberInputStepUp"],
    .stNumberInput [data-testid="stNumberInputStepDown"] {{
        background: {"rgba(25,25,50,0.5)" if theme == "dark" else "rgba(0,0,0,0.05)"} !important;
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
        border-color: {"rgba(100,100,200,0.2)" if theme == "dark" else "rgba(0,0,0,0.1)"} !important;
    }}
    /* Target the SVG icons specifically to make them dark in light mode */
    .stNumberInput button svg,
    div[data-baseweb="input"] button svg,
    .stNumberInput [data-testid="stNumberInputStepUp"] svg,
    .stNumberInput [data-testid="stNumberInputStepDown"] svg {{
        fill: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
    }}

    /* ─── Slider: value label and tick text ─── */
    .stSlider > div > div > div > div,
    .stSlider label > div,
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"],
    div[data-baseweb="slider"] div[role="slider"]::after {{
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
    }}

    /* ─── Sidebar widget labels (selectbox, slider, radio, etc.) ─── */
    .stSidebar label,
    .stSidebar .stSelectbox label,
    .stSidebar .stSlider label,
    .stSidebar .stRadio label,
    .stSidebar .stNumberInput label,
    .stSidebar .stTextInput label,
    .stSidebar [data-testid="stWidgetLabel"],
    .stSidebar [data-testid="stWidgetLabel"] p {{
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
    }}

    /* ─── Sidebar section headers (#### markdown) ─── */
    .stSidebar h4, .stSidebar h3, .stSidebar h5,
    .stSidebar strong, .stSidebar b {{
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
    }}

    /* ─── Help tooltip text ─── */
    div[data-baseweb="tooltip"] {{
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
    }}

    /* ─── Disabled button text visibility ─── */
    .stButton > button:disabled {{
        background: {"rgba(40,40,70,0.4)" if theme == "dark" else "rgba(200,200,220,0.5)"} !important;
        color: {"#6868a8" if theme == "dark" else "#8888aa"} !important;
        box-shadow: none !important;
    }}

    /* ─── Info/Warning/Error boxes ─── */
    .stAlert {{
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"} !important;
    }}

    /* ─── Sidebar overall text elements ─── */
    section[data-testid="stSidebar"] * {{
        color: {"#e8e8f0" if theme == "dark" else "#1a1a2e"};
    }}
    section[data-testid="stSidebar"] .stButton > button {{
        color: white !important;
    }}
    section[data-testid="stSidebar"] .stButton > button:disabled {{
        color: {"#6868a8" if theme == "dark" else "#8888aa"} !important;
    }}

    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════

def get_color(value: float, metric: str = "") -> str:
    """Return a color based on value (0-100)."""
    if metric == "stress":
        if value >= 70: return "#ef4444"
        if value >= 40: return "#f59e0b"
        return "#10b981"
    else:
        if value >= 70: return "#10b981"
        if value >= 40: return "#f59e0b"
        return "#ef4444"


def render_metric_card(emoji: str, label: str, value: float,
                       max_val: float = 100, color_class: str = "health"):
    """Render a glassmorphism metric card with animated progress bar."""
    pct = min(value / max_val * 100, 100)
    theme = st.session_state.get("theme", "dark")
    bg = "rgba(25,25,50,0.65)" if theme == "dark" else "rgba(255,255,255,0.55)"
    border = "rgba(100,100,200,0.2)" if theme == "dark" else "rgba(0,0,0,0.08)"
    text = "#e8e8f0" if theme == "dark" else "#1a1a2e"
    muted = "#6868a8" if theme == "dark" else "#8888aa"

    st.markdown(f"""
    <div style="
        background: {bg};
        backdrop-filter: blur(16px);
        border: 1px solid {border};
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        cursor: pointer;
    ">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;
             background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
             border-radius:12px 12px 0 0;"></div>
        <div style="font-size:1.5rem;margin-bottom:6px;">{emoji}</div>
        <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:1.5px;
             color:{muted};margin-bottom:8px;font-weight:600;">{label}</div>
        <div style="font-size:1.8rem;font-weight:700;color:{text};margin-bottom:8px;">
            {value:.1f}</div>
        <div style="background:rgba(128,128,128,0.15);border-radius:10px;
             overflow:hidden;height:10px;">
            <div class="progress-fill progress-{color_class}"
                 style="width:{pct}%;height:100%;border-radius:10px;
                 transition:width 0.8s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_score_card(score: float, label: str):
    """Render a prominent score display card."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        color: white;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.35);
        cursor: pointer;
    ">
        <div style="position:absolute;top:-50%;left:-50%;width:200%;height:200%;
             background:radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
             animation: pulse 3s ease-in-out infinite;"></div>
        <div style="font-size:0.9rem;opacity:0.85;position:relative;z-index:1;
             text-transform:uppercase;letter-spacing:2px;font-weight:600;">
            LIFE SCORE</div>
        <div style="font-size:3.5rem;font-weight:800;position:relative;z-index:1;">
            {score:.2f}</div>
        <div style="font-size:1rem;opacity:0.9;position:relative;z-index:1;">
            {label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_event_card(event: dict, theme: str):
    """Render a single event notification card."""
    bg = "rgba(25,25,50,0.4)" if theme == "dark" else "rgba(255,255,255,0.4)"
    text = "#e8e8f0" if theme == "dark" else "#1a1a2e"
    muted = "#6868a8" if theme == "dark" else "#8888aa"
    accent = "#818cf8" if theme == "dark" else "#667eea"

    st.markdown(f"""
    <div style="
        background: {bg};
        backdrop-filter: blur(12px);
        border-left: 4px solid {accent};
        border-radius: 0 12px 12px 0;
        padding: 10px 14px;
        margin-bottom: 6px;
    ">
        <div style="font-size:0.7rem;color:{muted};font-weight:600;">
            Step {event.get('step', '?')} · Week {event.get('week', '?')}</div>
        <div style="color:{text};font-size:0.85rem;margin-top:2px;">
            {event.get('description', '')}</div>
    </div>
    """, unsafe_allow_html=True)


def create_timeline_chart(history: list, theme: str):
    """Create a Plotly timeline chart showing all metrics over time."""
    if not history:
        return None

    steps = list(range(len(history)))

    bg = "rgba(10,10,26,0)" if theme == "dark" else "rgba(240,242,245,0)"
    grid = "rgba(100,100,200,0.1)" if theme == "dark" else "rgba(0,0,0,0.06)"
    text_color = "#a8a8c8" if theme == "dark" else "#4a4a6a"

    fig = make_subplots(rows=1, cols=1)

    metrics = {
        "Health": {"color": "#10b981", "key": "health"},
        "Money (÷100)": {"color": "#f59e0b", "key": "money", "scale": 0.01},
        "Stress": {"color": "#ef4444", "key": "stress"},
        "Career": {"color": "#3b82f6", "key": "career"},
        "Relationships": {"color": "#ec4899", "key": "relationships"},
        "Happiness": {"color": "#8b5cf6", "key": "happiness"},
    }

    for name, cfg in metrics.items():
        scale = cfg.get("scale", 1.0)
        values = [h.get(cfg["key"], 0) * scale for h in history]
        fig.add_trace(go.Scatter(
            x=steps, y=values, name=name,
            mode="lines",
            line=dict(color=cfg["color"], width=2.5, shape="spline"),
            fill="none",
            hovertemplate=f"{name}: %{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=text_color, family="Inter"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=20, t=30, b=30),
        height=350,
        xaxis=dict(
            title="Step", gridcolor=grid, zerolinecolor=grid,
            showgrid=True, gridwidth=1,
        ),
        yaxis=dict(
            title=dict(text="Value", standoff=15),
            gridcolor=grid, zerolinecolor=grid,
            showgrid=True, gridwidth=1,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(20,20,40,0.95)" if theme == "dark" else "rgba(255,255,255,0.95)",
            font_color="#e8e8f0" if theme == "dark" else "#1a1a2e",
            bordercolor="rgba(100,100,200,0.3)" if theme == "dark" else "rgba(0,0,0,0.1)",
            font_size=12,
        ),
    )

    return fig


def create_reward_chart(rewards: list, theme: str):
    """Create a cumulative reward chart."""
    if not rewards:
        return None

    cumulative = []
    total = 0
    for r in rewards:
        total += r
        cumulative.append(total)

    bg = "rgba(10,10,26,0)" if theme == "dark" else "rgba(240,242,245,0)"
    grid = "rgba(100,100,200,0.1)" if theme == "dark" else "rgba(0,0,0,0.06)"
    text_color = "#a8a8c8" if theme == "dark" else "#4a4a6a"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(cumulative))), y=cumulative,
        mode="lines", name="Cumulative Reward",
        line=dict(color="#667eea", width=3, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(102,126,234,0.15)",
    ))
    fig.add_trace(go.Bar(
        x=list(range(len(rewards))), y=rewards,
        name="Step Reward", opacity=0.4,
        marker_color="#764ba2",
    ))

    fig.update_layout(
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(color=text_color, family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=30, b=30),
        height=300,
        xaxis=dict(title="Step", gridcolor=grid, showgrid=True),
        yaxis=dict(title=dict(text="Reward", standoff=15), gridcolor=grid, showgrid=True),
        barmode="overlay",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(20,20,40,0.95)" if theme == "dark" else "rgba(255,255,255,0.95)",
            font_color="#e8e8f0" if theme == "dark" else "#1a1a2e",
            bordercolor="rgba(100,100,200,0.3)" if theme == "dark" else "rgba(0,0,0,0.1)",
            font_size=12,
        ),
    )
    return fig


# ═══════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════

def init_session():
    """Initialize session state variables."""
    defaults = {
        "theme": "dark",
        "registered": False,
        "user_name": "",
        "user_career": "",
        "play_mode": "AI Agent",
        "task_goal": "perfect_balance",
        "env": None,
        "agent": None,
        "sim_running": False,
        "sim_done": False,
        "current_step": 0,
        "history": [],
        "action_history": [],
        "reward_history": [],
        "event_log": [],
        "decisions": [],
        "final_grade": None,
        "auto_play": False,
        "play_speed": 0.3,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════

def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:16px;">
            <div style="font-size:2.5rem;">🧬</div>
            <div style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 background-clip:text;font-size:1.3rem;font-weight:800;">
                AI Life Simulator</div>
            <div style="color:{'#a8a8c8' if st.session_state.theme == 'dark' else '#6a6a8a'};font-size:0.85rem;margin-top:4px;">
                Player: <b>{st.session_state.user_name}</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Theme toggle
        theme_col1, theme_col2 = st.columns(2)
        with theme_col1:
            if st.button("🌙 Dark", use_container_width=True,
                         disabled=st.session_state.theme == "dark"):
                st.session_state.theme = "dark"
                st.rerun()
        with theme_col2:
            if st.button("☀️ Light", use_container_width=True,
                         disabled=st.session_state.theme == "light"):
                st.session_state.theme = "light"
                st.rerun()

        st.markdown("---")
        st.markdown("#### ⚙️ Configuration")

        # Play Mode
        play_mode = st.radio("Play Mode", ["AI Agent", "Manual Player"], horizontal=True, 
                             help="Watch AI play or make decisions yourself!")

        # Task Goal
        task_goal = st.selectbox(
            "Evaluation Task Goal",
            [t.value for t in TaskType],
            index=2,  # perfect_balance default
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Select the OpenEnv objective for the AI (or yourself) to achieve."
        )

        # Personality
        personality = st.selectbox(
            "Personality",
            [p.value for p in Personality],
            index=4,  # balanced
            help="Personality affects action impacts and stress levels",
        )

        # Difficulty
        difficulty = st.selectbox(
            "Difficulty",
            [d.value for d in Difficulty],
            index=1,  # medium
            help="Controls event frequency and trade-off severity",
        )

        # Seed
        seed = st.number_input("Random Seed", value=42, min_value=0, max_value=9999)

        # Max steps
        max_steps = st.slider("Max Steps", 50, 300, 100, 10)

        # Speed
        st.session_state.play_speed = st.slider(
            "Playback Speed (sec)", 0.05, 1.0, 0.3, 0.05)

        st.markdown("---")
        st.markdown("#### Simulation Controls")

        # Start / Reset
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶ Initialize", use_container_width=True):
                # Initialize environment and agent
                env = LifeSimulatorEnv(
                    task_type=TaskType(task_goal),
                    personality=Personality(personality),
                    difficulty=Difficulty(difficulty),
                    seed=seed,
                    max_steps=max_steps,
                )
                state = env.reset()
                agent = BaselineAgent(task_type=TaskType(task_goal), seed=seed)

                st.session_state.play_mode = play_mode
                st.session_state.task_goal = task_goal
                st.session_state.env = env
                st.session_state.agent = agent
                st.session_state.sim_running = True
                st.session_state.sim_done = False
                st.session_state.current_step = 0
                st.session_state.history = [state]
                st.session_state.action_history = []
                st.session_state.reward_history = []
                st.session_state.event_log = []
                st.session_state.decisions = []
                st.session_state.final_grade = None
                st.session_state.auto_play = False
                st.rerun()

        with c2:
            if st.button("🔄 Reset", use_container_width=True):
                for key in ["env", "agent"]:
                    st.session_state[key] = None
                st.session_state.sim_running = False
                st.session_state.sim_done = False
                st.session_state.current_step = 0
                st.session_state.history = []
                st.session_state.action_history = []
                st.session_state.reward_history = []
                st.session_state.event_log = []
                st.session_state.decisions = []
                st.session_state.final_grade = None
                st.session_state.auto_play = False
                st.rerun()

        if st.session_state.sim_running and not st.session_state.sim_done:
            st.markdown("---")
            if st.session_state.play_mode == "Manual Player":
                st.markdown("#### 🕹️ Manual Actions")
                c_a1, c_a2 = st.columns(2)
                if c_a1.button("💼 Work", use_container_width=True): do_manual_step("work_overtime")
                if c_a2.button("🏃 Exercise", use_container_width=True): do_manual_step("exercise")
                c_a3, c_a4 = st.columns(2)
                if c_a3.button("📈 Invest", use_container_width=True): do_manual_step("invest_money")
                if c_a4.button("📚 Learn", use_container_width=True): do_manual_step("learn_skill")
                c_a5, c_a6 = st.columns(2)
                if c_a5.button("🎉 Socialize", use_container_width=True): do_manual_step("socialize")
                if c_a6.button("😴 Rest", use_container_width=True): do_manual_step("rest")
            else:
                c3, c4 = st.columns(2)
                with c3:
                    if st.button("⏭ Step AI", use_container_width=True):
                        do_step()
                        st.rerun()
                with c4:
                    if st.button("⏩ Auto-Step" if not st.session_state.auto_play
                                 else "⏸ Pause", use_container_width=True):
                        st.session_state.auto_play = not st.session_state.auto_play
                        st.rerun()

        # Info
        st.markdown("---")
        if st.session_state.sim_running:
            st.markdown(f"**Step:** {st.session_state.current_step}")
            if st.session_state.history:
                s = st.session_state.history[-1]
                st.markdown(f"**Age:** {s.get('age', 25):.1f} years")
                st.markdown(f"**Week:** {s.get('week', 0)}")

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;font-size:0.7rem;opacity:0.5;'>"
            "AI Digital Life Simulator v1.0<br>Hackathon Edition</div>",
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════
#  STEP LOGIC
# ═══════════════════════════════════════════════

def do_step():
    """Execute one simulation step."""
    env = st.session_state.env
    agent = st.session_state.agent
    if env is None or agent is None or st.session_state.sim_done:
        return

    state = st.session_state.history[-1]
    action, reasoning = agent.decide(state)

    new_state, reward, done, info = env.step(action)

    st.session_state.current_step += 1
    st.session_state.history.append(new_state)
    st.session_state.action_history.append(action)
    st.session_state.reward_history.append(reward)

    # Record decision
    st.session_state.decisions.append({
        "step": st.session_state.current_step,
        "action": action,
        "reasoning": reasoning,
        "reward": reward,
    })

    # Record events
    if "events" in info:
        for ev in info["events"]:
            ev["step"] = st.session_state.current_step
            ev["week"] = info.get("week", st.session_state.current_step)
            st.session_state.event_log.append(ev)

    if done:
        st.session_state.sim_done = True
        st.session_state.auto_play = False
        st.session_state.final_grade = grade_agent(new_state, task_type=st.session_state.task_goal)


def do_manual_step(action: str):
    """Execute a manual step chosen by the user."""
    env = st.session_state.env
    if env is None or st.session_state.sim_done:
        return

    reasoning = "Manual player decision."
    
    new_state, reward, done, info = env.step(action)

    st.session_state.current_step += 1
    st.session_state.history.append(new_state)
    st.session_state.action_history.append(action)
    st.session_state.reward_history.append(reward)

    # Record decision
    st.session_state.decisions.append({
        "step": st.session_state.current_step,
        "action": action,
        "reasoning": reasoning,
        "reward": reward,
    })

    # Record events
    if "events" in info:
        for ev in info["events"]:
            ev["step"] = st.session_state.current_step
            ev["week"] = info.get("week", st.session_state.current_step)
            st.session_state.event_log.append(ev)

    if done:
        st.session_state.sim_done = True
        st.session_state.auto_play = False
        st.session_state.final_grade = grade_agent(new_state)
        
    st.rerun()


# ═══════════════════════════════════════════════
#  MAIN PAGE
# ═══════════════════════════════════════════════

def render_main():
    """Render the main dashboard content."""
    theme = st.session_state.get("theme", "dark")

    # Header
    career_text = f" • Aspiring {st.session_state.user_career}" if st.session_state.get('user_career') else ""
    st.markdown(f"""
    <div style="text-align:center;padding:10px 0 20px 0;">
        <div style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text;font-size:2.2rem;font-weight:800;letter-spacing:-0.5px;
              cursor: pointer;">
            {st.session_state.user_name}'s Digital Life</div>
        <div style="color:{'#a8a8c8' if theme == 'dark' else '#4a4a6a'};
             font-size:1rem;margin-top:4px;cursor: pointer;">
             Navigate life's decisions{career_text}</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.sim_running:
        render_welcome(theme)
        return

    # Get current state
    state = st.session_state.history[-1] if st.session_state.history else {}

    # ─── Top Metrics Row ───
    cols = st.columns(6)
    metrics = [
        ("❤️", "Health", state.get("health", 0), 100, "health"),
        ("💰", "Money", state.get("money", 0), 10000, "money"),
        ("😰", "Stress", state.get("stress", 0), 100, "stress"),
        ("💼", "Career", state.get("career", 0), 100, "career"),
        ("👥", "Relations", state.get("relationships", 0), 100, "relationships"),
        ("😊", "Happiness", state.get("happiness", 0), 100, "happiness"),
    ]
    for col, (emoji, label, val, mx, cls) in zip(cols, metrics):
        with col:
            render_metric_card(emoji, label, val, mx, cls)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Main Content Tabs ───
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Timeline", "🔔 Events", "🤖 AI Decisions", "📊 Rewards"
    ])

    with tab1:
        fig = create_timeline_chart(st.session_state.history, theme)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, theme=None)
        else:
            st.info("Run the simulation to see timeline data.")

    with tab2:
        if st.session_state.event_log:
            for ev in reversed(st.session_state.event_log[-20:]):
                render_event_card(ev, theme)
        else:
            st.info("No events triggered yet.")

    with tab3:
        if st.session_state.decisions:
            for d in reversed(st.session_state.decisions[-15:]):
                accent = "#818cf8" if theme == "dark" else "#667eea"
                bg = "rgba(25,25,50,0.4)" if theme == "dark" else "rgba(255,255,255,0.4)"
                text_c = "#e8e8f0" if theme == "dark" else "#1a1a2e"
                muted = "#6868a8" if theme == "dark" else "#8888aa"
                action_emojis = {
                    "work_overtime": "💼", "exercise": "🏃", "invest_money": "📈",
                    "learn_skill": "📚", "socialize": "🎉", "rest": "😴",
                }
                em = action_emojis.get(d["action"], "🔹")
                st.markdown(f"""
                <div style="background:{bg};backdrop-filter:blur(12px);
                     border:1px solid {'rgba(100,100,200,0.15)' if theme=='dark' else 'rgba(0,0,0,0.08)'};
                     border-radius:12px;padding:14px;margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="font-size:0.7rem;color:{muted};font-weight:600;">
                                Step {d['step']}</span>
                            <span style="font-weight:700;color:{accent};margin-left:8px;">
                                {em} {d['action'].replace('_',' ').title()}</span>
                        </div>
                        <span style="font-size:0.85rem;font-weight:600;
                              color:{'#10b981' if d['reward']>=0 else '#ef4444'};">
                            {d['reward']:+.3f}</span>
                    </div>
                    <div style="color:{muted};font-size:0.82rem;margin-top:4px;">
                        {d['reasoning']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No decisions made yet.")

    with tab4:
        fig = create_reward_chart(st.session_state.reward_history, theme)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, theme=None)
        else:
            st.info("Run the simulation to see reward data.")

    # ─── Score Card & Action History (Bottom) ───
    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 2])

    with col_left:
        if st.session_state.final_grade is not None:
            render_score_card(
                st.session_state.final_grade,
                grade_label(st.session_state.final_grade)
            )
        elif st.session_state.reward_history:
            # Show live running score
            live_score = grade_agent(state, task_type=st.session_state.task_goal)
            render_score_card(live_score, "🔄 Live Score (in progress)")

    with col_right:
        # Action distribution pie chart
        if st.session_state.action_history:
            from collections import Counter
            counts = Counter(st.session_state.action_history)
            colors = {
                "work_overtime": "#3b82f6", "exercise": "#10b981",
                "invest_money": "#f59e0b", "learn_skill": "#8b5cf6",
                "socialize": "#ec4899", "rest": "#6366f1",
            }
            fig = go.Figure(data=[go.Pie(
                labels=[k.replace("_", " ").title() for k in counts.keys()],
                values=list(counts.values()),
                hole=0.55,
                marker=dict(colors=[colors.get(k, "#667eea") for k in counts.keys()]),
                textinfo="percent+label",
                textfont=dict(size=11, color="white"),
            )])
            bg = "rgba(10,10,26,0)" if theme == "dark" else "rgba(240,242,245,0)"
            fig.update_layout(
                plot_bgcolor=bg, paper_bgcolor=bg,
                font=dict(color="#a8a8c8" if theme == "dark" else "#4a4a6a",
                          family="Inter"),
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
                height=280,
                title=dict(text="Action Distribution", font=dict(size=14)),
            )
            st.plotly_chart(fig, use_container_width=True,
                          config={"displayModeBar": False}, theme=None)

    # Done message
    if st.session_state.sim_done:
        st.markdown(f"""
        <div style="text-align:center;padding:20px;margin-top:20px;
             background:{'rgba(25,25,50,0.5)' if theme=='dark' else 'rgba(255,255,255,0.5)'};
             backdrop-filter:blur(16px);border-radius:16px;
             border:1px solid {'rgba(100,100,200,0.2)' if theme=='dark' else 'rgba(0,0,0,0.08)'};">
            <div style="font-size:2rem;">🏁</div>
            <div style="font-size:1.2rem;font-weight:700;color:{'#e8e8f0' if theme=='dark' else '#1a1a2e'};">
                Simulation Complete!</div>
            <div style="color:{'#a8a8c8' if theme=='dark' else '#4a4a6a'};margin-top:4px;">
                {st.session_state.current_step} steps · Final Grade: {st.session_state.final_grade:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Auto-play loop
    if st.session_state.auto_play and not st.session_state.sim_done:
        time.sleep(st.session_state.play_speed)
        do_step()
        st.rerun()


def render_welcome(theme: str):
    """Render the welcome screen when no simulation is running."""
    bg = "rgba(25,25,50,0.4)" if theme == "dark" else "rgba(255,255,255,0.4)"
    text = "#e8e8f0" if theme == "dark" else "#1a1a2e"
    muted = "#a8a8c8" if theme == "dark" else "#6a6a8a"
    border = "rgba(100,100,200,0.15)" if theme == "dark" else "rgba(0,0,0,0.08)"

    st.markdown(f"""
    <div style="text-align:center;padding:60px 20px;
         background:{bg};backdrop-filter:blur(20px);
         border-radius:24px;border:1px solid {border};margin-top:20px;
         cursor: default;">
        <div style="font-size:4rem;margin-bottom:16px;cursor: default;">🧬</div>
        <div style="font-size:1.6rem;font-weight:700;color:{text};margin-bottom:8px;cursor: pointer;">
            Welcome to the AI Digital Life Simulator</div>
        <div style="font-size:1rem;color:{muted};max-width:500px;margin:0 auto 24px;cursor: pointer;">
            Navigate life's toughest decisions. Balance health, career, relationships,
            and finances. How well can you live?</div>
        <div style="display:flex;justify-content:center;gap:24px;flex-wrap:wrap;margin-top:30px;">
            <div style="background:{'rgba(16,185,129,0.15)' if theme=='dark' else 'rgba(16,185,129,0.1)'};
                 border-radius:16px;padding:20px;width:140px;">
                <div style="font-size:2rem;">❤️</div>
                <div style="font-size:0.85rem;color:{muted};margin-top:4px;">Health</div>
            </div>
            <div style="background:{'rgba(59,130,246,0.15)' if theme=='dark' else 'rgba(59,130,246,0.1)'};
                 border-radius:16px;padding:20px;width:140px;">
                <div style="font-size:2rem;">💼</div>
                <div style="font-size:0.85rem;color:{muted};margin-top:4px;">Career</div>
            </div>
            <div style="background:{'rgba(236,72,153,0.15)' if theme=='dark' else 'rgba(236,72,153,0.1)'};
                 border-radius:16px;padding:20px;width:140px;">
                <div style="font-size:2rem;">👥</div>
                <div style="font-size:0.85rem;color:{muted};margin-top:4px;">Relationships</div>
            </div>
            <div style="background:{'rgba(245,158,11,0.15)' if theme=='dark' else 'rgba(245,158,11,0.1)'};
                 border-radius:16px;padding:20px;width:140px;">
                <div style="font-size:2rem;">💰</div>
                <div style="font-size:0.85rem;color:{muted};margin-top:4px;">Wealth</div>
            </div>
        </div>
        <div style="margin-top:30px;font-size:0.9rem;color:{muted};">
            👈 Configure and press <b>Initialize</b> in the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)


def render_registration(theme: str):
    """Render the user registration screen."""
    bg = "rgba(25,25,50,0.4)" if theme == "dark" else "rgba(255,255,255,0.4)"
    text = "#e8e8f0" if theme == "dark" else "#1a1a2e"
    muted = "#a8a8c8" if theme == "dark" else "#6a6a8a"
    border = "rgba(100,100,200,0.15)" if theme == "dark" else "rgba(0,0,0,0.08)"

    logo_path = Path(__file__).parent / "logo.png"
    if logo_path.exists():
        import base64
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{b64}" style="width:140px; border-radius:20px; margin-bottom:16px; box-shadow: 0 8px 25px rgba(0,0,0,0.3);">'
    else:
        logo_html = '<div style="font-size:3.5rem;margin-bottom:16px;">📝</div>'

    st.markdown(f"""
    <div style="text-align:center;padding:40px 20px 20px 20px;
         background:{bg};backdrop-filter:blur(20px);
         border-radius:24px;border:1px solid {border};margin-top:40px;
         max-width: 500px; margin-left: auto; margin-right: auto; margin-bottom: 20px;
         cursor: default;">
        {logo_html}
        <div style="font-size:1.6rem;font-weight:700;color:{text};margin-bottom:8px;cursor: pointer;">
            Player Registration</div>
        <div style="font-size:0.95rem;color:{muted};margin-bottom:10px;cursor: pointer;">
            Enter your details to create your digital life profile.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown("<br>", unsafe_allow_html=True)
            user_name = st.text_input("Name", placeholder="Enter your name...")
            user_career = st.text_input("Dream Career", placeholder="e.g. Software Engineer...")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Enter Simulator 🚀", use_container_width=True):
                if user_name.strip() == "":
                    st.error("Please enter your name to continue.")
                else:
                    st.session_state.registered = True
                    st.session_state.user_name = user_name.strip()
                    st.session_state.user_career = user_career.strip()
                    st.rerun()

# ═══════════════════════════════════════════════
#  MAIN ENTRY
# ═══════════════════════════════════════════════

def main():
    init_session()
    load_css()
    
    theme = st.session_state.get("theme", "dark")
    
    if not st.session_state.get("registered", False):
        render_registration(theme)
    else:
        render_sidebar()
        render_main()


if __name__ == "__main__":
    main()
