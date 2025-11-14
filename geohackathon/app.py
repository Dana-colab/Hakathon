#!/usr/bin/env python3
"""
Streamlit Chatbot for Well Analysis RAG Pipeline
Deploy with: streamlit run app.py
"""

import streamlit as st
import os
import tempfile
from datetime import datetime

# Import your existing pipeline
from well_rag_pipeline import (
    WellAnalysisAgent,
    extract_well_parameters,
    extract_nodal_inputs,
    calculate_nodal_analysis,
)

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------


def format_analysis_results(report):
    """Format analysis results for chat display."""
    msg = "✅ **Analysis Complete!**\n\n"

    # Basic info
    params = report.get("extracted_parameters", {})
    if params.get("well_name"):
        msg += f"**Well:** {params['well_name']}\n\n"

    # Nodal results
    nodal = report.get("nodal_analysis_results", {})
    if nodal.get("status") == "success":
        results = nodal["results"]
        op = results["operating_point"]
        prod = results["productivity"]

        msg += "### ⚙️ Nodal Analysis Results\n\n"
        msg += f"- **Flow Rate:** {op['flow_rate_m3_h']} m³/h\n"
        msg += f"- **Wellhead Pressure:** {op['wellhead_pressure_bar']} bar\n"
        msg += f"- **Bottomhole Pressure:** {op['bottomhole_pressure_bar']} bar\n"
        msg += f"- **Max Flow Potential:** {prod['max_flow_rate_m3_h']} m³/h\n"
        msg += f"- **Current Utilization:** {prod['current_utilization_pct']}%\n\n"

    # Summary
    summary = report.get("summary", "")
    if summary:
        msg += f"### 📝 Summary\n\n{summary}\n\n"

    msg += "💬 *Ask me questions about the results or request specific details!*"

    return msg


def format_nodal_details(results):
    """Format detailed nodal analysis results."""
    op = results["operating_point"]
    pa = results["pressure_analysis"]
    fc = results["flow_characteristics"]
    prod = results["productivity"]

    msg = "### ⚙️ Detailed Nodal Analysis\n\n"

    msg += "**Operating Point:**\n"
    msg += f"- Flow Rate: {op['flow_rate_m3_h']} m³/h\n"
    msg += f"- Wellhead Pressure: {op['wellhead_pressure_bar']} bar\n"
    msg += f"- Bottomhole Pressure: {op['bottomhole_pressure_bar']} bar\n"
    msg += f"- Reservoir Pressure: {op['reservoir_pressure_bar']} bar\n\n"

    msg += "**Pressure Analysis:**\n"
    msg += f"- Hydrostatic Drop: {pa['hydrostatic_pressure_drop_bar']} bar\n"
    msg += f"- Friction Drop: {pa['friction_pressure_drop_bar']} bar\n"
    msg += f"- Total Drop: {pa['total_pressure_drop_bar']} bar\n\n"

    msg += "**Flow Characteristics:**\n"
    msg += f"- Reynolds Number: {fc['reynolds_number']}\n"
    msg += f"- Flow Regime: {fc['flow_regime']}\n"
    msg += f"- Friction Factor: {fc['friction_factor']}\n"
    msg += f"- Velocity: {fc['velocity_m_s']} m/s\n\n"

    msg += "**Productivity:**\n"
    msg += f"- Productivity Index: {prod['productivity_index_m3h_bar']} m³/h/bar\n"
    msg += f"- Maximum Flow: {prod['max_flow_rate_m3_h']} m³/h\n"
    msg += f"- Current Utilization: {prod['current_utilization_pct']}%\n"

    return msg


def generate_optimization_advice(results):
    """Generate optimization recommendations."""
    prod = results["productivity"]
    utilization = prod["current_utilization_pct"]

    msg = "### 💡 Production Optimization Recommendations\n\n"
    msg += f"**Current Status:** Operating at {utilization}% of maximum potential\n\n"

    if utilization < 50:
        msg += "🚀 **High optimization potential!**\n\n"
        msg += "**Recommendations:**\n"
        msg += "1. **Increase ESP frequency** - Could boost production significantly\n"
        msg += "2. **Reduce wellhead backpressure** - Check surface facilities\n"
        msg += "3. **Review choke settings** - May be restricting flow\n"
        msg += (
            f"4. **Potential gain:** Up to "
            f"{prod['max_flow_rate_m3_h'] - results['operating_point']['flow_rate_m3_h']:.1f} m³/h\n"
        )
    elif utilization < 75:
        msg += "📈 **Moderate optimization potential**\n\n"
        msg += "**Recommendations:**\n"
        msg += "1. **Fine-tune ESP settings** - Gradual frequency increase\n"
        msg += "2. **Monitor reservoir pressure** - Ensure adequate drive\n"
        msg += "3. **Optimize artificial lift** - Balance power vs production\n"
    else:
        msg += "✅ **Well is operating efficiently!**\n\n"
        msg += "Current utilization is good. Focus on:\n"
        msg += "1. **Maintain current settings** - Don't over-produce\n"
        msg += "2. **Monitor for decline** - Track performance over time\n"
        msg += "3. **Prevent equipment damage** - Operating near capacity\n"

    return msg


def identify_limitations(results):
    """Identify production limitations."""
    pa = results["pressure_analysis"]
    fc = results["flow_characteristics"]

    msg = "### 🔍 Production Limitation Analysis\n\n"

    total_drop = pa["total_pressure_drop_bar"]
    if total_drop == 0:
        hydrostatic_pct = 0.0
        friction_pct = 0.0
    else:
        hydrostatic_pct = (pa["hydrostatic_pressure_drop_bar"] / total_drop) * 100
        friction_pct = (pa["friction_pressure_drop_bar"] / total_drop) * 100

    msg += f"**Pressure Drop Breakdown:**\n"
    msg += f"- Hydrostatic: {hydrostatic_pct:.1f}% ({pa['hydrostatic_pressure_drop_bar']} bar)\n"
    msg += f"- Friction: {friction_pct:.1f}% ({pa['friction_pressure_drop_bar']} bar)\n\n"

    msg += "**Main Limiting Factors:**\n\n"

    if friction_pct > 10:
        msg += "⚠️ **High friction losses** - Consider:\n"
        msg += "- Larger tubing diameter\n"
        msg += "- Scale/wax treatment\n"
        msg += "- Flow regime optimization\n\n"

    msg += "🔹 **Hydrostatic head** - Natural limitation due to well depth\n"
    msg += "🔹 **Wellhead pressure** - Surface equipment backpressure\n"
    msg += f"🔹 **Flow regime** - Currently {fc['flow_regime'].lower()}\n\n"

    msg += "*The friction losses indicate tubing efficiency. Lower is better!*"

    return msg


def generate_response(user_input, report):
    """Generate chatbot response based on user input and current report."""
    lower_input = user_input.lower()

    # No document uploaded
    if not report:
        if any(word in lower_input for word in ["upload", "how", "start"]):
            return """To get started:
            
1. Click the **Browse files** button in the sidebar  
2. Select a PDF well completion report  
3. Click **🚀 Analyze Document**  
4. Wait for the analysis to complete  
5. Ask me questions about the results!

I support standard well completion reports with tables and technical data."""
        elif any(word in lower_input for word in ["extract", "parameter"]):
            return """I can extract many parameters from well documents:

**Basic Info:** Well name, operation type, dates, duration  
**Depths:** Packer, PBR, pump intake, total depth  
**Equipment:** Tubing size, ESP details, completion config  
**Reservoir:** Temperature, fluid type, pressure  
**Production:** Flow rates, pressures, fluid properties  
**Safety:** HSE incidents, operational notes  

Upload a document to see what I can extract! 📄"""
        elif any(word in lower_input for word in ["nodal", "analysis", "calculate"]):
            return """**Nodal Analysis** determines well production capacity by analyzing pressure relationships.

I calculate:  
- **Pressure Distribution:** From reservoir to wellhead  
- **Flow Characteristics:** Reynolds number, flow regime, friction  
- **Productivity:** Current vs maximum potential  
- **Bottlenecks:** What's limiting production  

Upload a document and I'll perform the analysis automatically! ⚙️"""
        else:
            return (
                f"""I understand you're asking: *"{user_input}"*\n\n"""
                "Please upload a PDF document first so I can help you analyze it! Use the sidebar to upload. 📤"
            )

    # Document is loaded
    if any(word in lower_input for word in ["parameter", "extract", "data"]):
        params = report["extracted_parameters"]
        msg = "### 📋 Extracted Parameters\n\n"
        for key, value in params.items():
            if value:
                msg += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        return msg

    elif any(word in lower_input for word in ["nodal", "pressure", "flow"]):
        nodal = report["nodal_analysis_results"]
        if nodal["status"] == "success":
            results = nodal["results"]
            return format_nodal_details(results)
        else:
            return f"⚠️ Nodal analysis was incomplete: {nodal['message']}"

    elif any(word in lower_input for word in ["summary", "overview"]):
        return f"### 📝 Document Summary\n\n{report['summary']}"

    elif any(word in lower_input for word in ["increase", "optimize", "improve"]):
        nodal = report["nodal_analysis_results"]
        if nodal["status"] == "success":
            return generate_optimization_advice(nodal["results"])
        return "I need successful nodal analysis results to provide optimization advice."

    elif any(word in lower_input for word in ["limit", "bottleneck", "problem"]):
        nodal = report["nodal_analysis_results"]
        if nodal["status"] == "success":
            return identify_limitations(nodal["results"])
        return "I need successful nodal analysis results to identify limitations."

    else:
        return (
            f"""I understand you're asking: *"{user_input}"*\n\n"""
            "I can help you with:\n"
            "- Show extracted parameters\n"
            "- Explain nodal analysis results\n"
            "- Provide optimization suggestions\n"
            "- Identify production limitations\n"
            "- Export the full report\n\n"
            'Try asking: **"What are the nodal results?"** or **"How can we optimize production?"** 💡'
        )


def show_parameters():
    """Display parameters in a structured format."""
    if st.session_state.current_report:
        params = st.session_state.current_report["extracted_parameters"]

        st.subheader("📋 Extracted Parameters")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Basic Information**")
            for key in ["well_name", "operation", "start_date", "duration"]:
                if params.get(key):
                    st.text(f"{key.replace('_', ' ').title()}: {params[key]}")

        with col2:
            st.markdown("**Technical Data**")
            for key in [
                "packer_depth_m",
                "tubing_size",
                "reservoir_temp_c",
                "flow_rate_m3h",
            ]:
                if params.get(key):
                    st.text(f"{key.replace('_', ' ').title()}: {params[key]}")


def show_nodal_results():
    """Display nodal analysis results with metrics."""
    if st.session_state.current_report:
        nodal = st.session_state.current_report["nodal_analysis_results"]

        if nodal["status"] == "success":
            results = nodal["results"]
            op = results["operating_point"]
            prod = results["productivity"]

            st.subheader("⚙️ Nodal Analysis Results")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Flow Rate", f"{op['flow_rate_m3_h']} m³/h")

            with col2:
                st.metric("WHP", f"{op['wellhead_pressure_bar']} bar")

            with col3:
                st.metric("BHP", f"{op['bottomhole_pressure_bar']} bar")

            with col4:
                st.metric("Utilization", f"{prod['current_utilization_pct']}%")


def show_summary():
    """Display summary."""
    if st.session_state.current_report:
        summary = st.session_state.current_report["summary"]
        st.subheader("📝 Executive Summary")
        st.write(summary)


def download_report():
    """Allow user to download the full JSON report."""
    if st.session_state.current_report:
        import json

        report_json = json.dumps(st.session_state.current_report, indent=2)

        st.download_button(
            label="📥 Download Full Report (JSON)",
            data=report_json,
            file_name=f"well_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


# --------------------------------------------------------------------
# Streamlit Page setup
# --------------------------------------------------------------------

st.set_page_config(
    page_title="Well Analysis Assistant",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #667eea;
        color: white;
    }
    .bot-message {
        background-color: #f0f2f6;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """👋 **Hello! I'm your Well Analysis Assistant.**

I can help you with:
- 📄 Analyzing well completion reports (PDF)  
- 🔍 Extracting parameters from documents  
- ⚙️ Performing automated nodal analysis  
- 📊 Generating summaries and reports  
- ❓ Answering questions about your wells  

Upload a PDF to get started!""",
        }
    ]

if "current_report" not in st.session_state:
    st.session_state.current_report = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# Header
st.markdown(
    '<h1 class="main-header">🛢️ Well Analysis Assistant</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align: center; color: #666;">RAG-powered nodal analysis chatbot</p>',
    unsafe_allow_html=True,
)

# Sidebar for file upload and settings
with st.sidebar:
    st.header("📁 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload Well Report (PDF)",
        type=["pdf"],
        help="Upload a well completion report for analysis",
    )

    if uploaded_file:
        st.session_state.uploaded_file_name = uploaded_file.name
        st.success(f"✅ {uploaded_file.name} uploaded!")

        # Settings
        st.header("⚙️ Settings")
        word_limit = st.slider("Summary word limit", 100, 500, 250, 50)

        # Analysis button
        if st.button("🚀 Analyze Document", type="primary"):
            with st.spinner("Analyzing document... This may take a minute."):
                try:
                    # Save uploaded file in a safe /tmp directory (required on HuggingFace Spaces)
                    with tempfile.NamedTemporaryFile(
                        dir="/tmp", delete=False, suffix=".pdf"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # Run analysis
                    agent = WellAnalysisAgent(tmp_path, word_limit=word_limit)
                    report = agent.run()

                    # Clean up temp file
                    os.unlink(tmp_path)

                    # Store report
                    st.session_state.current_report = report

                    # Add success message to chat
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": f"Analyze {uploaded_file.name}",
                        }
                    )

                    # Format results message
                    results_msg = format_analysis_results(report)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": results_msg}
                    )

                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")

    # Quick actions
    if st.session_state.current_report:
        st.header("⚡ Quick Actions")

        if st.button("📋 Show Parameters"):
            show_parameters()

        if st.button("📊 Show Nodal Results"):
            show_nodal_results()

        if st.button("📝 Show Summary"):
            show_summary()

        if st.button("💾 Download Report"):
            download_report()

    # Info
    st.markdown("---")
    st.markdown("### 💡 How to Use")
    st.markdown(
        """
1. Upload a PDF document  
2. Click **'Analyze Document'**  
3. Ask questions in the chat  
4. View results and download  
"""
    )

# Main chat interface
chat_container = st.container()

with chat_container:
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            st.markdown(
                f'<div class="chat-message user-message">👤 **You:** {content}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-message bot-message">🤖 **Assistant:**\n\n{content}</div>',
                unsafe_allow_html=True,
            )

# Chat input
user_input = st.chat_input("Ask me anything about well analysis...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response
    response = generate_response(user_input, st.session_state.current_report)

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()

if __name__ == "__main__":
    st.write("Run with: streamlit run app.py")
