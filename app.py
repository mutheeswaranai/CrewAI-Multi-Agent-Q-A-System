import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# ==========================================
# 1. Custom Tool for the Entry Agent
# ==========================================
@tool("File Saver Tool")
def file_saver_tool(content: str) -> str:
    """Saves the provided text content to a file named 'query_results.txt'."""
    try:
        with open("query_results.txt", "w", encoding="utf-8") as f:
            f.write(content)
        return "Successfully saved all data to query_results.txt"
    except Exception as e:
        return f"Failed to save file: {str(e)}"

# ==========================================
# 2. Helper to Extract Task Output safely
# ==========================================
def get_task_output(task):
    if not task.output:
        return "No output available."
    # Handles both recent CrewAI TaskOutput objects and fallback string representations
    return getattr(task.output, 'raw', str(task.output))

# ==========================================
# 3. Main Streamlit App
# ==========================================
def main():
    st.set_page_config(page_title="CrewAI Sequential Q&A", layout="wide")
    st.title("CrewAI Multi-Agent Q&A System")
    st.write("This app uses a sequential process with three agents: General Assistant, Web Researcher, and Entry Agent.")

    # Validate API Key from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("Missing OPENAI_API_KEY in environment variables. Please set it before running the app.")
        st.stop()

    query = st.text_input("Enter your question or task:")

    if st.button("Run Agents") and query:
        with st.spinner("Agents are working..."):
            
            # --- AGENT DEFINITIONS ---
            assistant_agent = Agent(
                role="General Knowledge Assistant",
                goal="Provide a clear, comprehensive, and well-structured baseline answer to the user's query using foundational knowledge.",
                backstory="You are a helpful, highly capable AI assistant trained on vast amounts of data. You pride yourself on explaining complex concepts clearly and concisely without needing external resources.",
                allow_delegation=False,
                verbose=True
            )

            web_search_agent = Agent(
                role="Web Search Specialist",
                goal="Search for and provide an updated, detailed, and factual perspective on the user's query to complement the Assistant's knowledge.",
                backstory="You are an expert internet researcher who scours the web for the most accurate, up-to-date information. You synthesize external data into easily readable, highly informative insights.",
                allow_delegation=False,
                verbose=True
            )

            entry_agent = Agent(
                role="Data Entry & Archival Specialist",
                goal="Compile the original query, the Assistant's answer, and the Researcher's answer into a cohesive document and save it locally to a .txt file.",
                backstory="You are a meticulous data entry clerk and archiver. Your sole purpose is to ensure that no valuable AI-generated insights are lost, diligently formatting and saving interaction logs securely to the disk.",
                tools=[file_saver_tool],
                allow_delegation=False,
                verbose=True
            )

            # --- TASK DEFINITIONS ---
            task1_assistant = Task(
                description=f"Answer the following query based on your general knowledge: '{query}'",
                expected_output="A clear, well-structured baseline answer to the user's query.",
                agent=assistant_agent
            )

            task2_researcher = Task(
                description=f"Provide an advanced, simulated web-research perspective on this query: '{query}'",
                expected_output="A detailed, research-style factual answer to the user's query.",
                agent=web_search_agent
            )

            task3_entry = Task(
                description=(
                    f"You have the original user query: '{query}'.\n"
                    "You also have the output from the General Knowledge Assistant and the Web Search Specialist.\n"
                    "1. Compile everything into a single, clearly formatted string (Query, Assistant Answer, Web Search Answer).\n"
                    "2. Use the 'File Saver Tool' to write this compiled string to disk.\n"
                    "Return a summary confirming the save was successful."
                ),
                expected_output="Confirmation that the compiled string was written to the text file.",
                agent=entry_agent,
                context=[task1_assistant, task2_researcher]
            )

            # --- CREW SETUP & EXECUTION ---
            crew = Crew(
                agents=[assistant_agent, web_search_agent, entry_agent],
                tasks=[task1_assistant, task2_researcher, task3_entry],
                process=Process.sequential,
                verbose=True
            )

            # Kickoff the sequential process
            final_result = crew.kickoff()

            # --- DISPLAY RESULTS ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🤖 Assistant Answer")
                st.write(get_task_output(task1_assistant))
                
            with col2:
                st.subheader("🌐 Web Search Answer")
                st.write(get_task_output(task2_researcher))

            st.divider()
            st.subheader("💾 Archiver Status")
            st.info(get_task_output(task3_entry))
            st.success("Check your current directory for 'query_results.txt'!")

if __name__ == "__main__":
    main()