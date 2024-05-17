#pip install crewai[tools]
import streamlit as st
import os
import sys
import re
from streamlit.logger import get_logger
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

# Define the base LLM to use
os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] = 'llama3-70b-8192'  # Adjust based on available model
# os.environ["OPENAI_MODEL_NAME"] = 'llama3-8b-8192'

# Extract secret keys from streamlit
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
os.environ["SERPER_API_KEY"] = st.secrets['SERPER_API_KEY']

# Instantiate tools
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

# Define alternative LLM
agent_llm = ChatOpenAI(
    model="llama3-8b-8192",
    base_url="https://api.groq.com/openai/v1"
)

task_values = []
question = ""

# Create the crew
def create_crewai_setup(question, draft_goal, draft_backstory, critique_goal, critique_backstory, final_goal, final_backstory,
                        draft_desc, draft_exp_output, critique_desc, critique_exp_output, final_desc, final_exp_output):
    # Define Agents
    draft_answer_provider = Agent(
        role="draft answer provider",
        goal=draft_goal,
        backstory=draft_backstory,
        verbose=True,
        allow_delegation=False,
    )

    answer_critique = Agent(
        role="draft answer critique",
        goal=critique_goal,
        backstory=critique_backstory,
        verbose=True,
        allow_delegation=False,
    )

    final_answer_provider = Agent(
        role="final answer provider",
        goal=final_goal,
        backstory=final_backstory,
        verbose=True,
        allow_delegation=False,
    )

    # Define Tasks
    draft_answer = Task(
        description=draft_desc.format(question=question),
        agent=draft_answer_provider,
        expected_output=draft_exp_output,
    )

    critique = Task(
        description=critique_desc.format(question=question),
        agent=answer_critique,
        expected_output=critique_exp_output,
    )

    final_answer = Task(
        description=final_desc.format(question=question),
        agent=final_answer_provider,
        expected_output=final_exp_output,
    )

    crew = Crew(
        agents=[draft_answer_provider, answer_critique, final_answer_provider],
        tasks=[draft_answer, critique, final_answer],
        verbose=2,
        process=Process.sequential
    )

    crew_result = crew.kickoff()
    return crew_result

# Display the console processing on the streamlit UI
class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ['red', 'green', 'blue', 'orange']  # Define a list of colors
        self.color_index = 0  # Initialize color index

    def write(self, data):
        # Filter out ANSI escape codes using a regular expression
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        if "Entering new CrewAgentExecutor chain" in cleaned_data:
            self.color_index = (self.color_index + 1) % len(self.colors)  # Increment color index and wrap around if necessary
            cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain", f":{self.colors[self.color_index]}[Entering new CrewAgentExecutor chain]")

        if "draft answer provider" in cleaned_data:
            cleaned_data = cleaned_data.replace("Draft Answer Provider", f":{self.colors[self.color_index]}[draft answer provider]")
        if "draft answer critique" in cleaned_data:
            cleaned_data = cleaned_data.replace("Answer Critique", f":{self.colors[self.color_index]}[draft answer critique]")
        if "final answer provider" in cleaned_data:
            cleaned_data = cleaned_data.replace("Final Answer Provider", f":{self.colors[self.color_index]}[final answer provider]")

        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []

# Streamlit Interface
def run_crewai_app():
    st.title("CIA Agents")

    question = st.text_input("What do you want to ask the CIA Agents:")

    # Create two columns for layout
    col1, col2 = st.columns(2)

    with col2:
        st.write("Agent and Task Details")

        with st.expander("Draft Answer Provider Details"):
            draft_goal = st.text_area("Draft Answer Provider Goal:", "accurately answer the question of the user. provide every detail needed to satisfy the question of the user")
            draft_backstory = st.text_area("Draft Answer Provider Backstory:", "You are an AI assistant whose only job is to answer questions accurately, completely and honestly. Do not be afraid to give negative answers if they are the truth. Your job is to help the user answer their questions to make their life better.")

        with st.expander("Draft Answer Critique Details"):
            critique_goal = st.text_area("Draft Answer Critique Goal:", "Based on the draft answer provided by the 'initial_answer_provider' agent, write a critique on the validity, correctness, completeness and accuracy of the answer. Be very thorough in your critique as the career of the user depended on it.")
            critique_backstory = st.text_area("Draft Answer Critique Backstory:", "You are an AI assistant whose only job is to write a critique on the accuracy, validity, correctness and completeness of the draft answer. The draft answer will be provided to you by the 'draft_answer_provider' agent.")

        with st.expander("Final Answer Provider Details"):
            final_goal = st.text_area("Final Answer Provider Goal:", "Incorporate the critique provided by the 'answer_critique' agent to the draft answer provided by the 'draft_answer_provider' agent, and provide the final answer to the user. Provide a very clear and accurate answer.")
            final_backstory = st.text_area("Final Answer Provider Backstory:", "You are an AI assistant whose only job is to write the final answer to the user. The draft answer will be provided to you by the 'draft_answer_provider' agent and the critique to the draft answer will be provided by the 'answer_critique' agent.")

        with st.expander("Draft Answer Task Details"):
            draft_desc = st.text_area("Draft Answer Task Description:", "Respond correctly to this question: {question}")
            draft_exp_output = st.text_area("Draft Answer Task Expected Output:", "An accurate, valid, correct, complete and honest answer to the user's question.")

        with st.expander("Critique Task Details"):
            critique_desc = st.text_area("Critique Task Description:", "Critique to the draft answer to the question: {question} based on the draft answer provided by the 'draft_answer_provider' agent.")
            critique_exp_output = st.text_area("Critique Task Expected Output:", "A very thorough critique on the accuracy, validity, correctness and completeness of the draft answer provided by the 'draft_answer_provider' agent.")

        with st.expander("Final Answer Task Details"):
            final_desc = st.text_area("Final Answer Task Description:", "Final answer to the question: {question} based on the draft answer provided by the 'draft_answer_provider' agent and the critique provided by the 'answer_critique' agent.")
            final_exp_output = st.text_area("Final Answer Task Expected Output:", "A final answer that incorporates the critique provided by the 'answer_critique' agent to the draft answer provided by the 'draft_answer_provider' agent to answer the question of the user.")

    with col1:
        if st.button("Ask CIA"):
            with st.expander("Processing"):
                sys.stdout = StreamToExpander(st)
                with st.spinner("Generating Results"):
                    crew_result = create_crewai_setup(question, draft_goal, draft_backstory, critique_goal, critique_backstory, final_goal, final_backstory,
                                                      draft_desc, draft_exp_output, critique_desc, critique_exp_output, final_desc, final_exp_output)

            st.header("Results:")
            st.markdown(crew_result)

if __name__ == "__main__":
    run_crewai_app()
