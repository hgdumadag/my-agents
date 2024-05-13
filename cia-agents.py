#pip install crewai[tools]
import streamlit as st
import os
from streamlit.logger import get_logger
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

#Define the base LLM to use
os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] ='llama3-70b-8192'  # Adjust based on available model
#os.environ["OPENAI_MODEL_NAME"] ='llama3-8b-8192'

#Extract secret keys from streamlit
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
os.environ["SERPER_API_KEY"] = st.secrets['SERPER_API_KEY']

# Instantiate tools
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

#Define alternative LLM
agent_llm = ChatOpenAI(
    model = "llama3-8b-8192",
    base_url = "https://api.groq.com/openai/v1"
    )

question = ""

#Create the crew
def create_crewai_setup(question):
    #Define Agents
    draft_answer_provider = Agent(
        role = "draft answer provider",
        goal = "accurately answer the question of the user. provide every detail needed to satisfy the question of the user",
        backstory = "You are an AI assistant whose only job is to answer questions accurately, completely and honestly. Do not be afraid to give negative answer if they are the truth. Your job is to help the user answer their questions to make their life better.",
        verbose = True,
        allow_delegation = False,
    )

    answer_critique = Agent(
        role = "draft answer critique",
        goal = "Based on the draft answer provided by the 'initial_answer_provider' agent, write a critique on the validity, correctness, completeness and accuracy of the answer. Be very thorough in your critique as the career of the user depended on it.",
        backstory = "You are an AI assistant whose only job is to write a critique on the accuracy, validity, correctness and completeness of the draft answer. The draft answer will be provided to you by the 'draft_answer_provider' agent.",
        verbose = True,
        allow_delegation = False,
    )
    ###
    #researcher = Agent(
    #    role="AI Research Specialist",
    #    goal="To conduct thorough internet-based research to validate and expand upon the draft answers provided by the draft_answer_provider and to incorporate feedback from the answer_critique. The goal includes identifying credible sources, gathering pertinent facts, and contributing insights that ensure the final answer is accurate, up-to-date, detailed, and aligned with the critique's suggestions.",
    #    backstory="Born out of a need to combat misinformation and provide only the most reliable data, I was created as a digital librarian with a twist. Equipped with advanced natural language processing skills and extensive access to a wide array of databases, my purpose is to sift through the vast ocean of information on the internet and extract only the most relevant and accurate data. My existence is dedicated to elevating the quality of information by providing well-researched contributions to any discourse, specifically tailored to enhance the accuracy and depth of responses in an AI-driven question-answer system.",
    #    tools=[search_tool, web_rag_tool],
    #    verbose=True,
    #    max_rpm=50,
    #    max_iter=20,
    #    allow_delegation=False,
    #    llm = agent_llm
    #)
    ###
    final_answer_provider = Agent(
        role = "final answer provider",
        goal = "Incorporate the critique provided by the 'answer_critique' agent to the draft answer provided by the 'draft_answer_provider' agent, and provide the final answer to the user. Provide a very clear and accurate answer.",
        backstory = "You are an AI assistant whose only job is to write the final answer to the user. The draft answer will be provided to you by the 'draft_answer_provider' agent and the critique to the draft answer will be provided by the 'answer_critique' agent.",
        verbose = True,
        allow_delegation = False,
        #human_feedback = True,
    )

    #Define Tasks
    draft_answer = Task(
        description = f"Respond correctly to this question: {question}",
        agent = draft_answer_provider,
        expected_output = "An accurate, valid, correct, complete and honest answer to the user's question.",
    )

    critique = Task(
        description = f"Critique to the draft answer to the question: {question} based on the draft answer provided by the 'draft_answer_provider' agent.",
        agent = answer_critique,
        expected_output = "a very thorough critique on the accuracy, validity, correctness and completeness of the draft answer provided by the 'draft_answer_provider' agent.",
    )

    #research_task = Task(
    #    description=f"Research and verify information related to the query: '{question}'",
    #    agent=researcher,
    #    expected_output="Top 3 results of verified facts and comprehensive data that enhance the understanding and accuracy of the response to the user's question."
    #)

    final_answer = Task(
        description = f"Final answer to the question: {question} based on the draft answer provided by the 'draft_answer_provider' agent and the critique provided by the 'answer_critique' agent.",
        agent = final_answer_provider,
        expected_output = "A final answer that incorporates the critique provided by the 'answer_critique' agent to the draft answer provided by the 'draft_answer_provider' agent to answer the question of the user.",
    )

    crew = Crew(
        #agents = [draft_answer_provider, answer_critique, researcher, final_answer_provider],
        #tasks = [draft_answer, critique, research_task, final_answer 
        agents = [draft_answer_provider, answer_critique, final_answer_provider],
        tasks = [draft_answer, critique,  final_answer],
        verbose = 2,
        process = Process.sequential
    )

    crew_result = crew.kickoff()
    return crew_result

#display the consold processing on the streamlit UI
#display the console processing on streamlit UI
class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ['red', 'green', 'blue', 'orange']  # Define a list of colors
        self.color_index = 0  # Initialize color index

    def write(self, data):
        # Filter out ANSI escape codes using a regular expression
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        # Check if the data contains 'task' information
        task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()

        if task_value:
            st.toast(":robot_face: " + task_value)

        # Check if the text contains the specified phrase and apply color
        if "Entering new CrewAgentExecutor chain" in cleaned_data:
            # Apply different color and switch color index
            self.color_index = (self.color_index + 1) % len(self.colors)  # Increment color index and wrap around if necessary

            cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain", f":{self.colors[self.color_index]}[Entering new CrewAgentExecutor chain]")

        if " draft answer provider" in cleaned_data:
            # Apply different color 
            cleaned_data = cleaned_data.replace("Draft Answer Provider", f":{self.colors[self.color_index]}[draft answer provider]")
        if "draft answer critique" in cleaned_data:
            cleaned_data = cleaned_data.replace("Answer Critique", f":{self.colors[self.color_index]}[draft answer critique]")
        if "final answer provider" in cleaned_data:
            cleaned_data = cleaned_data.replace("Final Answer Provider", f":{self.colors[self.color_index]}[final answer provider]")


        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []


#Streamlit Interface
def run_crewai_app():
    st.title("CIA Agents")

    question = st.text_input("What do you want to ask the CIA Agents:")
    
    if st.button("Ask CIA"):
        with st.expander("Processing"):
            sys.stdout = StreamToExpander(st)
            with st.spinner("Generating Results"):
                crew_result = create_crewai_setup(question)

        st.header("Tasks:")
        st.table({"Tasks" : task_values})

        st.header("Results:")
        st.markdown(crew_results)



if __name__ == "__main__":
    run_crewai_app()
