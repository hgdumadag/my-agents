#pip install crewai[tools]
import streamlit as st
from streamlit.logger import get_logger
from crewai import Agent, Task, Crew, Process

from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

import os

os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] ='llama3-70b-8192'  # Adjust based on available model
#os.environ["OPENAI_MODEL_NAME"] ='llama3-8b-8192'

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
os.environ["SERPER_API_KEY"] = st.secrets['SERPER_API_KEY']

# Instantiate tools
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

from langchain_openai import ChatOpenAI

agent_llm = ChatOpenAI(
    model = "llama3-8b-8192",
    base_url = "https://api.groq.com/openai/v1"
    )

#question = "What are the risks associated with running an airline company?"
#question = "How to conduct BCP advisory audit activity for a department that you know doesn't have BCP?"
question = ""

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

draft_answer = Task(
    description = f"Respond correctly to this question: '{question}'",
    agent = draft_answer_provider,
    expected_output = "An accurate, valid, correct, complete and honest answer to the user's question.",
)

critique = Task(
    description = f"Critique to the draft answer to the question: '{question}' based on the draft answer provided by the 'draft_answer_provider' agent.",
    agent = answer_critique,
    expected_output = "a very thorough critique on the accuracy, validity, correctness and completeness of the draft answer provided by the 'draft_answer_provider' agent.",
)

#research_task = Task(
#    description=f"Research and verify information related to the query: '{question}'",
#    agent=researcher,
#    expected_output="Top 3 results of verified facts and comprehensive data that enhance the understanding and accuracy of the response to the user's question."
#)

final_answer = Task(
    description = f"Final answer to the question: '{question}' based on the draft answer provided by the 'draft_answer_provider' agent and the critique provided by the 'answer_critique' agent.",
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



def main():
    st.title("CIA Agents")
    question = st.text_input("What do you want to ask the CIA Agents:")
    
    if st.button("Ask CIA"):
        if question:
            results = crew.kickoff(inputs={"question": question})
            #st.subheader("Draft Answer to Question:")
            #st.write(results[draft_answer_provider.role])  # Display results from the draft answer provider
            #st.subheader("Critique to Draft Answer:")
            #st.write(results[answer_critique.role])  # Display results from the critique
            st.subheader("Summary to the question: '{question}'")
            st.write(results)  # Display results from the final answer provider
        else:
            st.write("Please enter a question to proceed.")

if __name__ == "__main__":
    main()

