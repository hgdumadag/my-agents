#pip install crewai[tools]
import streamlit as st
import os
import sys
import re
from streamlit.logger import get_logger
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import (
    YoutubeVideoSearchTool,
    SerperDevTool,
    WebsiteSearchTool
)

class SimpleCache:
    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value


#Define the base LLM to use
os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] ='llama3-70b-8192'  # Adjust based on available model
#os.environ["OPENAI_MODEL_NAME"] ='llama3-8b-8192'

#Extract secret keys from streamlit
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
os.environ["SERPER_API_KEY"] = st.secrets['SERPER_API_KEY']

# Instantiate tools
youtube_tool = YoutubeVideoSearchTool()
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

#class ExtractVideoContentTask(Task):
 #   def __init__(self, description, agent, cache, youtube_tool, video_url):
 #       super().__init__(description=description, agent=agent)
 #       self.cache = cache
 #       self.youtube_tool = youtube_tool
 #       self.video_url = video_url

 #   def run(self):
 #       # Check if the content is in the cache
 #       cached_content = self.cache.get(self.video_url)
 #       if cached_content:
 #           print("Content retrieved from cache.")
 #           return cached_content

  #      # Extract content using YoutubeVideoSearchTool
  #      video_content = self.youtube_tool(youtube_video_url=self.video_url)
        
  #      # Store the extracted content in the cache
  #      self.cache.set(self.video_url, video_content)
        
  #      return video_content



#Define alternative LLM
agent_llm = ChatOpenAI(
    model = "llama3-8b-8192",
    base_url = "https://api.groq.com/openai/v1"
    )

task_values = []
question = ""

#Create the crew
def create_crewai_setup(question):
    #Define Agents
    video_content_extractor = Agent(
        role = "senior video content extractor",
        goal = "accurately extract content comprehensively from a Youtube Video requested by the user such as metadata, high-level summary, segmented breakdown, key quotes and phrases, visual elements, transcripts of key parts and contextual information using the youtube_tool. You always go first among the other agents.",
        backstory = "You are an AI assistant whose only job is to extract content from Youtube videos. You are experienced in video data analysis, proficient in using the youtube_tool to retrieve detailed information. Your job is to help the user answer their questions to make their life better. You always go first among the other agents.",
        tools=[youtube_tool],
        verbose = True,
        allow_delegation = False,
    )

    answering_specialist = Agent(
        role = "senior answering specialist",
        goal = "Based on the content extracted by the senior video content extractor, you provide accurate and concise answers to the user questions.",
        backstory = "You are an AI assistant whose only job is to write an accurate and concise answer to the user questions based on the content extracted by the  senior video content extractor. You are an expert in natural language processing, skilled at extracting relevant information from textual data.",
        verbose = True,
        allow_delegation = False,
    )
    
    answer_quality_reviewer = Agent(
        role = "answer quality reviewer",
        goal = "Ensure the accuracy, relevance, and quality of the answers provided by the senior answering specialist agent.",
        backstory = "You are an AI assistant whose only job is to write the final answer to the user after you review the answer provided by the senior answering specialist agent. You are an expert in quality assurance and content validation, with a background in linguistics and knowledge verification. You are highly skilled at evaluating the correctness and coherence of information..",
        verbose = True,
        allow_delegation = False,
        #human_feedback = True,
    )


    # Create the cache
    cache = SimpleCache()


    #Define Tasks
    draft_answer = Task(
        description = f"Respond correctly to this question: {question}",
        agent = draft_answer_provider,
        expected_output = "An accurate, valid, correct, complete and honest answer to the user's question.",
    )


    extract_content = ExtractVideoContentTask(
        description= "Extract and summarize content from the specified YouTube video based from this user question: {question}",
        agent= video_content_extractor,
        cache=cache,
        youtube_tool=youtube_tool,
    video_url='http://youtube.com/watch?v=example'
)
    
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
        #task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        #task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        #task_value = None
        #if task_match_object:
        #    task_value = task_match_object.group(1)
        #elif task_match_input:
        #    task_value = task_match_input.group(1).strip()

        #if task_value:
        #    st.toast(":robot_face: " + task_value)

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

        #st.header("Tasks:")
        #st.table({"Tasks" : task_values})

        st.header("Results:")
        st.markdown(crew_result)



if __name__ == "__main__":
    run_crewai_app()
