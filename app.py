# Math Problem Solver 
# A Streamlit application that solves math problems and searches for information using Google Gemma 2 model through Groq API

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper, DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain, LLMChain
from langchain.schema import SystemMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Math Problem Solver",
    page_icon="🧮",
    layout="wide"
)

# App title and description
st.title("🧮 Math Problem Solver")
st.markdown("""
This application solves mathematical problems and provides step-by-step explanations using Groq's AI models.
It can handle various math topics including:
- Arithmetic calculations
- Algebra
- Geometry
- Statistics
- Word problems
- And more!
""")

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# API key input - use environment variable if available
default_api_key = os.getenv("GROQ_API_KEY", "")
groq_api_key = st.sidebar.text_input(
    label="Groq API Key", 
    value=default_api_key,
    type="password",
    help="Get your API key from https://console.groq.com/keys"
)

# Model selection
model_options = {
    "gemma2-9b-it": "Gemma 2 9B (Fast)",
    "llama3-8b-8192": "Llama 3 8B (Balanced)",
    "llama3-70b-8192": "Llama 3 70B (Powerful)",
    "mixtral-8x7b-32768": "Mixtral 8x7B (Comprehensive)"
}
selected_model = st.sidebar.selectbox(
    "Select AI model:", 
    list(model_options.keys()), 
    format_func=lambda x: model_options[x],
    index=0
)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1,
                           help="Lower values make responses more deterministic, higher values more creative")
    show_reasoning = st.checkbox("Show AI reasoning steps", value=True,
                                help="Display the AI's step-by-step reasoning process")

# Initialize LLM if API key is provided
if not groq_api_key:
    st.info("Please add your Groq API key in the sidebar to continue")
    st.sidebar.info("Don't have an API key? Sign up at [console.groq.com](https://console.groq.com)")
    
    # Example problem display
    st.markdown("### Example Problem")
    st.markdown("""
    ```
    I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. 
    Then I buy a dozen apples and 2 packs of blueberries. 
    Each pack of blueberries contains 25 berries. 
    How many total pieces of fruit do I have at the end?
    ```
    """)
    st.stop()

# Initialize LLM
try:
    llm = ChatGroq(
        model=selected_model, 
        groq_api_key=groq_api_key,
        temperature=temperature,
        streaming=True
    )
except Exception as e:
    st.error(f"Error initializing the AI model: {str(e)}")
    st.stop()

# Initialize tools
def setup_tools():
    """Initialize and return tools for the agent"""
    
    # Wikipedia tool
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=2)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
    
    # ArXiv tool for academic papers
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    
    # DuckDuckGo search tool
    search_tool = DuckDuckGoSearchRun()
    
    # Math calculation tool
    llm_math_chain = LLMMathChain.from_llm(llm=llm)
    calculator = Tool(
        func=llm_math_chain.run,
        name="Calculator",
        description="A tool for performing mathematical calculations. Input only mathematical expressions."
    )
    
    # Reasoning tool for detailed explanations
    prompt = """
    You're an expert mathematics teacher. Solve the following problem step by step:
    
    {question}
    
    First, identify what information is given and what is being asked.
    Then, lay out a clear strategy for solving the problem.
    Show your work carefully, with each step clearly labeled.
    Provide a final answer with appropriate units if applicable.
    
    Your solution:
    """
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template=prompt
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    reasoning_tool = Tool(
        name="MathReasoning",
        func=chain.run,
        description="A tool for solving math problems step-by-step with detailed explanations."
    )
    
    return [wiki_tool, arxiv_tool, search_tool, calculator, reasoning_tool]

# Set up the agent
def initialize_math_agent(tools):
    """Initialize the math problem solving agent"""
    system_message = SystemMessage(content="""
    You are an expert mathematics tutor and problem solver. Your goal is to:
    1. Solve mathematical problems accurately
    2. Provide clear, step-by-step explanations
    3. Use the appropriate tools for calculations and information gathering
    4. Organize your answers in a structured format with headings
    5. Include formulas and equations where relevant
    
    For math problems, always show your work and explain your thinking.
    For information queries, cite your sources where appropriate.
    """)
    
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=show_reasoning,
        handle_parsing_errors=True,
        system_message=system_message,
        early_stopping_method="generate"
    )

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm your Math Problem Solver! Ask me any math question or problem, and I'll solve it step-by-step."}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Example problems for the user to try
example_problems = [
    "Solve for x: 2x + 5 = 15",
    "Find the area of a circle with radius 4 cm",
    "If a train travels at 60 mph and takes 3 hours to reach its destination, how far did it travel?",
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack contains 25 berries. How many total pieces of fruit do I have at the end?"
]

# User input area with tabs for different input methods
tab1, tab2 = st.tabs(["Enter Your Question", "Try Examples"])

with tab1:
    question = st.text_area(
        "Type your math problem or question:",
        placeholder="Enter your math problem here...",
        height=100
    )
    solve_button = st.button("Solve Problem", type="primary")

with tab2:
    st.write("Select an example problem:")
    for i, example in enumerate(example_problems):
        if st.button(f"Example {i+1}", key=f"example_{i}"):
            question = example
            solve_button = True
            st.session_state.messages.append({"role": "user", "content": question})
            st.experimental_rerun()

# Process the request when button is clicked
if solve_button and question:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    
    # Initialize tools and agent
    tools = setup_tools()
    assistant_agent = initialize_math_agent(tools)
    
    # Generate response with callback handler for streaming
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=show_reasoning)
        try:
            response = assistant_agent.run(question, callbacks=[st_cb])
            st.write(response)
            
            # Add assistant response to chat
            st.session_state.messages.append({'role': 'assistant', "content": response})
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({'role': 'assistant', "content": error_msg})

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app uses Groq's AI models to solve math problems "
    "and provide step-by-step explanations. It can handle a wide range "
    "of mathematical topics from basic arithmetic to advanced calculus."
)
st.sidebar.markdown("### Feedback")
st.sidebar.markdown("[Report an issue](https://github.com/yourusername/math-problem-solver/issues)")