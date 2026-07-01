import os

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from memory import Memory
from utils import format_convo

reflection_template = ChatPromptTemplate(["""
    You are analyzing interactions between the user and the AI assistant to create memories that will help guide future interactions. 
    Your task is to extract key elements that would be most helpful when encountering similar requests in the future.

Review the conversation and create a memory reflection following these rules:

1. For any field where you don't have enough information or the field isn't relevant, use "N/A"
2. Be extremely concise - each string should be one clear, actionable sentence
3. Focus only on information that would be useful for handling similar future conversations
4. Context_tags should be specific enough to match similar situations but general enough to be reusable
5. Never use unquoted values like N/A

Output valid JSON in exactly this format:
{{
    "context_tags": string,                //1 keyword that would help identify similar future conversations. Use field-specific terms like "fitness_plan", "chemistry_exam_prep", "crud_app_project", "emotional_stress", "current_struggles"
    "conversation_summary": string, // One sentence describing what the conversation accomplished
    "what_worked": string,         // Most effective approach or strategy used in this conversation
    "what_to_avoid": string        // Most important pitfall or ineffective approach to avoid
}}

Examples:
- Good context_tags: "marathon_sub3_plan", "chemistry_midterm_review", "yearly_goal"
- Bad context_tags: "self_improvement", "academic_overview", "future_expectations"

- Good conversation_summary: "Created a training plan for the user to increase running pace for upcoming marathon"
- Bad conversation_summary: "Discussed a machine learning paper"

- Good what_worked: "Using analogies from matrix multiplication to explain attention score calculations"
- Bad what_worked: "Explained the technical concepts well"

- Good what_to_avoid: "Diving into mathematical formulas before establishing user's familiarity with linear algebra fundamentals"
- Bad what_to_avoid: "Used complicated language"

Additional examples for different research scenarios:

Context tags examples:
- "experimental_design", "control_groups", "methodology_critique"
- "statistical_significance", "p_value_interpretation", "sample_size"
- "research_limitations", "future_work", "methodology_gaps"

Conversation summary examples:
- "Clarified why the paper's cross-validation approach was more robust than traditional hold-out methods"
- "Helped identify potential confounding variables in the study's experimental design"

What worked examples:
- "Breaking down complex statistical concepts using visual analogies and real-world examples"
- "Connecting the paper's methodology to similar approaches in related seminal papers"

What to avoid examples:
- "Assuming familiarity with domain-specific jargon without first checking understanding"
- "Over-focusing on mathematical proofs when the user needed intuitive understanding"

Do not include any text outside the JSON object in your response.

Here is the prior conversation:

{conversation}
"""
                                         ])


def add_episodic_mem(memobj: Memory, conver):
    conversation = format_convo(conver)

    reflect_model = OllamaLLM(model="llama3.2", base_url=os.getenv("LOCAL_ENDPOINT"),temperature=0.5)
    reflect = reflection_template | reflect_model | JsonOutputParser()

    reflection = reflect.invoke(conversation)

    memobj.add_mem("episodic_mem",conversation,reflection)

def episodic_recall(memobj: Memory, query: str):
    return memobj.query_mem("episodic_mem",query)

def episodic_sys_prompt(memobj: Memory, query , convo):
    mem = episodic_recall(memobj, query)
    curr_convo = format_convo(convo[-4:])

    ww = mem['metadatas'][0][0]['what_worked']
    wta = mem['metadatas'][0][0]["what_to_avoid"]
    summary = mem['metadatas'][0][0]['conversation_summary']
    memobj.update_ww(ww.split('. '))
    memobj.update_wta(wta.split('. '))

    prev_convo = format_convo(convo[-4:-1]) if len(convo) >= 4 else format_convo(convo[:-1])
    epi_prompt = f"""
    You are a personal assistance, specializing in answering questions, 
    keep the answers short but precise,
    you can get better overtime by asking questions on topics you don't know, an evolving AI assistant,
    if you don't know the answer to the user's question, 
    clarify your incompetence and ask the user to give you more information so in the future if the user ask again you can answer it 
    You recall similar conversations with the user, here are the details:
    
    Past relevant memory:
    Summary: {summary}
    Previous conversations: {prev_convo}
    What has worked well: {memobj.what_worked}
    What to avoid: {memobj.what_to_avoid}
    
    Current conversation: {curr_convo}
    Use these memories as context for your response to the user
    """
    return SystemMessage(content=epi_prompt)