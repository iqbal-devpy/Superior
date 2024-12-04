from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import logging
import spacy

# Configure logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Load SpaCy model
nlp = spacy.load("en_core_web_md")  # Use medium model for better similarity accuracy

# Function to load terms from a file
def load_medical_terms(file_path):
    with open(file_path, 'r') as file:
        # Read each line, strip whitespace, and create SpaCy Doc objects
        terms = [nlp(line.strip()) for line in file.readlines()]
    return terms

# Load medical terms from file
core_terms = load_medical_terms('medical_term.txt')
# AI Model Setup
template = '''
Answer the question below based on the specified topic.

Here is the conversation history:{context}

Topic: {topic}

Question: {question}

Answer:
'''
model = OllamaLLM(model='gemma:2b')
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Function to check if the input is a medical question
def is_medical_question(user_input, threshold=0.4):
    user_doc = nlp(user_input.lower())
    for term_doc in core_terms:
        if user_doc.similarity(term_doc) > threshold:
            return True
    return False

# Function to update the context with a maximum number of turns
def update_context(context, user_input, response, max_turns=5):
    context_lines = context.strip().split('\n')[-2 * max_turns:]
    return '\n'.join(context_lines + [f"User: {user_input}", f"AI: {response}"])

# Main conversation handler
def handle_topic_specific_convo():
    context = ''
    print("Welcome to your personal medical assistant! Type 'Exit' to quit.")
        
    # Define topic focus
    topic_focus = '''Assume the role of a knowledgeable medical assistant, but clarify you are not a licensed medical professional.
    Use clear, concise, and evidence-based language to ensure accuracy and user understanding.
    Provide information on symptoms, conditions, treatments, and preventive care but avoid diagnosing or prescribing.
    For emergencies or critical issues, recommend contacting emergency services immediately.
    Ask clarifying follow-up questions when necessary to gather context.
    Respect confidentiality and avoid collecting or storing personal identifiable information (PII).
    Be empathetic, supportive, and respectful in all interactions.
    Promote preventive health practices and general wellness advice.
    Acknowledge knowledge limits and recommend professional consultation when needed.
    Suggest mental health services or helplines for stress or emotional concerns.
    Be mindful of cultural and societal differences in healthcare practices.
    Include a disclaimer: "I am an AI assistant providing general medical information. Consult a licensed healthcare provider for serious concerns."'''

    while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("AI: Goodbye! Have a great day!")
                break

            # Initialize response to ensure it's always defined
            response = None

            if is_medical_question(user_input):
                # Try to get a response from the LLM for medical-related queries
                try:
                    result = chain.invoke({'context': context, 'topic': topic_focus, 'question': user_input})
                    response = result.strip()
                except Exception as e:
                    response = "I'm sorry, something went wrong. Could you rephrase your question?"
                    logging.error(f"Error in LLM response: {e}")
            else:
                response = "This doesn't seem like a medical-related question. Please ask something about health."

            # Append a disclaimer to the response
            disclaimer = "\n\nDisclaimer: I am an AI assistant providing general medical information. Consult a licensed healthcare provider for serious concerns."
            response += disclaimer

            # Print the response and update the context
            print("AI:", response)
            context = update_context(context, user_input, response)

# Run the chatbot
if __name__ == "__main__":
    handle_topic_specific_convo()

# Ask for feedback at the end of the session
feedback = input("How helpful was this session? (Excellent/Good/Fair/Poor): ")
logging.info(f"User feedback: {feedback}")
