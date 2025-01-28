# from fastapi import FastAPI, Body
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain.chains import LLMChain
# import openai
# import os



# app = FastAPI()

# @app.post("/evaluate_llm_code_langchain")
# async def evaluate_llm_code_langchain(llm_code: str = Body(...)):
#     system_template = "You are an AI compliance expert knowledgeable about the NIST AI RMF."
#     user_template = (
#         "The following is the code of a large language model (LLM). "
#         "Evaluate whether the code violates the NIST AI Risk Management Framework (AI RMF) in terms "
#         "of governance, risk mapping, measurement, and management principles. Highlight specific areas "
#         "of concern or potential violations, and suggest ways to mitigate these issues if found.\n\n"
#         "Code:\n{llm_code}"
#     )

#     system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
#     user_message_prompt = HumanMessagePromptTemplate.from_template(user_template)
#     chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_prompt])
    
#     llm = ChatOpenAI(model_name="gpt-4", temperature=0)
#     chain = LLMChain(llm=llm, prompt=chat_prompt)
    
#     response = chain.run(llm_code=llm_code)
#     return {"evaluation": response}


# import os
# import requests

# # Set your API key here, or pull from an environment variable

# OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# def evaluate_code(llmcode):

#     prompt_header = (
#         "The following is the code of a large language model (LLM). "
#         "Evaluate whether the code violates the NIST AI Risk Management Framework (AI RMF) in terms "
#         "of governance, risk mapping, measurement, and management principles. Highlight specific areas "
#         "of concern or potential violations, and suggest ways to mitigate these issues if found.\n\n"
#         "Code:\n{llmcode}"
#     )


#     # Combine into the final prompt
#     prompt = prompt_header

#     # Create the request body for the ChatCompletion API
#     request_body = {
#         "model": "gpt-4o",  # or "gpt-4" if desired (and you have access)
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     }

#     try:
#         # Make the POST request
#         response = requests.post(
#             OPENAI_API_URL,
#             headers={
#                 "Content-Type": "application/json",
#                 "Authorization": f"Bearer {OPENAI_API_KEY}"
#             },
#             json=request_body
#         )
        
#         # Raise an error for non-200 responses
#         response.raise_for_status()

#         data = response.json()
#         # Extract the summary from the response
#         summary = data["choices"][0]["message"]["content"].strip()
#         return summary

#     except Exception as e:
#         print("Error fetching summary from OpenAI:", e)
#         return None


# # Example usage:
# if __name__ == "__main__":
#     test_code = """import numpy as np

# def detect_anomalies(data):
#     threshold = 3.0  # anything beyond 3 standard deviations is flagged
#     anomalies = []

#     mean_val = np.mean(data)
#     std_dev = np.std(data)

#     for i, value in enumerate(data):
#         # If the data point is too far from the mean, flag it
#         if abs(value - mean_val) > threshold * std_dev:
#             anomalies.append((i, value))
    
#     return anomalies
# """
        

#     summary_result = evaluate_code(test_code)
#     print("Summary:\n", summary_result)

from openai import OpenAI
client = OpenAI(api_key = "")
llmcode = """import numpy as np

def detect_anomalies(data):
    threshold = 3.0  # anything beyond 3 standard deviations is flagged
    anomalies = []

    mean_val = np.mean(data)
    std_dev = np.std(data)

    for i, value in enumerate(data):
        # If the data point is too far from the mean, flag it
        if abs(value - mean_val) > threshold * std_dev:
            anomalies.append((i, value))
    
    return anomalies
"""
response = client.chat.completions.create(
    model = "gpt-4o",
    messages = [
        {
            "role": "system",
            "content": "You are a person that decides if a tool Adheres to the NIST AI RMF guidelines"
        },
        {
            "role": "user",
            "content": f"The following is the code of a large language model (LLM). Evaluate whether the code violates the NIST AI Risk Management Framework (AI RMF) in terms of governance, risk mapping, measurement, and management principles. Highlight specific areas of concern or potential violations, and suggest ways to mitigate these issues if found.\n\n Code:\n{llmcode} Evaluate from a scale of 1-5 give your chain of thought and then the number at the end"
        }
    ]
)

response.choices[0].message.content