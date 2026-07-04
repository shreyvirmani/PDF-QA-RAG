from src.llm import get_llm

llm = get_llm()

response = llm.invoke("Explain Machine Learning in one paragraph.")

print(response.content)