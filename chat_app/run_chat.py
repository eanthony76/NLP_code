import requests

API_ENDPOINT = "http://127.0.0.1:5000/ask"

def ask_question(question):
    response = requests.post(API_ENDPOINT, json={"question": question})
    if response.status_code == 200:
        return response.json().get("answer")
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

if __name__ == "__main__":
    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        answer = ask_question(question)
        if answer:
            print(f"Answer: {answer}")
