from transformers import pipeline

def answer_question(context, question):
    question_answerer = pipeline("question-answering", model="deepset/roberta-base-squad2")
    answer = question_answerer(question=question, context=context)
    return answer

if __name__ == "__main__":
    context = "Exemplo de contexto educacional para responder perguntas."
    question = "O que é o Hugging Face?"
    print(answer_question(context, question))
