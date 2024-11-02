from transformers import pipeline

def generate_questions(text):
    question_generator = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")
    questions = question_generator(text)
    return questions

if __name__ == "__main__":
    text = "Machine learning é uma área da inteligência artificial."
    print(generate_questions(text))
