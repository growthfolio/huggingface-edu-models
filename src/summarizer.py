from transformers import pipeline

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary

if __name__ == "__main__":
    text = "Texto de exemplo para sumarização..."
    print(summarize_text(text))
