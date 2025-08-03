# 🤖 Hugging Face Educational Models - IA para Educação

## 🎯 Objetivo de Aprendizado
Projeto desenvolvido para estudar **Machine Learning** e **NLP (Natural Language Processing)** com **Hugging Face Transformers**, implementando modelos de IA para sumarização de textos, Q&A e geração de perguntas educacionais.

## 🛠️ Tecnologias Utilizadas
- **Framework:** Hugging Face Transformers
- **Linguagem:** Python 3.7+
- **ML/DL:** PyTorch, TensorFlow
- **NLP:** BERT, T5, GPT models
- **Modelos:** Pre-trained transformers
- **Conceitos estudados:**
  - Natural Language Processing
  - Transfer Learning
  - Transformer architecture
  - Text summarization
  - Question answering
  - Text generation

## 🚀 Demonstração
```python
# Sumarização de textos
from transformers import pipeline

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Question Answering
def answer_question(context, question):
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Geração de perguntas
def generate_questions(text):
    question_generator = pipeline("text2text-generation", 
                                 model="mrm8488/t5-base-finetuned-question-generation-ap")
    questions = question_generator(text)
    return questions
```

## 💡 Principais Aprendizados

### 🧠 Natural Language Processing
- **Tokenização:** Processamento de texto para modelos
- **Embeddings:** Representação vetorial de palavras
- **Attention Mechanism:** Como modelos focam em partes relevantes
- **Transfer Learning:** Uso de modelos pré-treinados

### 🤖 Modelos Transformer
- **BERT:** Bidirectional Encoder Representations
- **T5:** Text-to-Text Transfer Transformer
- **BART:** Denoising Autoencoder
- **DistilBERT:** Versão otimizada do BERT

### 📚 Aplicações Educacionais
- **Sumarização:** Resumos automáticos de conteúdo
- **Q&A Systems:** Sistemas de perguntas e respostas
- **Question Generation:** Criação automática de questões
- **Content Analysis:** Análise de textos educacionais

## 🧠 Conceitos Técnicos Estudados

### 1. **Text Summarization**
```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.summarizer = pipeline("summarization", 
                                  model=self.model, 
                                  tokenizer=self.tokenizer)
    
    def summarize(self, text, max_length=130, min_length=30):
        # Verificar tamanho do texto
        if len(text.split()) < 50:
            return "Texto muito curto para sumarização"
        
        summary = self.summarizer(text, 
                                max_length=max_length, 
                                min_length=min_length, 
                                do_sample=False)
        return summary[0]['summary_text']
```

### 2. **Question Answering System**
```python
class QuestionAnsweringSystem:
    def __init__(self, model_name="distilbert-base-cased-distilled-squad"):
        self.qa_pipeline = pipeline("question-answering", model=model_name)
    
    def answer(self, context, question):
        result = self.qa_pipeline(question=question, context=context)
        
        return {
            'answer': result['answer'],
            'confidence': result['score'],
            'start': result['start'],
            'end': result['end']
        }
    
    def batch_answer(self, context, questions):
        answers = []
        for question in questions:
            answer = self.answer(context, question)
            answers.append({
                'question': question,
                'answer': answer['answer'],
                'confidence': answer['confidence']
            })
        return answers
```

### 3. **Question Generator**
```python
class QuestionGenerator:
    def __init__(self, model_name="mrm8488/t5-base-finetuned-question-generation-ap"):
        self.generator = pipeline("text2text-generation", model=model_name)
    
    def generate_questions(self, text, num_questions=5):
        # Preparar texto para o modelo T5
        input_text = f"generate questions: {text}"
        
        questions = self.generator(
            input_text,
            max_length=64,
            num_return_sequences=num_questions,
            temperature=0.7,
            do_sample=True
        )
        
        return [q['generated_text'] for q in questions]
    
    def generate_with_answers(self, text):
        questions = self.generate_questions(text)
        qa_system = QuestionAnsweringSystem()
        
        qa_pairs = []
        for question in questions:
            answer = qa_system.answer(text, question)
            qa_pairs.append({
                'question': question,
                'answer': answer['answer'],
                'confidence': answer['confidence']
            })
        
        return qa_pairs
```

## 📁 Estrutura do Projeto
```
huggingface-edu-models/
├── src/
│   ├── summarizer.py           # Sumarização de textos
│   ├── question_answering.py   # Sistema de Q&A
│   ├── question_generator.py   # Geração de perguntas
│   ├── models/                 # Classes dos modelos
│   └── utils/                  # Utilitários
├── examples/                   # Exemplos de uso
├── data/                       # Dados de exemplo
├── requirements.txt            # Dependências
└── notebooks/                  # Jupyter notebooks
```

## 🔧 Como Executar

### Instalação
```bash
# Clone o repositório
git clone <repo-url>
cd huggingface-edu-models

# Instale dependências
pip install -r requirements.txt

# Ou com conda
conda env create -f environment.yml
conda activate huggingface-edu
```

### Uso Básico
```python
# Sumarização
from src.summarizer import TextSummarizer

summarizer = TextSummarizer()
text = "Texto longo para ser sumarizado..."
summary = summarizer.summarize(text)
print(summary)

# Question Answering
from src.question_answering import QuestionAnsweringSystem

qa = QuestionAnsweringSystem()
context = "Contexto com informações..."
question = "Qual é a resposta?"
answer = qa.answer(context, question)
print(answer)

# Geração de Perguntas
from src.question_generator import QuestionGenerator

qg = QuestionGenerator()
questions = qg.generate_questions(text)
print(questions)
```

## 🎯 Aplicações Educacionais

### 1. **Sistema de Estudo Automatizado**
- Sumarização de livros didáticos
- Geração de questões para revisão
- Sistema de Q&A para dúvidas

### 2. **Criação de Conteúdo**
- Geração automática de exercícios
- Resumos de aulas e palestras
- Criação de flashcards

### 3. **Análise de Texto**
- Extração de conceitos principais
- Identificação de tópicos importantes
- Avaliação de compreensão

## 🚧 Desafios Enfrentados
1. **Model Selection:** Escolher modelos adequados para cada tarefa
2. **Performance:** Otimizar velocidade de inferência
3. **Memory Management:** Gerenciar uso de GPU/CPU
4. **Text Preprocessing:** Preparar textos para os modelos
5. **Quality Control:** Avaliar qualidade das saídas
6. **Portuguese Language:** Adaptar para textos em português

## 📚 Recursos Utilizados
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Papers With Code](https://paperswithcode.com/) - State-of-the-art models
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer paper
- [BERT Paper](https://arxiv.org/abs/1810.04805) - BERT architecture

## 📈 Próximos Passos
- [ ] Implementar fine-tuning para domínio específico
- [ ] Adicionar suporte para múltiplos idiomas
- [ ] Criar interface web com Streamlit
- [ ] Implementar avaliação automática de qualidade
- [ ] Adicionar modelos de classificação de texto
- [ ] Integrar com bases de conhecimento

## 🔗 Projetos Relacionados
- [Go PriceGuard API](../go-priceguard-api/) - Backend com IA features
- [React PriceGuard View](../react-priceguard-view/) - Frontend para IA
- [Java Generation Notes](../java-generation-notes/) - Base de estudos

---

**Desenvolvido por:** Felipe Macedo  
**Contato:** contato.dev.macedo@gmail.com  
**GitHub:** [FelipeMacedo](https://github.com/felipemacedo1)  
**LinkedIn:** [felipemacedo1](https://linkedin.com/in/felipemacedo1)

> 💡 **Reflexão:** Este projeto abriu minha visão para o potencial da IA na educação. Trabalhar com modelos Transformer e NLP consolidou conhecimentos fundamentais em Machine Learning e suas aplicações práticas.