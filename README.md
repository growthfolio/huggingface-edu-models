
# Modelos Educacionais com Hugging Face

Este projeto utiliza modelos do Hugging Face para finalidades educacionais, oferecendo soluções 
automatizadas para sumarização de textos, perguntas e respostas, e geração de questões.

## Funcionalidades

- **Sumarização de Textos**: Gera resumos de textos longos para facilitar a compreensão de conteúdos extensos.
- **Perguntas e Respostas**: Responde a perguntas baseadas em um contexto, útil para sistemas de aprendizado interativo.
- **Geração de Perguntas**: Cria perguntas automaticamente a partir de textos, auxiliando no desenvolvimento de conteúdo educacional.

## Como Usar

1. Clone o repositório:
   ```bash
   git clone https://github.com/felipemacedo1/huggingface-edu-models.git
   ```

2. Instale as dependências necessárias:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute os scripts localizados em `src/` para utilizar cada funcionalidade específica:
   ```bash
   python src/summarizer.py          # Sumarização de textos
   python src/question_answering.py   # Perguntas e Respostas
   python src/question_generator.py   # Geração de Perguntas
   ```

## Requisitos

- **Python 3.7+**
- **Bibliotecas**: `transformers`, `torch`

---

Para mais informações sobre os modelos utilizados, visite [Hugging Face](https://huggingface.co/models).