import requests
import zipfile
from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st


def load_file(url: str):
    """ Функция загрузки документа по url как текст."""
    try:
        response = requests.get(url)
        # Проверка ответа и если была ошибка - формирование исключения.
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(e)


def load_bd_vect(url: str):
    """ Функция загружает векторную Базу знаний."""
    name_bd = 'federallab_bd_index.zip'
    response = requests.get(url)
    # Проверка ответа и если была ошибка - формирование исключения.
    response.raise_for_status()
    # Сохранение архива.
    with open(name_bd, 'wb') as file:
        file.write(response.content)
    # Разархивирование Базы знаний.
    with zipfile.ZipFile(name_bd, 'r') as zip:
        zip.extractall()
    # Загрузка векторной Базы знаний.
    federallab_bd = FAISS.load_local(
        f'federallab_bd_index', OpenAIEmbeddings())
    return federallab_bd


def load_bd_question(url: str):
    """ Функция загружает векторную Базу знаний."""
    name_bd = 'federallab_bd_question.zip'
    response = requests.get(url)
    # Проверка ответа и если была ошибка - формирование исключения.
    response.raise_for_status()
    # Сохранение архива.
    with open(name_bd, 'wb') as file:
        file.write(response.content)
    # Разархивирование Базы знаний.
    with zipfile.ZipFile(name_bd, 'r') as zip:
        zip.extractall()
    # Загрузка векторной Базы знаний.
    federallab_bd = FAISS.load_local(
        f'federallab_bd_question', OpenAIEmbeddings())
    return federallab_bd


def query_refiner(client, model, conversation, query):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nAnswer in Russian.\n\nRefined Query:"},
            {"role": "user",   "content": f"Query: {query}"}
        ],
        # prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    return response.choices[0].message.content


def conversation_string(messages, dialog_depth):
    conversation_string = ''
    k = 0 if len(messages) <= dialog_depth * \
        2 else -1*dialog_depth*2
    for message in messages[k:]:
        conversation_string += f"{message['role']}: {message['content']}\n"
    return conversation_string


def bd_retrever(bd, query):
    docs = bd.similarity_search_with_score(query, k=3)
    message_content = ''
    for i, doc in enumerate(docs):
        message_content = message_content + \
            f'Отрывок документа №{i+1}:{doc[0].page_content}\n'
    return message_content


def find_query(bd, query):
    doc = bd.similarity_search_with_score(query, k=1)
    if doc[0][1] < 0.08:
        return True, doc[0][0].metadata['answer_gpt']
    else:
        return False, ''


def stick_it_good():
    # make header sticky.
    st.markdown(
        """
            <div class='fixed-header'/>
            <style>
                div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;                    
                    background-color: white;
                    z-index: 999;
                }
                .fixed-header {
                    border-bottom: 1px solid black;
                }
            </style>
        """,
        unsafe_allow_html=True
    )


def move_focus():
    # inspect the html to determine which control to specify to receive focus (e.g. text or textarea).
    st.components.v1.html(
        f"""
            <script>
                var textarea = window.parent.document.querySelectorAll("textarea[type=textarea]");
                for (var i = 0; i < textarea.length; ++i) {{
                    textarea[i].focus();
                }}
            </script>
        """,
    )
