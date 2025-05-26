from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("La variable d'environnement GROQ_API_KEY n'est pas définie.")

client = Groq(api_key=GROQ_API_KEY)

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Vérifier si l'index FAISS existe
    if os.path.exists("index.faiss") and os.path.exists("index.pkl"):
        return FAISS.load_local("index", embeddings=embeddings, allow_dangerous_deserialization=True)
    else:
        # Construire l'index si nécessaire (pour tests locaux)
        urls = [
            "https://matrixtelecoms.com/",
            "https://matrixtelecoms.com/services/",
            "https://matrixtelecoms.com/contact/",
        ]
        loader = WebBaseLoader(urls, header_template={"User-Agent": os.getenv("USER_AGENT", "MatrixTelecomsBot/1.0 (+http://matrixtelecoms.com)")})
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)  # Réduire la taille des chunks
        docs_split = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(docs_split, embedding=embeddings)
        vectorstore.save_local("index")
        return vectorstore

vectorstore = load_vectorstore()

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_query = data.get("query", "")
        if not user_query:
            return jsonify({"error": "Aucune question fournie."}), 400

        retrieved_docs = vectorstore.similarity_search(user_query, k=3)
        docs_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        final_prompt = f"""
        Tu es un assistant virtuel pour Matrix Télécoms, une entreprise de télécommunications B2B au Cameroun. Ton rôle est d'aider les visiteurs à découvrir nos services (fibre optique, téléphonie IP, VPN, vidéosurveillance) et répondre à leurs questions. Utilise un ton professionnel et direct, sans signes de politesse comme 'Bonjour', 'Merci', ou 'N'hésitez pas à'. Fournis des réponses précises basées sur les informations du site www.matrixtelecoms.com.
        Pour les questions hors de ton champ (ex. : détails techniques complexes), redirige vers le support : +237 242 139 545 ou info@matrixtelecoms.com.
        Propose des liens vers les pages du site (ex. : Services, Contact) pour encourager l'exploration.

        ### Informations issues du site :
        {docs_context}

        ### Question du visiteur :
        {user_query}

        Réponds en français de manière claire, concise, et professionnelle.
        """

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Tu es un assistant client intelligent pour Matrix Télécoms."},
                {"role": "user", "content": final_prompt}
            ]
        )

        answer = response.choices[0].message.content
        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"error": f"Erreur serveur : {str(e)}"}), 500