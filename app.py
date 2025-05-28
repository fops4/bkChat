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
CORS(app)  # Active CORS pour toutes les routes

load_dotenv()

# Vérifie la clé d'API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("La variable d'environnement GROQ_API_KEY n'est pas définie.")

client = Groq(api_key=GROQ_API_KEY)

# Charger et indexer les pages du site Matrix Télécoms
def load_vectorstore():
    urls = [
        "https://matrixtelecoms.com/",
        "https://matrixtelecoms.com/services/",
        "https://matrixtelecoms.com/contact/",
    ]
    loader = WebBaseLoader(urls, header_template={"User-Agent": os.getenv("USER_AGENT", "MatrixTelecomsBot/1.0 (+http://matrixtelecoms.com)")})
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs_split, embedding=embeddings)
    return vectorstore

vectorstore = load_vectorstore()

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_query = data.get("query", "")
        if not user_query:
            return jsonify({"error": "Aucune question fournie."}), 400

        # Recherche les documents similaires
        retrieved_docs = vectorstore.similarity_search(user_query, k=3)
        docs_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Construction du prompt
        final_prompt = f"""
Tu es un assistant virtuel pour Matrix Télécoms, une entreprise de télécommunications B2B au Cameroun. Ton rôle est d'aider les visiteurs à découvrir nos services (fibre optique, téléphonie IP, VPN, vidéosurveillance) et répondre à leurs questions. Utilise un ton professionnel et direct, sans signes de politesse comme 'Bonjour', 'Merci', ou 'N'hésitez pas à'. Fournis des réponses précises basées sur les informations du site www.matrixtelecoms.com.
Pour les questions hors de ton champ (ex. : détails techniques complexes), redirige vers le support : +237 242 139 545 ou info@matrixtelecoms.com.
Propose des liens vers les pages du site (ex. : Services, Contact) pour encourager l'exploration.


Matrix Télécoms S.A. est une entreprise camerounaise de télécommunications fondée en 1997 et membre du groupe ICCNET. Elle se positionne comme un acteur majeur dans le domaine des réseaux et télécoms au Cameroun, offrant une gamme étendue de services innovants et performants adaptés aux besoins des entreprises et institutions.
GRIN | Wissen finden & publizieren

Présentation Générale
Depuis sa création, Matrix Télécoms s'est engagée à fournir des solutions de connectivité de pointe, visant à améliorer la productivité des entreprises camerounaises. Avec plus de 25 ans d'expérience, elle dispose d'une infrastructure robuste et d'une expertise technique reconnue, lui permettant de garantir une qualité de service optimale et une satisfaction client élevée.

Historique
1995 : Création de l'International Computer Center (ICC), spécialisé dans le développement de logiciels.
1996 : Transformation en ICCNET S.A., devenant le premier opérateur privé de télécommunications au Cameroun.
1997 : Naissance de Matrix Télécoms S.A., issue de la fusion d'ICCNET avec Douala One.Com et CREOLINK Communications, consolidant sa position sur le marché.

Services et Solutions
Matrix Télécoms propose une large gamme de services adaptés aux besoins variés de ses clients :

Connectivité Internet : Fourniture d'accès Internet haut débit sécurisé, avec des solutions telles que le FTTH (Fiber To The Home) et le WiMax.
GRIN | Wissen finden & publizieren

Téléphonie sur IP (VoIP) : Solutions de téléphonie IP pour une communication efficace et économique.

Réseaux Privés Virtuels (VPN) : Mise en place de VPN pour sécuriser les échanges de données entre sites distants.

MPLS (Multi-Protocol Label Switching) : Services MPLS de niveau 2 et 3 pour une gestion optimisée du trafic réseau.

Vidéoconférence et Vidéosurveillance : Solutions de communication visuelle et de sécurité pour les entreprises.

IPLC (International Private Leased Circuit) : Circuits privés loués à l'international pour une connectivité fiable.


SD-WAN : Architecture WAN virtuelle permettant de centraliser et gérer les liaisons Internet multi-opérateurs.

Services de Sécurité Gérés (MSSP) : Protection contre les cyberattaques de 5e génération et optimisation de la bande passante.
matrixtelecoms.com

Vision et Engagement
Matrix Télécoms aspire à devenir le fournisseur par excellence de services de télécommunications innovants au Cameroun et à l'international. L'entreprise met un point d'honneur à offrir des solutions personnalisées, une assistance client 24h/24 et 7j/7, ainsi qu'une infrastructure réseau de pointe pour répondre aux exigences croissantes du marché.


Informations Complémentaires
Siège social : Ngousso, Yaoundé, Cameroun
Adresse : Immeuble Matrix Télécoms, Rue des Pêcheurs, Ngousso, Yaoundé, Cameroun
B.P. : 4124 Yaoundé
Téléphone : (+237) 242 232 201 / 691 837 897
Email : info@matrixtelecoms.com
Site web : https://matrixtelecoms.com

Effectif : Entre 100 et 250 employés
Langues parlées : Français et Anglais
Agrément : N°03775/ART/DG/DT/SDNA/SAG


Pour plus d'informations ou pour obtenir un devis personnalisé, vous pouvez visiter leur site officiel ou les contacter directement via les coordonnées fournies ci-dessus.

### Informations issues du site :
{docs_context}

### Question du visiteur :
{user_query}

Réponds en français de manière claire, concise, et professionnelle.
"""

        # Appel à Groq
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # par sécurité
    app.run(host="0.0.0.0", port=port)
