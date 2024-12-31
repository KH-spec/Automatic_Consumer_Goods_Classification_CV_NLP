# 🛍️ Automatic Consumer Goods Classification Project
*(French version below)*

This project aims to automate product categorization on an e-commerce platform using both product images and descriptions. The analysis was conducted within a Jupyter Notebook environment.

🌟 **Project Goal:** Build a system to automatically classify consumer goods into predefined categories.

**Methodology:**

🔬 **Data:**  A dataset containing product images, descriptions, and category labels was used.

🗣️ **NLP (Text Analysis):**
   * 🧹 Preprocessing: Tokenization, stop word removal, lemmatization.
   * 🤖 Models: Bag-of-Words (CountVectorizer, TF-IDF), Latent Dirichlet Allocation (LDA), Word Embeddings (Word2Vec, BERT, Universal Sentence Encoder).
   * 📊 Dimensionality Reduction: t-SNE, Principal Component Analysis (PCA).
   * 🧮 Clustering: K-Means.
   * 🛠️ **Technologies:** Python (3.8.8), Pandas (1.2.4), Seaborn (0.11.1), Matplotlib (3.3.4), scikit-learn, NLTK (likely, for NLP tasks), potentially others.

📸 **Image Processing:**
   * 🧹 Preprocessing: Grayscale conversion, Gaussian Blur, contrast adjustment, histogram equalization, resizing.
   * 🔎 Feature Extraction: Scale-Invariant Feature Transform (SIFT), Oriented FAST and Rotated BRIEF (ORB).
   * 📊 Dimensionality Reduction: Principal Component Analysis (PCA).
   * 🤖 Models: Convolutional Neural Network (CNN) - supervised and unsupervised learning, Transfer Learning.
   * 🧮 Clustering: K-Means.
   * 🛠️ **Technologies:** OpenCV, TensorFlow/Keras (likely), potentially others.


🤝 **Combined Approach:** Data from NLP and image processing were combined and clustered.

🎯 **Results:**  The project demonstrated feasibility but showed room for improvement in pre-processing and model selection. Accuracy was limited.

📈 **Areas for Improvement:**
   * ⬆️ Increase dataset size.
   * ⬆️ Improve data quality (labeling and image clarity).
   * ⚙️ Refine pre-processing techniques.
   * 🤖 Explore more advanced models.

💻 **Development Environment:** Jupyter Notebook

---

# 🛍️ Projet de Classification Automatique des Produits de Consommation

Ce projet vise à automatiser la catégorisation des produits sur une plateforme de commerce en ligne en utilisant à la fois des images de produits et des descriptions. L'analyse a été réalisée dans un environnement Jupyter Notebook.

🌟 **Objectif du projet :** Construire un système pour classer automatiquement les biens de consommation dans des catégories prédéfinies.

**Méthodologie :**

🔬 **Données :** Un jeu de données contenant des images de produits, des descriptions et des étiquettes de catégories a été utilisé.

🗣️ **NLP (Analyse de texte) :**
   * 🧹 Prétraitement : Tokenisation, suppression des mots vides, lemmatisation.
   * 🤖 Modèles : Bag-of-Words (CountVectorizer, TF-IDF), Latent Dirichlet Allocation (LDA), Embeddings de mots (Word2Vec, BERT, Universal Sentence Encoder).
   * 📊 Réduction de la dimensionnalité : t-SNE, Analyse en Composantes Principales (PCA).
   * 🧮 Clustering : K-Means.
   * 🛠️ **Technologies :** Python (3.8.8), Pandas (1.2.4), Seaborn (0.11.1), Matplotlib (3.3.4), scikit-learn, NLTK (probablement pour les tâches NLP), potentiellement d'autres.

📸 **Traitement d'image :**
   * 🧹 Prétraitement : Conversion en niveaux de gris, flou gaussien, ajustement du contraste, égalisation d'histogramme, redimensionnement.
   * 🔎 Extraction de caractéristiques : Scale-Invariant Feature Transform (SIFT), Oriented FAST and Rotated BRIEF (ORB).
   * 📊 Réduction de la dimensionnalité : Analyse en Composantes Principales (PCA).
   * 🤖 Modèles : Réseau de Neurones Convolutifs (CNN) - apprentissage supervisé et non supervisé, apprentissage par transfert.
   * 🧮 Clustering : K-Means.
   * 🛠️ **Technologies :** OpenCV, TensorFlow/Keras (probablement), potentiellement d'autres.

🤝 **Approche combinée :** Les données provenant de NLP et du traitement d'image ont été combinées et regroupées.

🎯 **Résultats :** Le projet a démontré la faisabilité mais a montré qu'il restait de la place pour l'amélioration du prétraitement et de la sélection des modèles. La précision était limitée.

📈 **Axes d'amélioration :**
   * ⬆️ Augmenter la taille du jeu de données.
   * ⬆️ Améliorer la qualité des données (étiquetage et clarté des images).
   * ⚙️ Affiner les techniques de prétraitement.
   * 🤖 Explorer des modèles plus avancés.

💻 **Environnement de développement :** Jupyter Notebook
