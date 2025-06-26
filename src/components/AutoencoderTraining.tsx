import React, { useState, useEffect } from 'react';
import { Brain, Play, Save, Upload, Info, Search, RefreshCw, CheckCircle, AlertCircle, Code, X, Shuffle, Database } from 'lucide-react';
import ConfigManager from '../config/AppConfig';

interface AutoencoderConfig {
  input_dim: number;
  encoding_dim: number;
  hidden_layers: number[];
  activation: string;
  learning_rate: number;
  epochs: number;
  batch_size: number;
}

interface ModelInfo {
  is_trained: boolean;
  tensorflow_available: boolean;
  architecture: string;
  config: AutoencoderConfig;
  corpus_size: number;
  vocabulary_size?: number;
  input_dim?: number;
  total_params?: number;
}

interface ReconstructionResult {
  original_text: string;
  encoded_shape: number[];
  reconstruction_error: number;
  similarity: number;
  compression_ratio: number;
  top_original_terms: [string, number][];
  top_reconstructed_terms: [string, number][];
  encoded_representation: number[];
}

const AutoencoderTraining: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('training');
  const [trainingTexts, setTrainingTexts] = useState<string>('');
  const [testText, setTestText] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [isTraining, setIsTraining] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [trainingResult, setTrainingResult] = useState<any>(null);
  const [reconstructionResult, setReconstructionResult] = useState<ReconstructionResult | null>(null);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [encodedText, setEncodedText] = useState<number[] | null>(null);
  const [message, setMessage] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [showCodePopup, setShowCodePopup] = useState(false);
  const [selectedCodeStep, setSelectedCodeStep] = useState<string>('');
  
  // États pour le clustering
  const [clusteringResults, setClusteringResults] = useState<any>(null);
  const [nClusters, setNClusters] = useState<number>(4);
  const [isExtracting, setIsExtracting] = useState(false);
  const [isClustering, setIsClustering] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationResults, setOptimizationResults] = useState<any>(null);
  
  // États pour les nouvelles fonctionnalités avancées
  const [evaluationResults, setEvaluationResults] = useState<any>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [useOptimizedTraining, setUseOptimizedTraining] = useState(true);
  const [advancedResults, setAdvancedResults] = useState<any>(null);
  const [isAdvancedAnalysis, setIsAdvancedAnalysis] = useState(false);
  const [modelToSave, setModelToSave] = useState<string>('');
  const [modelToLoad, setModelToLoad] = useState<string>('');
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  
  // États pour l'entraînement régularisé (techniques avancées)
  const [isTrainingRegularized, setIsTrainingRegularized] = useState(false);
  const [regularizedResults, setRegularizedResults] = useState<any>(null);
  
  // États pour la configuration du dataset
  const [datasetSize, setDatasetSize] = useState<number>(5000);
  const [maxDatasetSize, setMaxDatasetSize] = useState<number>(3600000); // 3.6M échantillons disponibles
  const [useRandomSample, setUseRandomSample] = useState<boolean>(true);
  const [useAmazonDataset, setUseAmazonDataset] = useState<boolean>(true);

  const autoencoderDefaults = ConfigManager.getModelDefaults('autoencoder');
  const [config, setConfig] = useState<AutoencoderConfig>({
    input_dim: autoencoderDefaults.defaultInputDim,
    encoding_dim: autoencoderDefaults.defaultEncodingDim,
    hidden_layers: autoencoderDefaults.hiddenLayers,
    activation: 'relu',
    learning_rate: autoencoderDefaults.defaultLearningRate,
    epochs: autoencoderDefaults.defaultEpochs,
    batch_size: autoencoderDefaults.defaultBatchSize
  });

  const API_BASE = ConfigManager.getApiUrl('autoencoder');

  // Codes source pour chaque étape
  const getStepCode = (step: string) => {
    const codes: { [key: string]: string } = {
      'tfidf': `# Étape 2: Vectorisation TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import re

class TFIDFProcessor:
    def __init__(self, max_features=1000):
        """
        Processeur TF-IDF pour l'autoencoder
        
        Args:
            max_features (int): Nombre maximum de features TF-IDF
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),      # Unigrammes + bigrammes
            min_df=2,                # Minimum 2 documents
            max_df=0.8,              # Maximum 80% des documents
            sublinear_tf=True        # Normalisation TF sublinéaire
        )
        self.scaler = StandardScaler()
        
    def preprocess_text(self, text):
        """Préprocessing d'un texte"""
        # Nettoyage basique
        text = re.sub(r'[^\\w\\s]', ' ', text.lower())
        text = re.sub(r'\\s+', ' ', text.strip())
        return text
        
    def fit_transform(self, texts):
        """
        Entraîne le TF-IDF et transforme les textes
        
        Args:
            texts (list): Liste de textes à vectoriser
            
        Returns:
            np.ndarray: Matrice TF-IDF normalisée
        """
        # Préprocessing des textes
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorisation TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        # Normalisation pour l'autoencoder
        tfidf_normalized = self.scaler.fit_transform(tfidf_matrix.toarray())
        
        print(f"✅ TF-IDF: {len(texts)} textes → {tfidf_matrix.shape}")
        print(f"📊 Vocabulaire: {len(self.tfidf_vectorizer.vocabulary_)} termes")
        print(f"📊 Sparsité: {1.0 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.3f}")
        
        return tfidf_normalized
        
    def transform(self, text):
        """Transforme un nouveau texte"""
        processed = self.preprocess_text(text)
        tfidf_vector = self.tfidf_vectorizer.transform([processed])
        return self.scaler.transform(tfidf_vector.toarray())

# Exemple d'utilisation
processor = TFIDFProcessor(max_features=${config.input_dim})
texts = ${JSON.stringify(trainingTexts.split('\n').slice(0, 3))}

# Entraînement et transformation
X = processor.fit_transform(texts)
print(f"Matrice d'entrée pour l'autoencoder: {X.shape}")`,

      'autoencoder': `# Étape 3: Architecture de l'Autoencoder
import numpy as np

class SimpleAutoencoder:
    def __init__(self, input_dim=${config.input_dim}, encoding_dim=${config.encoding_dim}):
        """
        Autoencoder simple avec NumPy
        
        Architecture: Input(${config.input_dim}) → Hidden(512,128) → Encoded(${config.encoding_dim}) → Hidden(128,512) → Output(${config.input_dim})
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Initialisation des poids (Xavier/Glorot)
        np.random.seed(42)
        
        # Encoder: input → 512 → 128 → encoding_dim
        self.W1 = np.random.randn(input_dim, 512) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(512)
        
        self.W2 = np.random.randn(512, 128) * np.sqrt(2.0 / 512)
        self.b2 = np.zeros(128)
        
        self.W_encode = np.random.randn(128, encoding_dim) * np.sqrt(2.0 / 128)
        self.b_encode = np.zeros(encoding_dim)
        
        # Decoder: encoding_dim → 128 → 512 → output
        self.W_decode = np.random.randn(encoding_dim, 128) * np.sqrt(2.0 / encoding_dim)
        self.b_decode = np.zeros(128)
        
        self.W3 = np.random.randn(128, 512) * np.sqrt(2.0 / 128)
        self.b3 = np.zeros(512)
        
        self.W4 = np.random.randn(512, input_dim) * np.sqrt(2.0 / 512)
        self.b4 = np.zeros(input_dim)
        
    def relu(self, x):
        """Fonction d'activation ReLU"""
        return np.maximum(0, x)
        
    def relu_derivative(self, x):
        """Dérivée de ReLU"""
        return (x > 0).astype(float)
        
    def forward(self, X):
        """
        Propagation avant
        
        Args:
            X (np.ndarray): Données d'entrée (batch_size, input_dim)
            
        Returns:
            tuple: (encoded, decoded, activations)
        """
        # Encoder
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.relu(z2)
        
        z_encode = np.dot(a2, self.W_encode) + self.b_encode
        encoded = self.relu(z_encode)  # Représentation compressée
        
        # Decoder
        z_decode = np.dot(encoded, self.W_decode) + self.b_decode
        a_decode = self.relu(z_decode)
        
        z3 = np.dot(a_decode, self.W3) + self.b3
        a3 = self.relu(z3)
        
        z4 = np.dot(a3, self.W4) + self.b4
        decoded = z4  # Sortie linéaire pour la reconstruction
        
        # Sauvegarder les activations pour la rétropropagation
        activations = {
            'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2,
            'z_encode': z_encode, 'encoded': encoded,
            'z_decode': z_decode, 'a_decode': a_decode,
            'z3': z3, 'a3': a3, 'z4': z4, 'decoded': decoded
        }
        
        return encoded, decoded, activations
        
    def encode(self, X):
        """Encode seulement (pour l'inférence)"""
        encoded, _, _ = self.forward(X)
        return encoded
        
    def decode(self, encoded):
        """Décode seulement"""
        z_decode = np.dot(encoded, self.W_decode) + self.b_decode
        a_decode = self.relu(z_decode)
        
        z3 = np.dot(a_decode, self.W3) + self.b3
        a3 = self.relu(z3)
        
        z4 = np.dot(a3, self.W4) + self.b4
        return z4

# Création de l'autoencoder
autoencoder = SimpleAutoencoder()
print(f"🤖 Autoencoder créé:")
print(f"   📥 Input: {autoencoder.input_dim} dimensions")
print(f"   🔄 Encoded: {autoencoder.encoding_dim} dimensions") 
print(f"   📤 Output: {autoencoder.input_dim} dimensions")
print(f"   📊 Compression: {autoencoder.input_dim / autoencoder.encoding_dim:.1f}:1")`,

      'training': `# Étape 4: Entraînement X → X
import numpy as np

def train_autoencoder(autoencoder, X, epochs=${config.epochs}, learning_rate=${config.learning_rate}, batch_size=${config.batch_size}):
    """
    Entraîne l'autoencoder avec la descente de gradient
    
    Args:
        autoencoder: Instance de l'autoencoder
        X (np.ndarray): Données d'entraînement (n_samples, input_dim)
        epochs (int): Nombre d'époques
        learning_rate (float): Taux d'apprentissage
        batch_size (int): Taille des lots
        
    Returns:
        list: Historique des pertes
    """
    n_samples = X.shape[0]
    history = []
    
    print(f"🚀 Début entraînement: {epochs} époques, lr={learning_rate}")
    print(f"📊 Données: {X.shape}, batch_size={batch_size}")
    
    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        
        # Mélanger les données à chaque époque
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        
        # Traitement par lots (mini-batch gradient descent)
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            X_batch = X_shuffled[i:batch_end]
            current_batch_size = X_batch.shape[0]
            
            # === FORWARD PASS ===
            encoded, decoded, activations = autoencoder.forward(X_batch)
            
            # Calcul de la perte (MSE)
            loss = np.mean((X_batch - decoded) ** 2)
            epoch_loss += loss
            n_batches += 1
            
            # === BACKWARD PASS (Rétropropagation) ===
            
            # Gradient de la perte par rapport à la sortie
            dL_decoded = 2 * (decoded - X_batch) / current_batch_size
            
            # Gradients du decoder
            dL_W4 = np.dot(activations['a3'].T, dL_decoded)
            dL_b4 = np.sum(dL_decoded, axis=0)
            
            dL_a3 = np.dot(dL_decoded, autoencoder.W4.T)
            dL_z3 = dL_a3 * autoencoder.relu_derivative(activations['z3'])
            
            dL_W3 = np.dot(activations['a_decode'].T, dL_z3)
            dL_b3 = np.sum(dL_z3, axis=0)
            
            dL_a_decode = np.dot(dL_z3, autoencoder.W3.T)
            dL_z_decode = dL_a_decode * autoencoder.relu_derivative(activations['z_decode'])
            
            dL_W_decode = np.dot(activations['encoded'].T, dL_z_decode)
            dL_b_decode = np.sum(dL_z_decode, axis=0)
            
            # Gradients de l'encoder
            dL_encoded = np.dot(dL_z_decode, autoencoder.W_decode.T)
            dL_z_encode = dL_encoded * autoencoder.relu_derivative(activations['z_encode'])
            
            dL_W_encode = np.dot(activations['a2'].T, dL_z_encode)
            dL_b_encode = np.sum(dL_z_encode, axis=0)
            
            dL_a2 = np.dot(dL_z_encode, autoencoder.W_encode.T)
            dL_z2 = dL_a2 * autoencoder.relu_derivative(activations['z2'])
            
            dL_W2 = np.dot(activations['a1'].T, dL_z2)
            dL_b2 = np.sum(dL_z2, axis=0)
            
            dL_a1 = np.dot(dL_z2, autoencoder.W2.T)
            dL_z1 = dL_a1 * autoencoder.relu_derivative(activations['z1'])
            
            dL_W1 = np.dot(X_batch.T, dL_z1)
            dL_b1 = np.sum(dL_z1, axis=0)
            
            # === MISE À JOUR DES POIDS ===
            autoencoder.W4 -= learning_rate * dL_W4
            autoencoder.b4 -= learning_rate * dL_b4
            autoencoder.W3 -= learning_rate * dL_W3
            autoencoder.b3 -= learning_rate * dL_b3
            autoencoder.W_decode -= learning_rate * dL_W_decode
            autoencoder.b_decode -= learning_rate * dL_b_decode
            
            autoencoder.W_encode -= learning_rate * dL_W_encode
            autoencoder.b_encode -= learning_rate * dL_b_encode
            autoencoder.W2 -= learning_rate * dL_W2
            autoencoder.b2 -= learning_rate * dL_b2
            autoencoder.W1 -= learning_rate * dL_W1
            autoencoder.b1 -= learning_rate * dL_b1
        
        # Perte moyenne de l'époque
        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)
        
        # Affichage du progrès
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Époque {epoch:3d}/{epochs}: Perte = {avg_loss:.6f}")
    
    print(f"✅ Entraînement terminé!")
    print(f"📉 Perte finale: {history[-1]:.6f}")
    
    return history

# Exemple d'entraînement
# history = train_autoencoder(autoencoder, X_train)`,

      'reconstruction': `# Test de Reconstruction
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_reconstruction(autoencoder, tfidf_processor, text="${testText || 'This product is amazing and works perfectly!'}"):
    """
    Teste la reconstruction d'un texte via l'autoencoder
    
    Args:
        autoencoder: Modèle entraîné
        tfidf_processor: Processeur TF-IDF
        text (str): Texte à reconstruire
        
    Returns:
        dict: Résultats de reconstruction
    """
    print(f"🧪 Test de reconstruction:")
    print(f"📝 Texte: {text}")
    
    # 1. Vectorisation TF-IDF
    tfidf_vector = tfidf_processor.transform(text)
    print(f"📊 Vecteur TF-IDF: {tfidf_vector.shape}")
    
    # 2. Encodage (compression)
    encoded = autoencoder.encode(tfidf_vector)
    print(f"🔄 Encodé: {encoded.shape} (compression {tfidf_vector.shape[1] / encoded.shape[1]:.1f}:1)")
    
    # 3. Décodage (reconstruction)
    reconstructed = autoencoder.decode(encoded)
    print(f"📤 Reconstruit: {reconstructed.shape}")
    
    # 4. Métriques de qualité
    
    # Erreur de reconstruction (MSE)
    mse = np.mean((tfidf_vector - reconstructed) ** 2)
    
    # Similarité cosinus
    cosine_sim = cosine_similarity(tfidf_vector, reconstructed)[0][0]
    
    # Erreur absolue moyenne
    mae = np.mean(np.abs(tfidf_vector - reconstructed))
    
    # Analyse des termes les plus importants
    feature_names = tfidf_processor.tfidf_vectorizer.get_feature_names_out()
    
    # Termes originaux les plus importants
    original_indices = tfidf_vector[0].argsort()[-10:][::-1]
    top_original = [(feature_names[i], float(tfidf_vector[0][i])) 
                   for i in original_indices if tfidf_vector[0][i] > 0]
    
    # Termes reconstruits les plus importants  
    reconstructed_indices = reconstructed[0].argsort()[-10:][::-1]
    top_reconstructed = [(feature_names[i], float(reconstructed[0][i])) 
                        for i in reconstructed_indices if reconstructed[0][i] > 0]
    
    results = {
        'original_text': text,
        'mse': float(mse),
        'cosine_similarity': float(cosine_sim),
        'mae': float(mae),
        'compression_ratio': tfidf_vector.shape[1] / encoded.shape[1],
        'encoded_representation': encoded[0].tolist(),
        'top_original_terms': top_original[:5],
        'top_reconstructed_terms': top_reconstructed[:5]
    }
    
    print(f"📊 Résultats:")
    print(f"   MSE: {mse:.6f}")
    print(f"   Similarité cosinus: {cosine_sim:.4f} ({cosine_sim*100:.1f}%)")
    print(f"   MAE: {mae:.6f}")
    
    print(f"🔍 Termes originaux importants:")
    for term, score in top_original[:3]:
        print(f"   • {term}: {score:.4f}")
        
    print(f"🔍 Termes reconstruits importants:")
    for term, score in top_reconstructed[:3]:
        print(f"   • {term}: {score:.4f}")
    
    return results

# Utilisation
# results = test_reconstruction(autoencoder, processor, "${testText || 'Amazing product quality!'}")`,

      'search': `# Recherche dans l'Espace Compressé
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search_compressed_space(autoencoder, tfidf_processor, corpus_texts, query="${searchQuery || 'good quality product'}", top_k=5):
    """
    Recherche sémantique dans l'espace compressé de l'autoencoder
    
    Args:
        autoencoder: Modèle entraîné
        tfidf_processor: Processeur TF-IDF
        corpus_texts (list): Corpus de textes
        query (str): Requête de recherche
        top_k (int): Nombre de résultats
        
    Returns:
        list: Résultats de recherche
    """
    print(f"🔍 Recherche: '{query}'")
    
    # 1. Encoder la requête
    query_tfidf = tfidf_processor.transform(query)
    query_encoded = autoencoder.encode(query_tfidf)
    print(f"📊 Requête encodée: {query_encoded.shape}")
    
    # 2. Encoder tout le corpus
    corpus_encoded = []
    for text in corpus_texts:
        try:
            text_tfidf = tfidf_processor.transform(text)
            text_encoded = autoencoder.encode(text_tfidf)
            corpus_encoded.append(text_encoded[0])
        except:
            # Texte problématique - utiliser un vecteur zéro
            corpus_encoded.append(np.zeros(autoencoder.encoding_dim))
    
    corpus_encoded = np.array(corpus_encoded)
    print(f"📊 Corpus encodé: {corpus_encoded.shape}")
    
    # 3. Calcul des similarités dans l'espace compressé
    similarities = cosine_similarity(query_encoded, corpus_encoded)[0]
    
    # 4. Tri et sélection des meilleurs résultats
    results = []
    for i, (text, similarity) in enumerate(zip(corpus_texts, similarities)):
        results.append({
            'index': i,
            'text': text,
            'similarity': float(similarity),
            'text_preview': text[:150] + "..." if len(text) > 150 else text
        })
    
    # Trier par similarité décroissante
    results.sort(key=lambda x: x['similarity'], reverse=True)
    top_results = results[:top_k]
    
    print(f"📊 Résultats (top {top_k}):")
    for i, result in enumerate(top_results):
        print(f"   {i+1}. Similarité: {result['similarity']:.4f}")
        print(f"      Texte: {result['text_preview']}")
        print()
    
    # 5. Statistiques de recherche
    stats = {
        'query': query,
        'total_searched': len(corpus_texts),
        'results_found': len([r for r in results if r['similarity'] > 0.01]),
        'avg_similarity': float(np.mean(similarities)),
        'max_similarity': float(np.max(similarities)),
        'compressed_space_dim': autoencoder.encoding_dim
    }
    
    print(f"📈 Statistiques:")
    print(f"   Textes recherchés: {stats['total_searched']}")
    print(f"   Résultats pertinents: {stats['results_found']}")
    print(f"   Similarité moyenne: {stats['avg_similarity']:.4f}")
    print(f"   Dimension recherche: {stats['compressed_space_dim']}D")
    
    return top_results, stats

# Exemple d'utilisation
corpus = ${JSON.stringify(trainingTexts.split('\n').slice(0, 5))}
# results, stats = search_compressed_space(autoencoder, processor, corpus, "${searchQuery || 'amazing product'}")

# Avantages de la recherche dans l'espace compressé:
# 1. 🚀 Plus rapide (${config.encoding_dim}D vs ${config.input_dim}D)
# 2. 💾 Moins de mémoire (compression ${Math.round(config.input_dim / config.encoding_dim)}:1)
# 3. 🎯 Capture les relations sémantiques apprises
# 4. 🔍 Moins de bruit (features importantes préservées)`
    };
    
    return codes[step] || '// Code non disponible pour cette étape';
  };

  const openCodePopup = (step: string) => {
    setSelectedCodeStep(step);
    setShowCodePopup(true);
  };

  const closeCodePopup = () => {
    setShowCodePopup(false);
    setSelectedCodeStep('');
  };

  useEffect(() => {
    loadModelInfo();
  }, []);

  const loadModelInfo = async () => {
    try {
      const response = await fetch(`${API_BASE}/info`);
      const data = await response.json();
      if (data.success) {
        setModelInfo(data.model_info);
      }
    } catch (error) {
      console.error('Erreur chargement info modèle:', error);
    }
  };

  const handleTrainAutoencoder = async () => {
    if (!trainingTexts.trim()) {
      setError('Veuillez fournir des textes d\'entraînement');
      return;
    }

    setIsTraining(true);
    setError('');
    setMessage('');

    try {
      const texts = trainingTexts.split('\n').filter(text => text.trim().length > 10);
      
      if (texts.length < 5) {
        setError('Veuillez fournir au moins 5 textes valides (plus de 10 caractères chacun)');
        setIsTraining(false);
        return;
      }

      const response = await fetch(`${API_BASE}/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          texts: texts,
          config: config
        }),
      });

      const data = await response.json();

      if (data.success) {
        setTrainingResult(data.result);
        setMessage(`✅ Autoencoder entraîné avec succès! Architecture: ${data.result.architecture}`);
        await loadModelInfo(); // Recharger les infos du modèle
      } else {
        setError(data.error || 'Erreur lors de l\'entraînement');
      }
    } catch (error) {
      setError('Erreur de connexion au serveur');
      console.error('Erreur:', error);
    } finally {
      setIsTraining(false);
    }
  };

  const handleEncodeText = async () => {
    if (!testText.trim()) {
      setError('Veuillez saisir un texte à encoder');
      return;
    }

    setIsProcessing(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE}/encode`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: testText }),
      });

      const data = await response.json();

      if (data.success) {
        setEncodedText(data.encoded);
        setMessage(`✅ Texte encodé en ${data.encoding_dim} dimensions`);
      } else {
        setError(data.error || 'Erreur lors de l\'encodage');
      }
    } catch (error) {
      setError('Erreur de connexion au serveur');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReconstructText = async () => {
    if (!testText.trim()) {
      setError('Veuillez saisir un texte à reconstruire');
      return;
    }

    setIsProcessing(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE}/reconstruct`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: testText }),
      });

      const data = await response.json();

      if (data.success) {
        setReconstructionResult(data.reconstruction);
        setMessage('✅ Reconstruction effectuée avec succès');
      } else {
        setError(data.error || 'Erreur lors de la reconstruction');
      }
    } catch (error) {
      setError('Erreur de connexion au serveur');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSearchCompressed = async () => {
    if (!searchQuery.trim()) {
      setError('Veuillez saisir une requête de recherche');
      return;
    }

    setIsProcessing(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          query: searchQuery,
          top_k: 5
        }),
      });

      const data = await response.json();

      if (data.success) {
        setSearchResults(data.results);
        setMessage(`✅ Recherche effectuée dans l'espace compressé`);
      } else {
        setError(data.error || 'Erreur lors de la recherche');
      }
    } catch (error) {
      setError('Erreur de connexion au serveur');
    } finally {
      setIsProcessing(false);
    }
  };

  const loadSampleTexts = async (randomize: boolean = false) => {
    try {
      setIsTraining(true);
      setMessage(`🔄 Chargement de ${datasetSize} avis Amazon/polarity ${randomize ? '(aléatoire)' : ''}...`);
      
      if (useAmazonDataset) {
        // Charger depuis le nouveau endpoint avec paramètres configurables
        const response = await fetch('/api/dataset/amazon', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            max_samples: datasetSize,
            random_sample: randomize || useRandomSample,
            split: 'all'
          }),
        });

        if (response.ok) {
          const data = await response.json();
          if (data.success && data.reviews) {
            const texts = data.reviews.map((review: any) => review.text);
            setTrainingTexts(texts.join('\n'));
            setMessage(`✅ ${data.reviews.length} avis Amazon/polarity chargés ${randomize || useRandomSample ? '(aléatoire)' : ''}`);
            return;
          }
        }
      }
      
      // Fallback vers des exemples Amazon locaux
      const fallbackExamples = [
        "This product is absolutely fantastic and exceeded all my expectations",
        "Outstanding quality and excellent customer service experience", 
        "Perfect item exactly as described with fast shipping",
        "Amazing value for money highly recommend to everyone",
        "Superb build quality and beautiful design love it",
        "Excellent product works perfectly as advertised",
        "Great purchase very satisfied with the quality",
        "Fantastic item arrived quickly and well packaged",
        "Perfect condition and exactly what I needed",
        "Outstanding product quality exceeds expectations",
        "Terrible product completely broken on arrival very disappointed",
        "Awful quality waste of money do not recommend",
        "Poor construction broke after one week of use",
        "Horrible customer service and defective product received",
        "Very poor quality not as described in listing",
        "Terrible experience product arrived damaged and unusable",
        "Awful build quality cheap materials and poor design",
        "Poor value for money overpriced and low quality",
        "Horrible product quality control issues evident",
        "Very disappointing purchase not worth the price"
      ].slice(0, Math.min(datasetSize, 20));
      
      setTrainingTexts(fallbackExamples.join('\n'));
      setMessage(`⚠️ Exemples Amazon/polarity de fallback chargés (${fallbackExamples.length} avis)`);
    } catch (error) {
      console.error('Erreur chargement exemples:', error);
      setMessage('❌ Erreur lors du chargement du dataset');
    } finally {
      setIsTraining(false);
    }
  };

  const handleExtractAndCluster = async () => {
    setIsExtracting(true);
    setIsClustering(true);
    setError('');
    
    try {
      // Étape 5: Extraire tous les vecteurs compressés
      setMessage('🔄 Extraction des vecteurs compressés...');
      
      const extractResponse = await fetch(`${API_BASE}/extract_all`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!extractResponse.ok) {
        throw new Error(`HTTP error! status: ${extractResponse.status}`);
      }

      const extractData = await extractResponse.json();
      if (!extractData.success) {
        throw new Error(extractData.error || 'Erreur lors de l\'extraction');
      }

      setMessage(`✅ ${extractData.count} vecteurs extraits. Application de KMeans...`);
      setIsExtracting(false);

      // Étape 6: Appliquer KMeans
      const clusterResponse = await fetch(`${API_BASE}/kmeans`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          encoded_vectors: extractData.encoded_vectors,
          original_texts: extractData.original_texts,
          n_clusters: nClusters
        }),
      });

      if (!clusterResponse.ok) {
        throw new Error(`HTTP error! status: ${clusterResponse.status}`);
      }

      const clusterData = await clusterResponse.json();
      if (!clusterData.success) {
        throw new Error(clusterData.error || 'Erreur lors du clustering');
      }

      setClusteringResults(clusterData);
      setMessage(`✅ Clustering terminé ! Score silhouette: ${clusterData.silhouette_score.toFixed(3)}`);

    } catch (err) {
      setError(`Erreur clustering: ${err instanceof Error ? err.message : 'Erreur inconnue'}`);
    } finally {
      setIsExtracting(false);
      setIsClustering(false);
    }
  };

  const handleOptimizeClusters = async () => {
    setIsOptimizing(true);
    setError('');
    
    try {
      // Utiliser le nouvel endpoint d'optimisation avancée
      const response = await fetch(`${API_BASE}/optimize_clusters`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          max_clusters: 10,
          use_compressed: true
        })
      });

      if (!response.ok) {
        throw new Error('Erreur lors de l\'optimisation');
      }

      const data = await response.json();
      setOptimizationResults(data.result);
      setNClusters(data.result.recommended_k);
      setMessage(`Optimisation terminée. k recommandé: ${data.result.recommended_k} (${data.result.recommendation_reason})`);

    } catch (err: any) {
      setError(`Erreur optimisation: ${err.message}`);
    } finally {
      setIsOptimizing(false);
    }
  };

  // Nouvelles fonctions pour les fonctionnalités avancées
  const handleTrainOptimized = async () => {
    setIsTraining(true);
    setError('');
    setMessage('');

    try {
      const response = await fetch(`${API_BASE}/train_optimized`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config })
      });

      if (!response.ok) {
        throw new Error('Erreur lors de l\'entraînement optimisé');
      }

      const data = await response.json();
      setTrainingResult(data.result);
      setMessage('Entraînement optimisé terminé avec succès !');
      await loadModelInfo();

    } catch (err: any) {
      setError(`Erreur entraînement optimisé: ${err.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const handleEvaluateModel = async () => {
    setIsEvaluating(true);
    setError('');
    setMessage('');

    try {
      const response = await fetch(`${API_BASE}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });

      if (!response.ok) {
        throw new Error('Erreur lors de l\'évaluation');
      }

      const data = await response.json();
      setEvaluationResults(data.evaluation);
      setMessage(`Évaluation terminée. Qualité: ${data.evaluation.quality_level} (${data.evaluation.quality_score.toFixed(3)})`);

    } catch (err: any) {
      setError(`Erreur évaluation: ${err.message}`);
    } finally {
      setIsEvaluating(false);
    }
  };

  const handleAdvancedAnalysis = async () => {
    setIsAdvancedAnalysis(true);
    setError('');
    setMessage('');

    try {
      const response = await fetch(`${API_BASE}/clustering_advanced`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n_clusters: nClusters,
          use_compressed: true
        })
      });

      if (!response.ok) {
        throw new Error('Erreur lors de l\'analyse avancée');
      }

      const data = await response.json();
      setAdvancedResults(data.result);
      setMessage('Analyse avancée terminée avec succès !');

    } catch (err: any) {
      setError(`Erreur analyse avancée: ${err.message}`);
    } finally {
      setIsAdvancedAnalysis(false);
    }
  };

  const handleSaveModel = async () => {
    if (!modelToSave.trim()) {
      setError('Veuillez entrer un nom pour le modèle');
      return;
    }

    setIsSaving(true);
    setError('');
    setMessage('');

    try {
      const response = await fetch(`${API_BASE}/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: modelToSave })
      });

      if (!response.ok) {
        throw new Error('Erreur lors de la sauvegarde');
      }

      const data = await response.json();
      setMessage(`Modèle sauvegardé: ${data.filepath}`);
      setModelToSave('');

    } catch (err: any) {
      setError(`Erreur sauvegarde: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  const handleLoadModel = async () => {
    if (!modelToLoad.trim()) {
      setError('Veuillez entrer un nom de modèle à charger');
      return;
    }

    setIsLoading(true);
    setError('');
    setMessage('');

    try {
      const response = await fetch(`${API_BASE}/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: modelToLoad })
      });

      if (!response.ok) {
        throw new Error('Erreur lors du chargement');
      }

      const data = await response.json();
      setMessage('Modèle chargé avec succès !');
      setModelToLoad('');
      await loadModelInfo();

    } catch (err: any) {
      setError(`Erreur chargement: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // 🎯 NOUVELLE FONCTION: Entraînement avec régularisation avancée
  const handleTrainRegularized = async () => {
    setIsTrainingRegularized(true);
    setError('');
    setMessage('');
    setRegularizedResults(null);

    try {
      setMessage('🎯 Démarrage entraînement REGULARISE avec techniques avancées...');
      
      const response = await fetch(`${API_BASE}/train_regularized`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          config: {
            ...config,
            // Paramètres de régularisation avancés
            l2_regularization: 0.001,
            dropout_rates: [0.1, 0.2, 0.3],
            use_batch_norm: true,
            early_stopping_patience: 15,
            reduce_lr_on_plateau: true
          }
        })
      });

      const data = await response.json();
      
      if (data.success) {
        setRegularizedResults(data.result);
        setMessage(`🎓 Entraînement REGULARISE terminé avec succès! ${data.advanced_validation}`);
        await loadModelInfo(); // Recharger les infos du modèle
      } else {
        setError(data.error || 'Erreur lors de l\'entraînement régularisé');
      }
    } catch (err) {
      setError('Erreur réseau lors de l\'entraînement régularisé');
    } finally {
      setIsTrainingRegularized(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center space-x-3 mb-4">
            <Brain className="h-12 w-12 text-orange-400" />
            <h1 className="text-4xl font-bold text-white">Autoencoder Training</h1>
          </div>
          <p className="text-slate-300 text-xl">
            🤖 Réseau de neurones pour compression TF-IDF → 64D → TF-IDF (Amazon/Polarity)
          </p>
          <div className="flex items-center justify-center space-x-4 mt-4">
            <div className="flex items-center space-x-2 text-orange-400 bg-orange-500/20 px-3 py-1 rounded-full border border-orange-500/30">
              <span className="text-sm font-medium">🛒 Dataset Amazon/Polarity</span>
            </div>
            <div className="flex items-center space-x-2 text-orange-400 bg-orange-500/20 px-3 py-1 rounded-full border border-orange-500/30">
              <span className="text-sm font-medium">🔧 X → X Training</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="bg-slate-800/90 backdrop-blur-xl rounded-2xl border border-white/10 mb-8">
          <div className="flex border-b border-white/10">
            {[
              { id: 'training', label: 'Entraînement', icon: Brain },
              { id: 'testing', label: 'Test', icon: Play },
              { id: 'clustering', label: 'Clustering', icon: RefreshCw },
              { id: 'advanced', label: 'Avancé', icon: CheckCircle },
              { id: 'search', label: 'Recherche', icon: Search },
              { id: 'info', label: 'Info Modèle', icon: Info }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 flex items-center justify-center space-x-2 px-6 py-4 transition-all ${
                  activeTab === tab.id
                    ? 'bg-orange-500/20 text-orange-400 border-b-2 border-orange-400'
                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                }`}
              >
                <tab.icon className="h-5 w-5" />
                <span className="font-medium">{tab.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Tab Content */}
        <div className="bg-slate-800/90 backdrop-blur-xl rounded-2xl border border-white/10 p-8">
          {activeTab === 'training' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">Configuration de l'Autoencoder</h2>
                <div className="flex gap-2">
                  <button
                    onClick={() => openCodePopup('tfidf')}
                    className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors text-sm"
                  >
                    <Code size={16} />
                    TF-IDF
                  </button>
                  <button
                    onClick={() => openCodePopup('autoencoder')}
                    className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors text-sm"
                  >
                    <Code size={16} />
                    Architecture
                  </button>
                  <button
                    onClick={() => openCodePopup('training')}
                    className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors text-sm"
                  >
                    <Code size={16} />
                    Entraînement
                  </button>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-2">
                  <label className="text-white text-sm font-medium">Dimension d'entrée (TF-IDF)</label>
                  <input
                    type="number"
                    value={config.input_dim}
                    onChange={(e) => setConfig({...config, input_dim: parseInt(e.target.value)})}
                    className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white focus:outline-none focus:border-orange-400"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-white text-sm font-medium">Dimension compressée</label>
                  <input
                    type="number"
                    value={config.encoding_dim}
                    onChange={(e) => setConfig({...config, encoding_dim: parseInt(e.target.value)})}
                    className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white focus:outline-none focus:border-orange-400"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-white text-sm font-medium">Taux d'apprentissage</label>
                  <input
                    type="number"
                    step="0.001"
                    value={config.learning_rate}
                    onChange={(e) => setConfig({...config, learning_rate: parseFloat(e.target.value)})}
                    className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white focus:outline-none focus:border-orange-400"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-white text-sm font-medium">Époques</label>
                  <input
                    type="number"
                    value={config.epochs}
                    onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
                    className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white focus:outline-none focus:border-orange-400"
                  />
                </div>
              </div>

              {/* Configuration du Dataset */}
              <div className="bg-slate-700/50 rounded-xl p-6 mb-6">
                <h3 className="text-white font-medium mb-4 flex items-center">
                  <Database className="h-5 w-5 mr-2 text-orange-400" />
                  Configuration du Dataset Amazon
                </h3>
                
                <div className="grid md:grid-cols-2 gap-6">
                  {/* Sélection de la taille */}
                  <div>
                    <label className="block text-white text-sm font-medium mb-2">
                      Nombre d'avis: {datasetSize.toLocaleString()}
                    </label>
                    <input
                      type="range"
                      min="100"
                      max={maxDatasetSize}
                      step="100"
                      value={datasetSize}
                      onChange={(e) => setDatasetSize(parseInt(e.target.value))}
                      className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex justify-between text-xs text-slate-400 mt-1">
                      <span>100</span>
                      <span>1K</span>
                      <span>3K</span>
                      <span>{maxDatasetSize.toLocaleString()}</span>
                    </div>
                    
                    {/* Boutons de sélection rapide */}
                    <div className="flex flex-wrap gap-2 mt-3">
                      {[100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, maxDatasetSize].map(size => (
                        <button
                          key={size}
                          onClick={() => setDatasetSize(size)}
                          className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                            datasetSize === size
                              ? 'bg-orange-500 text-white'
                              : 'bg-slate-600 text-slate-300 hover:bg-slate-500'
                          }`}
                        >
                          {size >= 1000000 ? `${(size/1000000).toFixed(1)}M` : size >= 1000 ? `${(size/1000)}K` : size.toString()}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Options */}
                  <div>
                    <label className="block text-white text-sm font-medium mb-2">
                      Options de chargement
                    </label>
                    <div className="space-y-3">
                      <label className="flex items-center text-white text-sm">
                        <input
                          type="checkbox"
                          checked={useAmazonDataset}
                          onChange={(e) => setUseAmazonDataset(e.target.checked)}
                          className="rounded border-slate-400 text-orange-500 focus:ring-orange-500 mr-2"
                        />
                        Utiliser le dataset Amazon
                      </label>
                      
                      <label className="flex items-center text-white text-sm">
                        <input
                          type="checkbox"
                          checked={useRandomSample}
                          onChange={(e) => setUseRandomSample(e.target.checked)}
                          className="rounded border-slate-400 text-orange-500 focus:ring-orange-500 mr-2"
                        />
                        Échantillonnage aléatoire
                      </label>
                    </div>
                    
                    {/* Conseils */}
                    <div className="text-xs text-slate-400 bg-slate-600/30 p-3 rounded-lg mt-3">
                      <p className="font-medium mb-1">💡 Conseils d'utilisation:</p>
                      <ul className="space-y-1">
                        <li>• <strong>100-1K:</strong> Tests rapides et prototypage</li>
                        <li>• <strong>5K-10K:</strong> Entraînement équilibré</li>
                        <li>• <strong>50K-100K:</strong> Haute qualité (temps modéré)</li>
                        <li>• <strong>500K+:</strong> Performance maximale (très lent)</li>
                        <li>• <strong>3.6M:</strong> Dataset complet (plusieurs heures)</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <label className="text-white text-sm font-medium">Textes d'entraînement (un par ligne)</label>
                  <div className="flex gap-2">
                    <button
                      onClick={() => loadSampleTexts(false)}
                      disabled={isTraining}
                      className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-500 transition-colors disabled:opacity-50 flex items-center space-x-2"
                    >
                      {isTraining ? (
                        <>
                          <RefreshCw className="h-4 w-4 animate-spin" />
                          <span>Chargement...</span>
                        </>
                      ) : (
                        <>
                          <span>🛒 Charger {datasetSize.toLocaleString()} avis</span>
                        </>
                      )}
                    </button>
                    <button
                      onClick={() => loadSampleTexts(true)}
                      disabled={isTraining}
                      className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center space-x-2"
                      title="Charger de nouveaux exemples aléatoires"
                    >
                      <Shuffle className="h-4 w-4" />
                      <span>Aléatoire</span>
                    </button>
                  </div>
                </div>
                <textarea
                  placeholder="Saisissez vos textes d'entraînement, un par ligne..."
                  value={trainingTexts}
                  onChange={(e) => setTrainingTexts(e.target.value)}
                  rows={8}
                  className="w-full p-4 bg-slate-700 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:border-orange-400 resize-none"
                />
                <div className="text-sm text-slate-400">
                  {trainingTexts.split('\n').filter(t => t.trim().length > 10).length} textes valides
                  {trainingTexts.includes('product') && trainingTexts.includes('quality') && (
                    <span className="ml-2 text-green-400">• Dataset Amazon/Polarity ✅</span>
                  )}
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center space-x-4 mb-4">
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={useOptimizedTraining}
                      onChange={(e) => setUseOptimizedTraining(e.target.checked)}
                      className="w-4 h-4 text-orange-600 bg-slate-700 border-slate-600 rounded focus:ring-orange-500"
                    />
                    <span className="text-white text-sm">Utiliser l'entraînement optimisé (recommandé)</span>
                  </label>
                </div>

                <div className="grid grid-cols-3 gap-4">
                  <button
                    onClick={useOptimizedTraining ? handleTrainOptimized : handleTrainAutoencoder} 
                    disabled={isTraining || isTrainingRegularized}
                    className="px-6 py-4 bg-gradient-to-r from-orange-500 to-red-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                  >
                    {isTraining ? (
                      <>
                        <RefreshCw className="h-5 w-5 animate-spin" />
                        <span>Entraînement...</span>
                      </>
                    ) : (
                      <>
                        <Play className="h-5 w-5" />
                        <span>{useOptimizedTraining ? 'Entraînement Optimisé' : 'Entraînement Standard'}</span>
                      </>
                    )}
                  </button>

                  <button
                    onClick={handleTrainRegularized}
                    disabled={isTraining || isTrainingRegularized}
                    className="px-6 py-4 bg-gradient-to-r from-purple-500 to-indigo-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 border-2 border-purple-400/50"
                    title="🎓 Techniques avancées: L2 + Dropout + Batch Norm + Early Stopping + LR Scheduling"
                  >
                    {isTrainingRegularized ? (
                      <>
                        <RefreshCw className="h-5 w-5 animate-spin" />
                        <span>Régularisation...</span>
                      </>
                    ) : (
                      <>
                        <Brain className="h-5 w-5" />
                        <span>🎓 Entraînement REGULARISE</span>
                      </>
                    )}
                  </button>

                  <button
                    onClick={handleEvaluateModel}
                    disabled={isEvaluating || !modelInfo?.is_trained}
                    className="px-6 py-4 bg-gradient-to-r from-blue-500 to-cyan-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                  >
                    {isEvaluating ? (
                      <>
                        <RefreshCw className="h-5 w-5 animate-spin" />
                        <span>Évaluation...</span>
                      </>
                    ) : (
                      <>
                        <CheckCircle className="h-5 w-5" />
                        <span>Évaluer Qualité</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {trainingResult && (
                <div className="bg-green-500/20 border border-green-500/30 rounded-xl p-6">
                  <div className="flex items-center space-x-2 mb-4">
                    <CheckCircle className="h-6 w-6 text-green-400" />
                    <h3 className="text-white font-bold text-lg">Résultats d'entraînement</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-slate-700 p-4 rounded-lg">
                      <p className="text-slate-400 text-sm">Architecture</p>
                      <p className="text-white font-mono">{trainingResult.architecture}</p>
                    </div>
                    <div className="bg-slate-700 p-4 rounded-lg">
                      <p className="text-slate-400 text-sm">Ratio de compression</p>
                      <p className="text-white font-mono">{trainingResult.compression_ratio?.toFixed(1)}:1</p>
                    </div>
                    <div className="bg-slate-700 p-4 rounded-lg">
                      <p className="text-slate-400 text-sm">Perte finale</p>
                      <p className="text-white font-mono">{trainingResult.final_loss?.toFixed(4)}</p>
                    </div>
                    <div className="bg-slate-700 p-4 rounded-lg">
                      <p className="text-slate-400 text-sm">Erreur reconstruction</p>
                      <p className="text-white font-mono">{trainingResult.reconstruction_error?.toFixed(4)}</p>
                    </div>
                  </div>
                </div>
              )}

              {evaluationResults && (
                <div className="bg-blue-500/20 border border-blue-500/30 rounded-xl p-6">
                  <div className="flex items-center space-x-2 mb-4">
                    <CheckCircle className="h-6 w-6 text-blue-400" />
                    <h3 className="text-white font-bold text-lg">Évaluation de la Qualité</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-slate-700 p-4 rounded-lg">
                      <p className="text-slate-400 text-sm">Qualité Globale</p>
                      <p className="text-white font-mono text-lg">{evaluationResults.quality_level}</p>
                      <p className="text-blue-400 font-mono text-sm">Score: {evaluationResults.quality_score.toFixed(3)}</p>
                    </div>
                    <div className="bg-slate-700 p-4 rounded-lg">
                      <p className="text-slate-400 text-sm">MSE</p>
                      <p className="text-white font-mono">{evaluationResults.mse.toFixed(4)}</p>
                    </div>
                    <div className="bg-slate-700 p-4 rounded-lg">
                      <p className="text-slate-400 text-sm">Similarité Moyenne</p>
                      <p className="text-white font-mono">{(evaluationResults.mean_similarity * 100).toFixed(1)}%</p>
                    </div>
                    <div className="bg-slate-700 p-4 rounded-lg">
                      <p className="text-slate-400 text-sm">Variance Expliquée</p>
                      <p className="text-white font-mono">{(evaluationResults.variance_explained * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                </div>
              )}

              {regularizedResults && (
                <div className="bg-purple-500/20 border border-purple-500/30 rounded-xl p-6">
                  <div className="flex items-center space-x-2 mb-4">
                    <Brain className="h-6 w-6 text-purple-400" />
                    <h3 className="text-white font-bold text-lg">🎓 Résultats Entraînement REGULARISE</h3>
                  </div>
                  
                  {/* Validation des techniques avancées */}
                  <div className="bg-purple-600/20 border border-purple-400/30 rounded-lg p-4 mb-6">
                    <p className="text-purple-300 font-medium mb-2">✅ Techniques avancées implémentées :</p>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                                              {regularizedResults.advanced_techniques_implemented?.map((technique: string, idx: number) => (
                        <div key={idx} className="text-purple-200">{technique}</div>
                      ))}
                    </div>
                  </div>

                  {/* Métriques d'entraînement */}
                  {regularizedResults.training && (
                    <div className="grid grid-cols-3 gap-4 mb-6">
                      <div className="bg-slate-700 p-4 rounded-lg">
                        <p className="text-slate-400 text-sm">Framework</p>
                        <p className="text-white font-mono">{regularizedResults.training.framework}</p>
                      </div>
                      <div className="bg-slate-700 p-4 rounded-lg">
                        <p className="text-slate-400 text-sm">Perte finale</p>
                        <p className="text-white font-mono">{regularizedResults.training.final_loss?.toFixed(6)}</p>
                      </div>
                      <div className="bg-slate-700 p-4 rounded-lg">
                        <p className="text-slate-400 text-sm">Perte validation</p>
                        <p className="text-white font-mono">{regularizedResults.training.final_val_loss?.toFixed(6)}</p>
                      </div>
                      <div className="bg-slate-700 p-4 rounded-lg">
                        <p className="text-slate-400 text-sm">Ratio Overfitting</p>
                        <p className="text-white font-mono">
                          {regularizedResults.training.overfitting_ratio?.toFixed(3)}
                          {regularizedResults.training.overfitting_ratio < 1.2 ? ' ✅' : ' ⚠️'}
                        </p>
                      </div>
                      <div className="bg-slate-700 p-4 rounded-lg">
                        <p className="text-slate-400 text-sm">Epochs utilisés</p>
                        <p className="text-white font-mono">
                          {regularizedResults.training.epochs_trained}/{regularizedResults.training.epochs_max}
                        </p>
                      </div>
                      <div className="bg-slate-700 p-4 rounded-lg">
                        <p className="text-slate-400 text-sm">Early Stopping</p>
                        <p className="text-white font-mono">
                          {regularizedResults.training.regularization_effectiveness?.early_stopping_triggered ? '✅ Activé' : '❌ Non activé'}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Paramètres de régularisation */}
                  {regularizedResults.regularization_summary && (
                    <div className="bg-slate-700/50 p-4 rounded-lg mb-4">
                      <h4 className="text-purple-400 font-medium mb-3">Paramètres de Régularisation</h4>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-slate-400">L2 Regularization (kernel)</p>
                          <p className="text-white font-mono">{regularizedResults.regularization_summary.l2_kernel_regularization}</p>
                        </div>
                        <div>
                          <p className="text-slate-400">L2 Regularization (bias)</p>
                          <p className="text-white font-mono">{regularizedResults.regularization_summary.l2_bias_regularization}</p>
                        </div>
                        <div>
                          <p className="text-slate-400">Dropout Rates</p>
                          <p className="text-white font-mono">[{regularizedResults.regularization_summary.dropout_rates?.join(', ')}]</p>
                        </div>
                        <div>
                          <p className="text-slate-400">Batch Normalization</p>
                          <p className="text-white font-mono">{regularizedResults.regularization_summary.batch_normalization ? '✅ Activé' : '❌ Désactivé'}</p>
                        </div>
                        <div>
                          <p className="text-slate-400">Early Stopping Patience</p>
                          <p className="text-white font-mono">{regularizedResults.regularization_summary.early_stopping_patience}</p>
                        </div>
                        <div>
                          <p className="text-slate-400">LR Scheduling</p>
                          <p className="text-white font-mono">{regularizedResults.regularization_summary.lr_scheduling ? '✅ Activé' : '❌ Désactivé'}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Architecture */}
                  {regularizedResults.architecture && (
                    <div className="bg-slate-700/50 p-4 rounded-lg">
                      <h4 className="text-purple-400 font-medium mb-3">Architecture Régularisée</h4>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-slate-400">Input → Encoding</p>
                          <p className="text-white font-mono">{regularizedResults.architecture.input_dim} → {regularizedResults.architecture.encoding_dim}</p>
                        </div>
                        <div>
                          <p className="text-slate-400">Compression Ratio</p>
                          <p className="text-white font-mono">{regularizedResults.architecture.compression_ratio?.toFixed(1)}:1</p>
                        </div>
                        <div>
                          <p className="text-slate-400">Paramètres totaux</p>
                          <p className="text-white font-mono">{regularizedResults.architecture.total_params?.toLocaleString()}</p>
                        </div>
                        <div>
                          <p className="text-slate-400">Techniques appliquées</p>
                          <p className="text-green-400 font-mono text-xs">{regularizedResults.architecture.advanced_requirements}</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {activeTab === 'testing' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">Test de l'Autoencoder</h2>
                <button
                  onClick={() => openCodePopup('reconstruction')}
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors text-sm"
                >
                  <Code size={16} />
                  Voir Code
                </button>
              </div>
              
              <div className="space-y-4">
                <label className="text-white text-sm font-medium">Texte de test</label>
                <textarea
                  placeholder="Saisissez un texte pour tester l'autoencoder..."
                  value={testText}
                  onChange={(e) => setTestText(e.target.value)}
                  rows={3}
                  className="w-full p-4 bg-slate-700 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:border-orange-400 resize-none"
                />
              </div>

              <div className="flex gap-4">
                <button
                  onClick={handleEncodeText}
                  disabled={isProcessing}
                  className="flex-1 px-6 py-3 bg-blue-500/20 text-blue-400 rounded-xl hover:bg-blue-500/30 transition-colors font-medium border border-blue-500/30 disabled:opacity-50 flex items-center justify-center space-x-2"
                >
                  <Brain className="h-5 w-5" />
                  <span>Encoder</span>
                </button>
                <button
                  onClick={handleReconstructText}
                  disabled={isProcessing}
                  className="flex-1 px-6 py-3 bg-green-500/20 text-green-400 rounded-xl hover:bg-green-500/30 transition-colors font-medium border border-green-500/30 disabled:opacity-50 flex items-center justify-center space-x-2"
                >
                  <RefreshCw className="h-5 w-5" />
                  <span>Reconstruire</span>
                </button>
              </div>

              {encodedText && (
                <div className="bg-blue-500/20 border border-blue-500/30 rounded-xl p-6">
                  <h3 className="text-white font-bold text-lg mb-4">Représentation encodée</h3>
                  <div className="bg-slate-700 p-2 rounded-lg mb-2">
                    <span className="text-blue-400 text-sm">{encodedText.length} dimensions</span>
                  </div>
                  <div className="max-h-32 overflow-y-auto bg-slate-900 p-3 rounded-lg text-xs font-mono text-slate-300">
                    [{encodedText.map(val => val.toFixed(3)).join(', ')}]
                  </div>
                </div>
              )}

              {reconstructionResult && (
                <div className="bg-purple-500/20 border border-purple-500/30 rounded-xl p-6">
                  <h3 className="text-white font-bold text-lg mb-4">Résultats de reconstruction</h3>
                  
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <div className="bg-slate-700 p-4 rounded-lg">
                      <p className="text-slate-400 text-sm">Erreur de reconstruction</p>
                      <p className="text-white font-mono text-lg">{reconstructionResult.reconstruction_error.toFixed(4)}</p>
                    </div>
                    <div className="bg-slate-700 p-4 rounded-lg">
                      <p className="text-slate-400 text-sm">Similarité cosinus</p>
                      <p className="text-white font-mono text-lg">{(reconstructionResult.similarity * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <h4 className="text-purple-400 font-medium mb-3">Termes originaux importants</h4>
                      <div className="space-y-2">
                        {reconstructionResult.top_original_terms.map(([term, score], idx) => (
                          <div key={idx} className="flex justify-between bg-slate-700 p-2 rounded">
                            <span className="text-white">{term}</span>
                            <span className="text-purple-400 font-mono">{score.toFixed(3)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h4 className="text-purple-400 font-medium mb-3">Termes reconstruits importants</h4>
                      <div className="space-y-2">
                        {reconstructionResult.top_reconstructed_terms.map(([term, score], idx) => (
                          <div key={idx} className="flex justify-between bg-slate-700 p-2 rounded">
                            <span className="text-white">{term}</span>
                            <span className="text-purple-400 font-mono">{score.toFixed(3)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'clustering' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">Clustering KMeans</h2>
                <div className="text-sm text-slate-400">
                  Étapes 5-7 : Extraction → KMeans → Analyse
                </div>
              </div>

              <div className="bg-orange-500/10 border border-orange-500/20 rounded-xl p-4 mb-6">
                <h3 className="text-orange-400 font-medium mb-2">📋 Pipeline de Clustering</h3>
                <div className="text-slate-300 text-sm space-y-1">
                  <p><strong>5.</strong> Extraire les vecteurs compressés (X_encoded)</p>
                  <p><strong>6.</strong> Appliquer KMeans sur ces vecteurs</p>
                  <p><strong>7.</strong> Analyser les clusters obtenus</p>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="space-y-2">
                  <label className="text-white text-sm font-medium">Nombre de clusters</label>
                  <input
                    type="number"
                    min="2"
                    max="10"
                    value={nClusters}
                    onChange={(e) => setNClusters(parseInt(e.target.value))}
                    className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white focus:outline-none focus:border-orange-400"
                  />
                </div>
                <div className="flex items-end">
                  <button
                    onClick={handleOptimizeClusters}
                    disabled={isOptimizing || isClustering}
                    className="w-full px-4 py-3 bg-gradient-to-r from-green-500 to-teal-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                  >
                    {isOptimizing ? (
                      <>
                        <RefreshCw className="h-4 w-4 animate-spin" />
                        <span>Optimisation...</span>
                      </>
                    ) : (
                      <>
                        <span>🎯 Auto-Optimiser</span>
                      </>
                    )}
                  </button>
                </div>
                <div className="flex items-end">
                  <button
                    onClick={handleExtractAndCluster}
                    disabled={isClustering || isOptimizing}
                    className="w-full px-6 py-3 bg-gradient-to-r from-purple-500 to-blue-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                  >
                    {isClustering ? (
                      <>
                        <RefreshCw className="h-5 w-5 animate-spin" />
                        <span>{isExtracting ? 'Extraction...' : 'Clustering...'}</span>
                      </>
                    ) : (
                      <>
                        <RefreshCw className="h-5 w-5" />
                        <span>Lancer Clustering</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {optimizationResults && (
                <div className="bg-green-500/20 border border-green-500/30 rounded-xl p-4 mb-6">
                  <h3 className="text-green-400 font-medium mb-2">🎯 Résultats d'optimisation</h3>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-slate-400">Meilleur k (Silhouette)</p>
                      <p className="text-white font-mono">{optimizationResults.best_k_silhouette}</p>
                    </div>
                    <div>
                      <p className="text-slate-400">Score Silhouette</p>
                      <p className="text-white font-mono">{optimizationResults.best_silhouette_score.toFixed(3)}</p>
                    </div>
                    <div>
                      <p className="text-slate-400">Recommandation</p>
                      <p className="text-green-400 font-mono">k = {optimizationResults.recommendation}</p>
                    </div>
                  </div>
                </div>
              )}

              {clusteringResults && (
                <div className="space-y-6">
                  {/* Métriques globales */}
                  <div className="bg-purple-500/20 border border-purple-500/30 rounded-xl p-6">
                    <h3 className="text-white font-bold text-lg mb-4">📊 Métriques de Clustering</h3>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-slate-700 p-4 rounded-lg">
                        <p className="text-slate-400 text-sm">Score Silhouette</p>
                        <p className="text-white font-mono text-lg">{clusteringResults.silhouette_score.toFixed(3)}</p>
                        <p className="text-xs text-slate-400">
                          {clusteringResults.silhouette_score > 0.5 ? '✅ Excellent' : 
                           clusteringResults.silhouette_score > 0.3 ? '🟡 Bon' : '❌ Faible'}
                        </p>
                      </div>
                      <div className="bg-slate-700 p-4 rounded-lg">
                        <p className="text-slate-400 text-sm">Inertie</p>
                        <p className="text-white font-mono text-lg">{clusteringResults.inertia.toFixed(2)}</p>
                        <p className="text-xs text-slate-400">Variance intra-cluster</p>
                      </div>
                      <div className="bg-slate-700 p-4 rounded-lg">
                        <p className="text-slate-400 text-sm">Clusters</p>
                        <p className="text-white font-mono text-lg">{clusteringResults.n_clusters}</p>
                        <p className="text-xs text-slate-400">Groupes identifiés</p>
                      </div>
                    </div>
                  </div>

                  {/* Analyse des clusters */}
                  <div className="bg-blue-500/20 border border-blue-500/30 rounded-xl p-6">
                    <h3 className="text-white font-bold text-lg mb-4">🔍 Analyse des Clusters</h3>
                    <div className="space-y-6">
                      {clusteringResults.clusters_analysis.map((cluster: any, idx: number) => (
                        <div key={idx} className="bg-slate-700 rounded-lg p-4">
                          <div className="flex justify-between items-center mb-4">
                            <h4 className="text-blue-400 font-medium text-lg">
                              Cluster {cluster.cluster_id}
                            </h4>
                            <div className="flex gap-2">
                              <span className="bg-blue-500/20 text-blue-400 px-2 py-1 rounded text-sm">
                                {cluster.size} textes
                              </span>
                              <span className="bg-slate-600 text-slate-300 px-2 py-1 rounded text-sm">
                                {cluster.percentage.toFixed(1)}%
                              </span>
                            </div>
                          </div>

                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <h5 className="text-slate-300 font-medium mb-2">Exemples de textes:</h5>
                              <div className="space-y-2">
                                {cluster.texts.map((text: string, textIdx: number) => (
                                  <div key={textIdx} className="bg-slate-800 p-2 rounded text-sm text-slate-300">
                                    {text.length > 100 ? text.substring(0, 100) + '...' : text}
                                  </div>
                                ))}
                              </div>
                            </div>
                            <div>
                              <h5 className="text-slate-300 font-medium mb-2">Mots les plus fréquents:</h5>
                              <div className="space-y-1">
                                {cluster.most_common_words.slice(0, 8).map(([word, count]: [string, number], wordIdx: number) => (
                                  <div key={wordIdx} className="flex justify-between bg-slate-800 p-1 rounded text-sm">
                                    <span className="text-white">{word}</span>
                                    <span className="text-blue-400">{count}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Interprétation */}
                  <div className="bg-green-500/20 border border-green-500/30 rounded-xl p-6">
                    <h3 className="text-white font-bold text-lg mb-4">💡 Interprétation des Résultats</h3>
                    <div className="space-y-3 text-slate-300">
                      <p><strong>Que regroupent-ils ?</strong></p>
                      <ul className="list-disc list-inside space-y-1 ml-4">
                                                 {clusteringResults.clusters_analysis.map((cluster: any, idx: number) => {
                           const topWords = cluster.most_common_words.slice(0, 5).map(([word]: [string, number]) => word);
                           const positiveWords = ['excellent', 'great', 'perfect', 'amazing', 'outstanding', 'fantastic', 'superb', 'brilliant'];
                           const negativeWords = ['terrible', 'awful', 'poor', 'horrible', 'disappointing', 'defective', 'broken', 'waste'];
                           
                           const positiveCount = topWords.filter((w: string) => positiveWords.includes(w.toLowerCase())).length;
                           const negativeCount = topWords.filter((w: string) => negativeWords.includes(w.toLowerCase())).length;
                           
                           const sentiment = positiveCount > negativeCount ? 'Positif' :
                                           negativeCount > positiveCount ? 'Négatif' : 'Neutre/Mixte';
                           
                           const theme = topWords.includes('service') || topWords.includes('customer') ? 'Service Client' :
                                        topWords.includes('quality') || topWords.includes('product') ? 'Qualité Produit' :
                                        topWords.includes('shipping') || topWords.includes('delivery') ? 'Livraison' : 'Général';
                           
                           return (
                             <li key={idx}>
                               <strong>Cluster {cluster.cluster_id}:</strong> {sentiment} • {theme}
                               <br />
                               <span className="text-sm text-slate-400">
                                 Mots-clés: {topWords.join(', ')} • {cluster.size} avis ({cluster.percentage.toFixed(1)}%)
                               </span>
                             </li>
                           );
                         })}
                      </ul>
                      <p className="mt-4"><strong>Quels mots ressortent ?</strong></p>
                      <p className="text-sm">
                        L'autoencoder a appris à comprendre la sémantique des avis Amazon et les regroupe 
                        selon leur sentiment et leur contenu thématique dans l'espace compressé.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'advanced' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">Fonctionnalités Avancées</h2>
                <div className="text-sm text-slate-400">
                  🔬 Data Science - Niveau M1
                </div>
              </div>

              {/* Section Analyse Avancée */}
              <div className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/20 rounded-xl p-6">
                <h3 className="text-purple-400 font-bold text-lg mb-4">🔬 Analyse Avancée du Clustering</h3>
                <p className="text-slate-300 text-sm mb-4">
                  Analyse complète avec métriques avancées, identification des thèmes et sentiment analysis automatique.
                </p>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-white text-sm font-medium">Nombre de clusters</label>
                    <input
                      type="number"
                      min="2"
                      max="10"
                      value={nClusters}
                      onChange={(e) => setNClusters(parseInt(e.target.value))}
                      className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white focus:outline-none focus:border-purple-400"
                    />
                  </div>
                  <div className="flex items-end">
                    <button
                      onClick={handleAdvancedAnalysis}
                      disabled={isAdvancedAnalysis || !modelInfo?.is_trained}
                      className="w-full px-6 py-3 bg-gradient-to-r from-purple-500 to-indigo-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                    >
                      {isAdvancedAnalysis ? (
                        <>
                          <RefreshCw className="h-5 w-5 animate-spin" />
                          <span>Analyse...</span>
                        </>
                      ) : (
                        <>
                          <CheckCircle className="h-5 w-5" />
                          <span>Analyse Complète</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {advancedResults && (
                  <div className="mt-6 space-y-4">
                    <div className="bg-slate-700 p-4 rounded-lg">
                      <h4 className="text-purple-400 font-medium mb-2">📊 Métriques Avancées</h4>
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <p className="text-slate-400">Silhouette Score</p>
                          <p className="text-white font-mono">{advancedResults.silhouette_score?.toFixed(3)}</p>
                        </div>
                        <div>
                          <p className="text-slate-400">Calinski-Harabasz</p>
                          <p className="text-white font-mono">{advancedResults.calinski_harabasz?.toFixed(2)}</p>
                        </div>
                        <div>
                          <p className="text-slate-400">Davies-Bouldin</p>
                          <p className="text-white font-mono">{advancedResults.davies_bouldin?.toFixed(3)}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Section Sauvegarde/Chargement */}
              <div className="bg-gradient-to-r from-green-500/10 to-teal-500/10 border border-green-500/20 rounded-xl p-6">
                <h3 className="text-green-400 font-bold text-lg mb-4">💾 Gestion des Modèles</h3>
                <p className="text-slate-300 text-sm mb-4">
                  Sauvegardez et chargez vos modèles entraînés pour une utilisation ultérieure.
                </p>
                
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h4 className="text-green-400 font-medium">Sauvegarder le modèle</h4>
                    <div className="space-y-2">
                      <input
                        type="text"
                        placeholder="Nom du modèle (ex: autoencoder_v1)"
                        value={modelToSave}
                        onChange={(e) => setModelToSave(e.target.value)}
                        className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:border-green-400"
                      />
                      <button
                        onClick={handleSaveModel}
                        disabled={isSaving || !modelInfo?.is_trained}
                        className="w-full px-4 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                      >
                        {isSaving ? (
                          <>
                            <RefreshCw className="h-4 w-4 animate-spin" />
                            <span>Sauvegarde...</span>
                          </>
                        ) : (
                          <>
                            <Save className="h-4 w-4" />
                            <span>Sauvegarder</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <h4 className="text-green-400 font-medium">Charger un modèle</h4>
                    <div className="space-y-2">
                      <input
                        type="text"
                        placeholder="Nom du modèle à charger"
                        value={modelToLoad}
                        onChange={(e) => setModelToLoad(e.target.value)}
                        className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:border-green-400"
                      />
                      <button
                        onClick={handleLoadModel}
                        disabled={isLoading}
                        className="w-full px-4 py-3 bg-gradient-to-r from-teal-500 to-cyan-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                      >
                        {isLoading ? (
                          <>
                            <RefreshCw className="h-4 w-4 animate-spin" />
                            <span>Chargement...</span>
                          </>
                        ) : (
                          <>
                            <Upload className="h-4 w-4" />
                            <span>Charger</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Section Métriques Détaillées */}
              {evaluationResults && (
                <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border border-blue-500/20 rounded-xl p-6">
                  <h3 className="text-blue-400 font-bold text-lg mb-4">📈 Métriques Détaillées</h3>
                  
                  <div className="grid grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <h4 className="text-blue-400 font-medium">Qualité de Reconstruction</h4>
                      <div className="space-y-3">
                        <div className="bg-slate-700 p-3 rounded-lg">
                          <div className="flex justify-between items-center">
                            <span className="text-slate-400 text-sm">MSE (Mean Squared Error)</span>
                            <span className="text-white font-mono">{evaluationResults.mse?.toFixed(4)}</span>
                          </div>
                        </div>
                        <div className="bg-slate-700 p-3 rounded-lg">
                          <div className="flex justify-between items-center">
                            <span className="text-slate-400 text-sm">MAE (Mean Absolute Error)</span>
                            <span className="text-white font-mono">{evaluationResults.mae?.toFixed(4)}</span>
                          </div>
                        </div>
                        <div className="bg-slate-700 p-3 rounded-lg">
                          <div className="flex justify-between items-center">
                            <span className="text-slate-400 text-sm">RMSE (Root Mean Squared Error)</span>
                            <span className="text-white font-mono">{evaluationResults.rmse?.toFixed(4)}</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <h4 className="text-blue-400 font-medium">Analyse Sémantique</h4>
                      <div className="space-y-3">
                        <div className="bg-slate-700 p-3 rounded-lg">
                          <div className="flex justify-between items-center">
                            <span className="text-slate-400 text-sm">Similarité Cosinus Moyenne</span>
                            <span className="text-white font-mono">{(evaluationResults.mean_similarity * 100)?.toFixed(1)}%</span>
                          </div>
                        </div>
                        <div className="bg-slate-700 p-3 rounded-lg">
                          <div className="flex justify-between items-center">
                            <span className="text-slate-400 text-sm">Variance Expliquée</span>
                            <span className="text-white font-mono">{(evaluationResults.variance_explained * 100)?.toFixed(1)}%</span>
                          </div>
                        </div>
                        <div className="bg-slate-700 p-3 rounded-lg">
                          <div className="flex justify-between items-center">
                            <span className="text-slate-400 text-sm">Score de Qualité Global</span>
                            <span className={`font-mono ${
                              evaluationResults.quality_score > 0.7 ? 'text-green-400' :
                              evaluationResults.quality_score > 0.4 ? 'text-yellow-400' : 'text-red-400'
                            }`}>
                              {evaluationResults.quality_score?.toFixed(3)} ({evaluationResults.quality_level})
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Section Optimisation */}
              {optimizationResults && (
                <div className="bg-gradient-to-r from-yellow-500/10 to-orange-500/10 border border-yellow-500/20 rounded-xl p-6">
                  <h3 className="text-yellow-400 font-bold text-lg mb-4">🎯 Résultats d'Optimisation</h3>
                  
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-slate-700 p-4 rounded-lg">
                      <p className="text-slate-400 text-sm">Méthode du Coude</p>
                      <p className="text-white font-mono text-lg">k = {optimizationResults.elbow_k}</p>
                      <p className="text-xs text-slate-400">Point d'inflexion optimal</p>
                    </div>
                    <div className="bg-slate-700 p-4 rounded-lg">
                      <p className="text-slate-400 text-sm">Meilleur Silhouette</p>
                      <p className="text-white font-mono text-lg">k = {optimizationResults.recommended_k}</p>
                      <p className="text-xs text-slate-400">Score: {optimizationResults.best_silhouette_score?.toFixed(3)}</p>
                    </div>
                    <div className="bg-slate-700 p-4 rounded-lg">
                      <p className="text-slate-400 text-sm">Recommandation</p>
                      <p className="text-yellow-400 font-mono text-lg">k = {optimizationResults.recommended_k}</p>
                      <p className="text-xs text-slate-400">{optimizationResults.recommendation_reason}</p>
                    </div>
                  </div>

                  {optimizationResults.scores && (
                    <div className="mt-4">
                      <h4 className="text-yellow-400 font-medium mb-2">Scores par nombre de clusters</h4>
                      <div className="bg-slate-700 p-3 rounded-lg">
                        <div className="grid grid-cols-4 gap-2 text-xs">
                          <div className="text-slate-400 font-medium">k</div>
                          <div className="text-slate-400 font-medium">Inertie</div>
                          <div className="text-slate-400 font-medium">Silhouette</div>
                          <div className="text-slate-400 font-medium">Statut</div>
                          {optimizationResults.scores.map((score: any, idx: number) => (
                            <React.Fragment key={idx}>
                              <div className="text-white font-mono">{score.k}</div>
                              <div className="text-white font-mono">{score.inertia?.toFixed(2)}</div>
                              <div className="text-white font-mono">{score.silhouette?.toFixed(3)}</div>
                              <div className={`font-mono text-xs ${
                                score.k === optimizationResults.recommended_k ? 'text-yellow-400' : 'text-slate-400'
                              }`}>
                                {score.k === optimizationResults.recommended_k ? '✅ Optimal' : ''}
                              </div>
                            </React.Fragment>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {activeTab === 'search' && (
            <div className="space-y-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">Recherche dans l'Espace Compressé</h2>
                <button
                  onClick={() => openCodePopup('search')}
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors text-sm"
                >
                  <Code size={16} />
                  Voir Code
                </button>
              </div>
              
              <div className="space-y-4">
                <label className="text-white text-sm font-medium">Requête de recherche</label>
                <div className="flex gap-4">
                  <input
                    placeholder="Rechercher dans l'espace compressé..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="flex-1 p-3 bg-slate-700 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:border-orange-400"
                  />
                  <button
                    onClick={handleSearchCompressed}
                    disabled={isProcessing}
                    className="px-6 py-3 bg-green-500/20 text-green-400 rounded-xl hover:bg-green-500/30 transition-colors font-medium border border-green-500/30 disabled:opacity-50 flex items-center space-x-2"
                  >
                    <Search className="h-5 w-5" />
                    <span>Rechercher</span>
                  </button>
                </div>
              </div>

              {searchResults.length > 0 && (
                <div className="bg-green-500/20 border border-green-500/30 rounded-xl p-6">
                  <h3 className="text-white font-bold text-lg mb-4">Résultats de recherche</h3>
                  <div className="space-y-4">
                    {searchResults.map((result, idx) => (
                      <div key={idx} className="bg-slate-700 rounded-lg p-4 border-l-4 border-green-400">
                        <div className="flex justify-between items-start mb-2">
                          <span className="text-green-400 font-medium">#{idx + 1}</span>
                          <span className="text-green-400 bg-green-500/20 px-2 py-1 rounded text-sm">
                            {(result.similarity * 100).toFixed(1)}%
                          </span>
                        </div>
                        <p className="text-white">{result.text_preview}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'info' && (
            <div className="space-y-8">
              <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-white">Informations du Modèle</h2>
                <button
                  onClick={loadModelInfo}
                  className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-500 transition-colors flex items-center space-x-2"
                >
                  <Info className="h-4 w-4" />
                  <span>Actualiser</span>
                </button>
              </div>

              {modelInfo && (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div className={`p-4 rounded-lg border ${modelInfo.is_trained ? 'bg-green-500/20 border-green-500/30' : 'bg-slate-700 border-slate-600'}`}>
                      <p className="text-white font-medium">
                        {modelInfo.is_trained ? "✅ Modèle Entraîné" : "❌ Non Entraîné"}
                      </p>
                    </div>
                    <div className={`p-4 rounded-lg border ${modelInfo.tensorflow_available ? 'bg-blue-500/20 border-blue-500/30' : 'bg-yellow-500/20 border-yellow-500/30'}`}>
                      <p className="text-white font-medium">
                        {modelInfo.tensorflow_available ? "🔥 TensorFlow" : "🔧 NumPy"}
                      </p>
                    </div>
                  </div>

                  {modelInfo.is_trained && (
                    <div className="grid grid-cols-2 gap-6">
                      <div className="bg-slate-700 p-4 rounded-lg">
                        <p className="text-slate-400 text-sm">Architecture</p>
                        <p className="text-white font-mono">{modelInfo.architecture}</p>
                      </div>
                      <div className="bg-slate-700 p-4 rounded-lg">
                        <p className="text-slate-400 text-sm">Taille du corpus</p>
                        <p className="text-white font-mono">{modelInfo.corpus_size}</p>
                      </div>
                      {modelInfo.vocabulary_size && (
                        <div className="bg-slate-700 p-4 rounded-lg">
                          <p className="text-slate-400 text-sm">Vocabulaire</p>
                          <p className="text-white font-mono">{modelInfo.vocabulary_size}</p>
                        </div>
                      )}
                      {modelInfo.total_params && (
                        <div className="bg-slate-700 p-4 rounded-lg">
                          <p className="text-slate-400 text-sm">Paramètres</p>
                          <p className="text-white font-mono">{modelInfo.total_params.toLocaleString()}</p>
                        </div>
                      )}
                    </div>
                  )}

                  <div className="bg-slate-700 p-4 rounded-lg">
                    <p className="text-slate-400 text-sm mb-2">Configuration</p>
                    <pre className="text-xs text-slate-300 bg-slate-900 p-3 rounded overflow-x-auto">
                      {JSON.stringify(modelInfo.config, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Messages */}
        {message && (
          <div className="bg-green-500/20 border border-green-500/30 rounded-xl p-4 mt-6">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-5 w-5 text-green-400" />
              <p className="text-white">{message}</p>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-4 mt-6">
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-5 w-5 text-red-400" />
              <p className="text-white">{error}</p>
            </div>
          </div>
        )}

        {/* Code Popup Modal */}
        {showCodePopup && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-slate-900 rounded-2xl border border-white/10 max-w-4xl w-full max-h-[90vh] overflow-hidden">
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-white/10">
                <div className="flex items-center space-x-3">
                  <Code className="h-6 w-6 text-orange-400" />
                  <h3 className="text-xl font-bold text-white">
                    Code Source - {selectedCodeStep === 'tfidf' ? 'Vectorisation TF-IDF' :
                                 selectedCodeStep === 'autoencoder' ? 'Architecture Autoencoder' :
                                 selectedCodeStep === 'training' ? 'Entraînement X→X' :
                                 selectedCodeStep === 'reconstruction' ? 'Test de Reconstruction' :
                                 selectedCodeStep === 'search' ? 'Recherche Compressée' : 'Code'}
                  </h3>
                </div>
                <button
                  onClick={closeCodePopup}
                  className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                >
                  <X className="h-5 w-5 text-slate-400" />
                </button>
              </div>
              
              {/* Code Content */}
              <div className="p-6 overflow-y-auto max-h-[70vh]">
                <div className="bg-slate-800 rounded-xl border border-slate-600 overflow-hidden">
                  <div className="bg-slate-700 px-4 py-2 border-b border-slate-600 flex items-center justify-between">
                    <span className="text-slate-300 text-sm font-medium">Python</span>
                    <button
                      onClick={() => navigator.clipboard.writeText(getStepCode(selectedCodeStep))}
                      className="text-slate-400 hover:text-white text-sm"
                    >
                      Copier
                    </button>
                  </div>
                  <pre className="p-6 text-sm text-slate-300 overflow-x-auto bg-slate-900">
                    <code>{getStepCode(selectedCodeStep)}</code>
                  </pre>
                </div>
                
                {/* Step Description */}
                <div className="mt-6 p-4 bg-orange-500/10 border border-orange-500/20 rounded-xl">
                  <h4 className="text-orange-400 font-medium mb-2">
                    {selectedCodeStep === 'tfidf' ? '🔢 Étape 2: Vectorisation TF-IDF' :
                     selectedCodeStep === 'autoencoder' ? '🤖 Étape 3: Architecture Autoencoder' :
                     selectedCodeStep === 'training' ? '🎯 Étape 4: Entraînement X→X' :
                     selectedCodeStep === 'reconstruction' ? '🧪 Test de Reconstruction' :
                     selectedCodeStep === 'search' ? '🔍 Recherche dans l\'Espace Compressé' : 'Description'}
                  </h4>
                  <p className="text-slate-300 text-sm">
                    {selectedCodeStep === 'tfidf' ? 'Transformation des textes en vecteurs TF-IDF normalisés pour l\'autoencoder. Utilise scikit-learn avec preprocessing avancé.' :
                     selectedCodeStep === 'autoencoder' ? 'Architecture complète de l\'autoencoder avec NumPy : Encoder (1000→512→128→32) + Decoder (32→128→512→1000).' :
                     selectedCodeStep === 'training' ? 'Entraînement par descente de gradient avec rétropropagation. Le modèle apprend à reconstruire X→X avec compression.' :
                     selectedCodeStep === 'reconstruction' ? 'Test de la qualité de reconstruction avec métriques (MSE, similarité cosinus) et analyse des termes.' :
                     selectedCodeStep === 'search' ? 'Recherche sémantique rapide dans l\'espace compressé 32D au lieu de l\'espace TF-IDF 1000D.' : 'Code source détaillé'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AutoencoderTraining; 