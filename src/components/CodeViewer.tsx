import React, { useState } from 'react';
import { Code2, Copy, CheckCircle, Eye, EyeOff, ChevronDown, ChevronRight } from 'lucide-react';

interface CodeSection {
  title: string;
  language: string;
  code: string;
  description: string;
}

interface CodeViewerProps {
  stepId: string;
  isVisible?: boolean;
}

const CodeViewer: React.FC<CodeViewerProps> = ({ stepId, isVisible = false }) => {
  const [showCode, setShowCode] = useState(isVisible);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set([0]));

  const getCodeForStep = (step: string): CodeSection[] => {
    switch (step) {
      case 'dataset':
        return [
          {
            title: 'Chargement du Dataset Amazon',
            language: 'typescript',
            description: 'Code pour charger et filtrer les avis Amazon Polarity',
            code: `// Chargement du dataset Amazon Polarity
const loadDataset = async () => {
  setIsLoadingDataset(true);
  try {
    console.log('Chargement du dataset Amazon Polarity...');
    const loadedReviews = await DatasetLoader.loadAmazonPolarityDataset(1000);
    setReviews(loadedReviews);
    setFilteredReviews(loadedReviews);
    console.log(\`Dataset chargé : \${loadedReviews.length} avis\`);
    
    // Marquer le dataset comme disponible
    setUserProgress(prev => ({ ...prev, hasDataset: true }));
    markStepCompleted('dataset');
  } catch (error) {
    console.error('Erreur lors du chargement du dataset:', error);
  } finally {
    setIsLoadingDataset(false);
  }
};

// Filtrage des avis par sentiment et recherche
useEffect(() => {
  let filtered = reviews;

  if (sentimentFilter !== 'all') {
    filtered = filtered.filter(r => r.sentiment === sentimentFilter);
  }

  if (searchQuery) {
    filtered = filtered.filter(r => 
      r.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.title.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }

  setFilteredReviews(filtered);
}, [reviews, sentimentFilter, searchQuery]);`
          },
          {
            title: 'Interface de Sélection',
            language: 'typescript',
            description: 'Interface React pour explorer et sélectionner les avis',
            code: `// Sélection d'un avis pour analyse
const selectReview = (review: Review) => {
  setSelectedReview(review);
  analyzeText(review.text);
  
  // Marquer l'exploration du dataset comme terminée
  handleUserAction('dataset_explored', { selectedReview: review });
};

// Avis aléatoire
const getRandomReview = () => {
  if (filteredReviews.length === 0) return;
  const randomIndex = Math.floor(Math.random() * filteredReviews.length);
  const review = filteredReviews[randomIndex];
  selectReview(review);
};

// Rendu de la grille d'avis
{filteredReviews.slice(0, 12).map((review, index) => (
  <div
    key={index}
    onClick={() => selectReview(review)}
    className="group p-4 bg-slate-700/50 rounded-xl border border-slate-600/30 hover:border-cyan-500/50 transition-all cursor-pointer"
  >
    <div className="flex items-center justify-between mb-3">
      <span className={\`px-3 py-1 rounded-full text-xs font-medium \${getSentimentStyle(review.sentiment)}\`}>
        {review.sentiment === 'positive' ? '😊 Positif' : '😞 Négatif'}
      </span>
      <Star className="h-4 w-4 text-yellow-400" />
    </div>
    <p className="text-white text-sm line-clamp-3 group-hover:text-cyan-100 transition-colors">
      {review.text}
    </p>
  </div>
))}`
          }
        ];

      case 'preprocess':
        return [
          {
            title: 'Entraînement TF-IDF',
            language: 'typescript',
            description: 'Vectorisation des textes avec TF-IDF pour créer des embeddings',
            code: `// Service d'entraînement TF-IDF
const trainTFIDF = async (texts: string[]) => {
  try {
    const response = await fetch(\`\${API_BASE}/api/embeddings/train/tfidf\`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts })
    });

    if (!response.ok) {
      throw new Error('Erreur lors de l\'entraînement TF-IDF');
    }

    const result = await response.json();
    return result.stats;
  } catch (error) {
    console.error('Erreur TF-IDF:', error);
    throw error;
  }
};

// Génération d'embeddings pour un texte
const getTextEmbedding = async (text: string) => {
  try {
    const response = await fetch(\`\${API_BASE}/api/embeddings/text\`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    const result = await response.json();
    return result.embedding; // Vecteur de features TF-IDF
  } catch (error) {
    console.error('Erreur embedding:', error);
    throw error;
  }
};`
          },
          {
            title: 'Visualisation des Embeddings',
            language: 'typescript',
            description: 'Réduction dimensionnelle et visualisation 2D/3D des embeddings',
            code: `// Visualisation des embeddings avec PCA/t-SNE
const visualizeEmbeddings = async (texts: string[], method = 'pca') => {
  try {
    const response = await fetch(\`\${API_BASE}/api/embeddings/visualize\`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        texts, 
        method, // 'pca' ou 'tsne'
        labels: texts.map((_, i) => \`Text \${i + 1}\`)
      })
    });

    const result = await response.json();
    
    // Données pour Plotly
    const plotData = {
      x: result.visualization.x,
      y: result.visualization.y,
      text: texts.map((text, i) => \`\${i + 1}: \${text.substring(0, 50)}...\`),
      mode: 'markers',
      type: 'scatter',
      marker: {
        size: 8,
        color: result.visualization.colors || 'blue',
        colorscale: 'Viridis'
      }
    };

    return plotData;
  } catch (error) {
    console.error('Erreur visualisation:', error);
    throw error;
  }
};`
          }
        ];

      case 'analyze':
        return [
          {
            title: 'Analyse NLTK + BERT',
            language: 'typescript',
            description: 'Analyse de sentiment avec NLTK VADER et modèles BERT',
            code: `// Analyse complète avec NLTK et BERT
const analyzeWithRealNLP = async (text: string, bertModelId?: string) => {
  try {
    // Analyse NLTK VADER
    const nltkResponse = await fetch(\`\${API_BASE}/api/analyze/nltk\`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const nltkResult = await nltkResponse.json();

    let bertResult = null;
    let comparison = null;

    // Analyse BERT si modèle disponible
    if (bertModelId) {
      try {
        const bertResponse = await fetch(\`\${API_BASE}/api/predict/bert/\${bertModelId}\`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });
        
        if (bertResponse.ok) {
          bertResult = await bertResponse.json();
          
          // Comparaison NLTK vs BERT
          comparison = {
            agreement: nltkResult.result.sentiment === bertResult.prediction.sentiment,
            nltkConfidence: nltkResult.result.confidence,
            bertConfidence: bertResult.prediction.confidence,
            finalSentiment: bertResult.prediction.confidence > nltkResult.result.confidence 
              ? bertResult.prediction.sentiment 
              : nltkResult.result.sentiment
          };
        }
      } catch (bertError) {
        console.warn('BERT non disponible, utilisation NLTK seulement');
      }
    }

    return {
      text,
      nltk: nltkResult.result,
      bert: bertResult?.prediction,
      comparison,
      features: extractFeatures(text),
      keywords: extractKeywords(text)
    };
  } catch (error) {
    console.error('Erreur analyse NLP:', error);
    throw error;
  }
};`
          },
          {
            title: 'Extraction de Features',
            language: 'typescript',
            description: 'Extraction automatique de caractéristiques linguistiques',
            code: `// Extraction de features textuelles
const extractFeatures = (text: string) => {
  const words = text.split(/\\s+/).filter(w => w.length > 0);
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  
  // Dictionnaire de mots émotionnels
  const emotionalWords = {
    positive: ['good', 'great', 'excellent', 'amazing', 'love', 'perfect'],
    negative: ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible']
  };
  
  const positiveCount = words.filter(word => 
    emotionalWords.positive.includes(word.toLowerCase())
  ).length;
  
  const negativeCount = words.filter(word => 
    emotionalWords.negative.includes(word.toLowerCase())
  ).length;
  
  return {
    wordCount: words.length,
    charCount: text.length,
    sentenceCount: sentences.length,
    avgWordsPerSentence: words.length / Math.max(sentences.length, 1),
    positiveWords: positiveCount,
    negativeWords: negativeCount,
    emotionalWords: positiveCount + negativeCount,
    emotionalRatio: (positiveCount + negativeCount) / words.length,
    readabilityScore: calculateReadabilityScore(words, sentences)
  };
};

// Score de lisibilité simplifié
const calculateReadabilityScore = (words: string[], sentences: string[]) => {
  const avgWordsPerSentence = words.length / Math.max(sentences.length, 1);
  const avgSyllablesPerWord = words.reduce((sum, word) => 
    sum + estimateSyllables(word), 0
  ) / words.length;
  
  // Formule Flesch simplifiée
  return 206.835 - (1.015 * avgWordsPerSentence) - (84.6 * avgSyllablesPerWord);
};`
          }
        ];

      case 'train':
        return [
          {
            title: 'Autoencoder avec Régularisation',
            language: 'python',
            description: 'Entraînement d\'un autoencoder avec techniques de régularisation avancées',
            code: `# Autoencoder avec régularisation L2, Dropout et Batch Normalization
def train_autoencoder_regularized(texts, config):
    """
    Entraînement avec toutes les techniques de régularisation
    """
    print("🎯 ========== ENTRAINEMENT REGULARISE ==========")
    print("🎓 Techniques: L2 + Dropout + Batch Norm + Callbacks")
    
    # Phase 1: TF-IDF optimisé
    tfidf_stats = fit_tfidf_optimized(texts)
    print(f"✅ TF-IDF: {tfidf_stats['vocabulary_size']} mots")
    
    # Phase 2: Construction du modèle avec régularisation
    model = build_regularized_autoencoder(
        input_dim=tfidf_stats['feature_count'],
        encoding_dim=256,
        l2_reg=0.001,           # Régularisation L2
        dropout_rates=[0.1, 0.2, 0.3],  # Dropout progressif
        use_batch_norm=True     # Normalisation par batch
    )
    
    # Phase 3: Callbacks avancés
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            min_delta=0.0001,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7
        )
    ]
    
    # Phase 4: Entraînement
    X_train, X_test = train_test_split(embeddings, test_size=0.2)
    
    history = model.fit(
        X_train, X_train,  # Autoencoder: input = output
        validation_data=(X_test, X_test),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    return {
        'model': model,
        'history': history,
        'metrics': evaluate_model_quality(model, X_test),
        'regularization_applied': '✅ L2 + Dropout + Batch Norm'
    }`
          },
          {
            title: 'Architecture du Modèle',
            language: 'python',
            description: 'Architecture détaillée de l\'autoencoder avec couches de régularisation',
            code: `# Architecture complète avec régularisation
def build_regularized_autoencoder(input_dim, encoding_dim, l2_reg, dropout_rates, use_batch_norm):
    """
    Construit un autoencoder avec toutes les techniques de régularisation
    """
    # Couches d'encodage
    encoder_layers = [
        # Couche 1: 2000D → 1024D
        Dense(1024, 
              kernel_regularizer=l2(l2_reg),
              bias_regularizer=l2(l2_reg * 0.5)),
        BatchNormalization(momentum=0.99) if use_batch_norm else None,
        Activation('relu'),
        Dropout(dropout_rates[0]),
        
        # Couche 2: 1024D → 512D  
        Dense(512,
              kernel_regularizer=l2(l2_reg),
              bias_regularizer=l2(l2_reg * 0.5)),
        BatchNormalization(momentum=0.99) if use_batch_norm else None,
        Activation('relu'),
        Dropout(dropout_rates[1]),
        
        # Goulot d'étranglement: 512D → 256D
        Dense(encoding_dim,
              kernel_regularizer=l2(l2_reg),
              name='encoded_layer')
    ]
    
    # Couches de décodage (symétriques)
    decoder_layers = [
        # 256D → 512D
        Dense(512,
              kernel_regularizer=l2(l2_reg)),
        BatchNormalization(momentum=0.99) if use_batch_norm else None,
        Activation('relu'),
        Dropout(dropout_rates[1]),
        
        # 512D → 1024D
        Dense(1024,
              kernel_regularizer=l2(l2_reg)),
        BatchNormalization(momentum=0.99) if use_batch_norm else None,
        Activation('relu'),
        Dropout(dropout_rates[0]),
        
        # 1024D → 2000D (reconstruction)
        Dense(input_dim,
              kernel_regularizer=l2(l2_reg),
              activation='linear')
    ]
    
    # Assemblage du modèle
    model = Sequential([layer for layer in encoder_layers + decoder_layers if layer is not None])
    
    # Optimiseur avec learning rate adaptatif
    optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model`
          }
        ];

      case 'rnn_training':
        return [
          {
            title: 'RNN from Scratch - Architecture',
            language: 'python',
            description: 'Implémentation d\'un RNN from scratch avec PyTorch',
            code: `# RNN from Scratch - Implémentation complète
class RNNFromScratch(nn.Module):
    """
    RNN implémenté from scratch avec PyTorch :
    - Embedding layer → Manual RNN block → Linear output layer
    - CrossEntropyLoss pour classification
    - Métriques détaillées sur validation set
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNFromScratch, self).__init__()
        
        # Couche d'embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Paramètres RNN manuels (pas nn.RNN!)
        self.hidden_dim = hidden_dim
        self.Wxh = nn.Linear(embedding_dim, hidden_dim)  # input to hidden
        self.Whh = nn.Linear(hidden_dim, hidden_dim)     # hidden to hidden
        self.Who = nn.Linear(hidden_dim, output_dim)     # hidden to output
        
        # Activation
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        """
        Forward pass avec dimension checking à chaque étape
        """
        batch_size, seq_len = x.shape
        print(f"📊 Input shape: {x.shape}")
        
        # Embedding layer
        embedded = self.embedding(x)  # [batch, seq_len, embedding_dim]
        print(f"📊 Embedded shape: {embedded.shape}")
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim)
        print(f"📊 Initial hidden shape: {h.shape}")
        
        # Manual RNN processing (séquentiel)
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # [batch, embedding_dim]
            
            # RNN cell computation (MANUAL - pas nn.RNN!)
            h = self.tanh(self.Wxh(x_t) + self.Whh(h))
            print(f"📊 Hidden at t={t}: {h.shape}")
        
        # Output layer
        output = self.Who(h)  # [batch, output_dim]
        print(f"📊 Final output shape: {output.shape}")
        
        return output`
          },
          {
            title: 'Entraînement et Métriques Détaillées',
            language: 'python',
            description: 'Entraînement avec CrossEntropyLoss et métriques détaillées',
            code: `# Entraînement avec métriques détaillées
def train_rnn_with_detailed_metrics(model, train_loader, val_loader, epochs=20):
    """
    Entraînement avec métriques avancées:
    - CrossEntropyLoss (pas MSELoss!)
    - Métriques détaillées sur validation
    - Affichage des métriques every X epochs
    """
    
    # CrossEntropyLoss pour classification
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Métriques tracking
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    print("🚀 ========== ENTRAINEMENT RNN FROM SCRATCH ==========")
    print("📚 Implémentation PyTorch from scratch")
    
    for epoch in range(epochs):
        # === PHASE ENTRAINEMENT ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass avec dimension checking
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Métriques training
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # === PHASE VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calcul des moyennes
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        
        # Stockage pour visualisation
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Affichage every 5 epochs pour suivi
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Époque {epoch:2d}: Train Loss={avg_train_loss:.4f}, Acc={train_acc:.1f}% | "
                  f"Val Loss={avg_val_loss:.4f}, Acc={val_acc:.1f}%")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }`
          },
          {
            title: 'Métriques Détaillées et Visualisation',
            language: 'python',
            description: 'Métriques de précision détaillées et visualisation Matplotlib des courbes d\'apprentissage',
            code: `# Métriques détaillées sur validation
def evaluate_detailed_metrics(model, val_loader):
    """
    Calcul des métriques détaillées de précision sur le validation set
    """
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Métriques détaillées
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, average=None
    )
    
    # Métriques moyennes
    avg_precision = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')[0]
    avg_recall = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')[1]
    avg_f1 = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')[2]
    
    # Matrice de confusion
    cm = confusion_matrix(all_targets, all_predictions)
    
    print("📊 ========== MÉTRIQUES DÉTAILLÉES ==========")
    print(f"✅ Accuracy: {100 * sum(np.array(all_predictions) == np.array(all_targets)) / len(all_targets):.1f}%")
    print(f"✅ Precision (weighted): {avg_precision:.4f}")
    print(f"✅ Recall (weighted): {avg_recall:.4f}")
    print(f"✅ F1-Score (weighted): {avg_f1:.4f}")
    print(f"📊 Confusion Matrix:\\n{cm}")
    
    return {
        'accuracy': 100 * sum(np.array(all_predictions) == np.array(all_targets)) / len(all_targets),
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1,
        'confusion_matrix': cm.tolist()
    }

# Visualisation Matplotlib
def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    """
    Visualisation des courbes d'apprentissage avec Matplotlib
    """
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss curves
    ax1.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training vs Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training vs Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rnn_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Courbes d'apprentissage sauvegardées: rnn_learning_curves.png")`
          }
        ];

      case 'visualize':
        return [
          {
            title: 'Clustering Avancé',
            language: 'python',
            description: 'Clustering KMeans avec métriques de qualité et optimisation automatique',
            code: `# Clustering avec métriques avancées
def advanced_clustering_analysis(encoded_vectors, n_clusters=4):
    """
    Analyse de clustering complète avec métriques de qualité
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(encoded_vectors)
    
    # Métriques de qualité du clustering
    metrics = {
        'silhouette_score': silhouette_score(encoded_vectors, cluster_labels),
        'calinski_harabasz': calinski_harabasz_score(encoded_vectors, cluster_labels),
        'davies_bouldin': davies_bouldin_score(encoded_vectors, cluster_labels),
        'inertia': kmeans.inertia_
    }
    
    # Analyse des clusters
    cluster_analysis = []
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_vectors = encoded_vectors[cluster_mask]
        cluster_center = kmeans.cluster_centers_[i]
        
        # Statistiques du cluster
        cluster_stats = {
            'cluster_id': i,
            'size': np.sum(cluster_mask),
            'percentage': (np.sum(cluster_mask) / len(cluster_labels)) * 100,
            'center': cluster_center.tolist(),
            'variance': np.var(cluster_vectors, axis=0).mean(),
            'compactness': np.mean(np.linalg.norm(cluster_vectors - cluster_center, axis=1))
        }
        cluster_analysis.append(cluster_stats)
    
    return {
        'labels': cluster_labels.tolist(),
        'centers': kmeans.cluster_centers_.tolist(),
        'metrics': metrics,
        'analysis': cluster_analysis,
        'n_clusters': n_clusters
    }`
          },
          {
            title: 'Optimisation Automatique',
            language: 'python',
            description: 'Recherche automatique du nombre optimal de clusters',
            code: `# Optimisation du nombre de clusters
def optimize_clusters(encoded_vectors, max_clusters=10):
    """
    Trouve le nombre optimal de clusters avec méthode du coude + silhouette
    """
    cluster_range = range(2, max_clusters + 1)
    inertias = []
    silhouette_scores = []
    
    for n_clusters in cluster_range:
        # Test avec n clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(encoded_vectors)
        
        # Métriques
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(encoded_vectors, cluster_labels)
        silhouette_scores.append(sil_score)
    
    # Méthode du coude pour l'inertie
    def find_elbow(values):
        n_points = len(values)
        coords = np.column_stack([range(n_points), values])
        
        # Ligne entre premier et dernier point
        line_vec = coords[-1] - coords[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        
        # Distance de chaque point à la ligne
        vec_from_first = coords - coords[0]
        scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
        vec_to_line = vec_from_first - scalar_product[:, np.newaxis] * line_vec_norm
        distances = np.sqrt(np.sum(vec_to_line**2, axis=1))
        
        return np.argmax(distances)
    
    # Optimal selon méthode du coude
    elbow_index = find_elbow(inertias)
    optimal_elbow = cluster_range[elbow_index]
    
    # Optimal selon silhouette
    optimal_silhouette = cluster_range[np.argmax(silhouette_scores)]
    
    return {
        'optimal_elbow': optimal_elbow,
        'optimal_silhouette': optimal_silhouette,
        'recommended': optimal_silhouette,  # Privilégier silhouette
        'scores': {
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'cluster_range': list(cluster_range)
        }
    }`
          }
        ];

      case 'rnn':
        return [
          {
            title: 'RNN From Scratch - Classe Principale',
            language: 'python',
            description: 'Implémentation manuelle d\'un RNN avec PyTorch (sans nn.RNN)',
            code: `# RNN From Scratch - Implémentation complète
class RNNFromScratch(nn.Module):
    """
    RNN implémenté from scratch avec PyTorch
    Implémentation manuelle : pas de nn.RNN, boucle manuelle
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNFromScratch, self).__init__()
        
        # Couche d'embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Paramètres RNN manuels (pas de nn.RNN !)
        self.hidden_dim = hidden_dim
        self.W_ih = nn.Parameter(torch.randn(embedding_dim, hidden_dim) * 0.1)  # Input-to-hidden
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)    # Hidden-to-hidden
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))                       # Bias
        
        # Couche de sortie
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
        # Initialisation Xavier
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation Xavier pour une meilleure convergence"""
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, x):
        """
        Forward pass avec boucle RNN manuelle
        x: (batch_size, seq_len)
        """
        batch_size, seq_len = x.size()
        
        # Embeddings
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # État caché initial
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # BOUCLE RNN MANUELLE (implémentation from scratch)
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (batch_size, embedding_dim)
            
            # Calcul RNN manuel : h_t = tanh(x_t @ W_ih + h_{t-1} @ W_hh + b_h)
            h = torch.tanh(
                torch.matmul(x_t, self.W_ih) +      # Input contribution
                torch.matmul(h, self.W_hh) +        # Hidden contribution  
                self.b_h                            # Bias
            )
        
        # Dropout et classification
        h = self.dropout(h)
        output = self.fc(h)  # (batch_size, output_dim)
        
        return output`
          },
          {
            title: 'Preprocesseur de Texte',
            language: 'python',
            description: 'Construction du vocabulaire et tokenisation pour le RNN',
            code: `# Preprocesseur pour RNN
class TextPreprocessor:
    """
    Préprocesseur pour construire le vocabulaire et tokeniser les textes
    """
    def __init__(self, max_length=100):
        self.max_length = max_length
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        
    def build_vocabulary(self, texts):
        """Construit le vocabulaire à partir des textes"""
        word_counts = {}
        
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Trier par fréquence et ajouter au vocabulaire
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for word, count in sorted_words:
            if word not in self.word_to_idx:
                idx = self.vocab_size
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                self.vocab_size += 1
        
        print(f"✅ Vocabulaire construit: {self.vocab_size} mots")
        print(f"📊 Mots les plus fréquents: {sorted_words[:10]}")
        
        return {
            'vocab_size': self.vocab_size,
            'most_frequent': sorted_words[:10]
        }
    
    def _tokenize(self, text):
        """Tokenisation simple"""
        import re
        # Nettoyer et tokeniser
        text = re.sub(r'[^a-zA-Z\\s]', '', text.lower())
        return text.split()
    
    def texts_to_sequences(self, texts):
        """Convertit les textes en séquences d'indices"""
        sequences = []
        
        for text in texts:
            words = self._tokenize(text)
            sequence = []
            
            for word in words[:self.max_length]:
                idx = self.word_to_idx.get(word, 1)  # 1 = <UNK>
                sequence.append(idx)
            
            # Padding
            while len(sequence) < self.max_length:
                sequence.append(0)  # 0 = <PAD>
            
            sequences.append(sequence)
        
        return torch.tensor(sequences, dtype=torch.long)`
          },
          {
            title: 'Entraînement RNN Complet',
            language: 'python',
            description: 'Pipeline d\'entraînement complet avec validation et métriques',
            code: `# Pipeline d'entraînement RNN
class RNNSentimentAnalyzer:
    """
    Analyseur de sentiment avec RNN from scratch
    """
    def __init__(self):
        self.model = None
        self.preprocessor = TextPreprocessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"💻 Device: {self.device}")
        
    def train(self, texts, labels, config):
        """
        Entraînement complet du RNN
        """
        print("🚀 ========== ENTRAINEMENT RNN FROM SCRATCH ==========")
        print("📚 Implémentation PyTorch from scratch")
        print(f"📊 Données: {len(texts)} échantillons")
        print(f"⚙️ Config: epochs={config['epochs']}, batch_size={config['batch_size']}, lr={config['learning_rate']}")
        
        # Phase 1: Préparation des données
        print(f"📊 Préparation de {len(texts)} échantillons...")
        
        # Construction du vocabulaire
        print("🔤 Construction du vocabulaire...")
        vocab_stats = self.preprocessor.build_vocabulary(texts)
        
        # Conversion en séquences
        X = self.preprocessor.texts_to_sequences(texts)
        y = torch.tensor(labels, dtype=torch.long)
        
        print(f"✅ Données préparées: {X.shape}")
        
        # Phase 2: Création du modèle
        self.model = RNNFromScratch(
            vocab_size=self.preprocessor.vocab_size,
            embedding_dim=128,
            hidden_dim=64,
            output_dim=2
        ).to(self.device)
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🤖 Modèle RNN créé:")
        print(f"   - Vocabulaire: {self.preprocessor.vocab_size} mots")
        print(f"   - Embedding: 128D")
        print(f"   - Hidden: 64D")
        print(f"   - Paramètres: {total_params:,}")
        
        # Phase 3: Split train/validation
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"📊 Split: {len(X_train)} train, {len(X_val)} validation")
        
        # Phase 4: Entraînement
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # Déplacer sur device
        X_train, X_val = X_train.to(self.device), X_val.to(self.device)
        y_train, y_val = y_train.to(self.device), y_val.to(self.device)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(config['epochs']):
            # Training
            self.model.train()
            train_loss, train_correct = 0, 0
            
            for i in range(0, len(X_train), config['batch_size']):
                batch_X = X_train[i:i+config['batch_size']]
                batch_y = y_train[i:i+config['batch_size']]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == batch_y).sum().item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_correct = (val_outputs.argmax(1) == y_val).sum().item()
            
            # Métriques
            train_acc = 100 * train_correct / len(X_train)
            val_acc = 100 * val_correct / len(X_val)
            avg_train_loss = train_loss / (len(X_train) // config['batch_size'])
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Affichage périodique
            if epoch % 5 == 0 or epoch == config['epochs'] - 1:
                print(f"Époque {epoch:2d}: Train Loss={avg_train_loss:.4f}, Acc={train_acc:.1f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.1f}%")
        
        print("✅ ========== ENTRAINEMENT TERMINE ==========")
        
        return {
            'history': history,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'vocab_size': self.preprocessor.vocab_size,
            'architecture': f"Embedding(128D) → RNN(64D) → Dense(2D)"
        }`
          },
          {
            title: 'Prédiction et Analyse',
            language: 'python',
            description: 'Fonction de prédiction avec analyse détaillée des états cachés',
            code: `# Prédiction avec RNN from scratch
def predict_sentiment(self, text):
    """
    Prédiction de sentiment avec analyse détaillée
    """
    if self.model is None:
        raise ValueError("Modèle non entraîné")
    
    self.model.eval()
    with torch.no_grad():
        # Préparation du texte
        sequence = self.preprocessor.texts_to_sequences([text])
        sequence = sequence.to(self.device)
        
        # Prédiction
        output = self.model(sequence)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(1).item()
        confidence = probabilities[0, predicted_class].item()
        
        # Analyse des états cachés (pour debug/visualisation)
        hidden_states = self._get_hidden_states(sequence[0])
        
        sentiment = 'positive' if predicted_class == 1 else 'negative'
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': probabilities[0, 0].item(),
                'positive': probabilities[0, 1].item()
            },
            'hidden_state': hidden_states[-1].tolist(),  # Dernier état caché
            'model_type': 'RNN from scratch (PyTorch)'
        }

def _get_hidden_states(self, sequence):
    """
    Récupère tous les états cachés pour analyse
    """
    self.model.eval()
    with torch.no_grad():
        embedded = self.model.embedding(sequence.unsqueeze(0))
        batch_size, seq_len, _ = embedded.size()
        
        h = torch.zeros(batch_size, self.model.hidden_dim, device=sequence.device)
        hidden_states = []
        
        # Reproduire la boucle RNN pour capturer les états
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h = torch.tanh(
                torch.matmul(x_t, self.model.W_ih) +
                torch.matmul(h, self.model.W_hh) +
                self.model.b_h
            )
            hidden_states.append(h.clone())
        
        return [h.squeeze(0) for h in hidden_states]

# Analyse de séquence détaillée
def analyze_sequence(self, text):
    """
    Analyse détaillée du traitement séquentiel
    """
    if self.model is None:
        raise ValueError("Modèle non entraîné")
    
    words = self.preprocessor._tokenize(text)
    sequence = self.preprocessor.texts_to_sequences([text])[0]
    hidden_states = self._get_hidden_states(sequence)
    
    analysis = []
    for i, (word, hidden) in enumerate(zip(words, hidden_states)):
        analysis.append({
            'position': i,
            'word': word,
            'hidden_norm': float(torch.norm(hidden)),
            'hidden_mean': float(hidden.mean()),
            'hidden_std': float(hidden.std())
        })
    
    return {
        'text': text,
        'word_analysis': analysis,
        'sequence_length': len(words),
        'final_prediction': self.predict_sentiment(text)
    }`
                     },
           {
             title: 'Alternative TensorFlow/Keras',
             language: 'python',
             description: 'Implémentation RNN avec TensorFlow/Keras pour comparaison',
             code: `# Alternative avec TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class RNNTensorFlow:
    """
    RNN avec TensorFlow/Keras pour comparaison avec PyTorch
    """
    def __init__(self, max_features=5000, maxlen=100):
        self.max_features = max_features
        self.maxlen = maxlen
        self.tokenizer = Tokenizer(num_words=max_features)
        self.model = None
        
    def build_model(self, embedding_dim=128, rnn_units=64):
        """
        Construction du modèle RNN avec Keras
        """
        model = models.Sequential([
            # Couche d'embedding
            layers.Embedding(
                input_dim=self.max_features,
                output_dim=embedding_dim,
                input_length=self.maxlen,
                name='embedding'
            ),
            
            # Couche RNN simple (équivalent à notre implémentation from scratch)
            layers.SimpleRNN(
                units=rnn_units,
                dropout=0.2,
                recurrent_dropout=0.2,
                return_sequences=False,  # Seulement la dernière sortie
                name='rnn_layer'
            ),
            
            # Couche de classification
            layers.Dense(64, activation='relu', name='dense_hidden'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax', name='output')
        ])
        
        # Compilation
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_texts(self, texts, labels=None):
        """
        Préprocessing des textes avec Tokenizer Keras
        """
        # Entraînement du tokenizer
        if labels is not None:  # Mode entraînement
            self.tokenizer.fit_on_texts(texts)
        
        # Conversion en séquences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Padding
        X = pad_sequences(sequences, maxlen=self.maxlen, padding='post')
        
        print(f"📊 Vocabulaire TensorFlow: {len(self.tokenizer.word_index)} mots")
        print(f"📊 Forme des données: {X.shape}")
        
        return X
    
    def train(self, texts, labels, validation_split=0.2, epochs=20, batch_size=32):
        """
        Entraînement du modèle RNN TensorFlow
        """
        print("🚀 ========== ENTRAINEMENT RNN TENSORFLOW ==========")
        
        # Préprocessing
        X = self.preprocess_texts(texts, labels)
        y = np.array(labels)
        
        # Construction du modèle
        if self.model is None:
            self.build_model()
        
        print(f"🤖 Modèle TensorFlow créé:")
        print(f"   - Vocabulaire: {len(self.tokenizer.word_index)} mots")
        print(f"   - Architecture: Embedding → SimpleRNN → Dense")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # Entraînement
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ ========== ENTRAINEMENT TENSORFLOW TERMINE ==========")
        
        return {
            'history': history.history,
            'final_train_acc': history.history['accuracy'][-1] * 100,
            'final_val_acc': history.history['val_accuracy'][-1] * 100,
            'vocab_size': len(self.tokenizer.word_index),
            'architecture': "Embedding → SimpleRNN → Dense"
        }
    
    def predict(self, text):
        """
        Prédiction avec le modèle TensorFlow
        """
        if self.model is None:
            raise ValueError("Modèle non entraîné")
        
        # Préprocessing
        X = self.preprocess_texts([text])
        
        # Prédiction
        predictions = self.model.predict(X, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        sentiment = 'positive' if predicted_class == 1 else 'negative'
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': float(predictions[0][0]),
                'positive': float(predictions[0][1])
            },
            'model_type': 'RNN TensorFlow/Keras'
        }

# Comparaison PyTorch vs TensorFlow
def compare_implementations():
    """
    Comparaison des deux implémentations RNN
    """
    print("🔄 ========== COMPARAISON PYTORCH vs TENSORFLOW ==========")
    
    # Données d'exemple
    texts = ["This is great!", "This is terrible!"] * 10
    labels = [1, 0] * 10
    
    # Test PyTorch (from scratch)
    print("\\n🎯 Test PyTorch from scratch:")
    pytorch_rnn = RNNSentimentAnalyzer()
    pytorch_results = pytorch_rnn.train(texts, labels, {'epochs': 10, 'batch_size': 4, 'learning_rate': 0.001})
    
    # Test TensorFlow
    print("\\n🎯 Test TensorFlow/Keras:")
    tf_rnn = RNNTensorFlow()
    tf_results = tf_rnn.train(texts, labels, epochs=10, batch_size=4)
    
    # Comparaison
    print("\\n📊 ========== RESULTATS COMPARATIFS ==========")
    print(f"PyTorch from scratch - Précision finale: {pytorch_results['final_val_acc']:.1f}%")
    print(f"TensorFlow/Keras    - Précision finale: {tf_results['final_val_acc']:.1f}%")
    
    # Test de prédiction
    test_text = "This product is amazing!"
    pytorch_pred = pytorch_rnn.predict_sentiment(test_text)
    tf_pred = tf_rnn.predict(test_text)
    
    print(f"\\n🔍 Prédiction sur: '{test_text}'")
    print(f"PyTorch: {pytorch_pred['sentiment']} ({pytorch_pred['confidence']:.3f})")
    print(f"TensorFlow: {tf_pred['sentiment']} ({tf_pred['confidence']:.3f})")
    
    return {
        'pytorch': pytorch_results,
        'tensorflow': tf_results,
        'predictions': {
            'pytorch': pytorch_pred,
            'tensorflow': tf_pred
        }
    }`
           }
         ];

      default:
        return [
          {
            title: 'Code Non Disponible',
            language: 'text',
            description: 'Aucun code disponible pour cette étape',
            code: '// Code non disponible pour cette étape'
          }
        ];
    }
  };

  const codeSections = getCodeForStep(stepId);

  const copyToClipboard = async (code: string, index: number) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (error) {
      console.error('Erreur copie:', error);
    }
  };

  const toggleSection = (index: number) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSections(newExpanded);
  };

  if (!showCode) {
    return (
      <div className="bg-slate-800/50 border border-slate-600/30 rounded-xl p-4 mb-6">
        <button
          onClick={() => setShowCode(true)}
          className="flex items-center justify-between w-full text-left"
        >
          <div className="flex items-center space-x-3">
            <Code2 className="h-5 w-5 text-blue-400" />
            <div>
              <h3 className="text-white font-medium">Voir le code de cette étape</h3>
              <p className="text-slate-400 text-sm">Implémentation technique détaillée</p>
            </div>
          </div>
          <Eye className="h-5 w-5 text-slate-400" />
        </button>
      </div>
    );
  }

  return (
    <div className="bg-slate-800/90 backdrop-blur-xl border border-slate-600/30 rounded-xl mb-6">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-600/30">
        <div className="flex items-center space-x-3">
          <Code2 className="h-5 w-5 text-blue-400" />
          <div>
            <h3 className="text-white font-medium">Code de l'étape</h3>
            <p className="text-slate-400 text-sm">{codeSections.length} section(s) disponible(s)</p>
          </div>
        </div>
          <button
          onClick={() => setShowCode(false)}
          className="p-2 text-slate-400 hover:text-white transition-colors"
          >
          <EyeOff className="h-5 w-5" />
          </button>
      </div>
          
      {/* Code sections */}
      <div className="p-4 space-y-4">
        {codeSections.map((section, index) => (
          <div key={index} className="border border-slate-600/30 rounded-lg overflow-hidden">
            {/* Section header */}
          <button
              onClick={() => toggleSection(index)}
              className="w-full flex items-center justify-between p-4 bg-slate-700/50 hover:bg-slate-700/70 transition-colors"
            >
              <div className="flex items-center space-x-3">
                {expandedSections.has(index) ? (
                  <ChevronDown className="h-4 w-4 text-blue-400" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-blue-400" />
                )}
                <div className="text-left">
                  <h4 className="text-white font-medium">{section.title}</h4>
                  <p className="text-slate-400 text-sm">{section.description}</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <span className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded text-xs font-mono">
                  {section.language}
                </span>
          <button
                  onClick={(e) => {
                    e.stopPropagation();
                    copyToClipboard(section.code, index);
                  }}
                  className="p-1 text-slate-400 hover:text-white transition-colors"
                >
                  {copiedIndex === index ? (
                    <CheckCircle className="h-4 w-4 text-green-400" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
          </button>
        </div>
            </button>

            {/* Code content */}
            {expandedSections.has(index) && (
              <div className="relative">
                <pre className="p-4 bg-slate-900/50 text-sm overflow-x-auto">
                  <code className="text-slate-300 font-mono leading-relaxed">
                    {section.code}
          </code>
        </pre>
      </div>
            )}
        </div>
        ))}
      </div>
    </div>
  );
};

export default CodeViewer;