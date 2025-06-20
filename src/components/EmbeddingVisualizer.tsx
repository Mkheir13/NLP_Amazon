import React, { useState, useEffect } from 'react';
import { Search, Eye, Layers, BarChart3, Zap, AlertCircle, CheckCircle, Settings, RefreshCw, Download, Copy, Target } from 'lucide-react';
import { EmbeddingService, EmbeddingModel, EmbeddingVisualization, EmbeddingStats } from '../services/EmbeddingService';

interface EmbeddingVisualizerProps {
  onClose?: () => void;
}

export const EmbeddingVisualizer: React.FC<EmbeddingVisualizerProps> = ({ onClose }) => {
  const [words, setWords] = useState<string>('good,bad,excellent,terrible,amazing,awful,love,hate,beautiful,ugly');
  const [visualizationMethod, setVisualizationMethod] = useState<string>('tsne');
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [availableModels, setAvailableModels] = useState<EmbeddingModel[]>([]);
  const [visualization, setVisualization] = useState<EmbeddingVisualization | null>(null);
  const [stats, setStats] = useState<EmbeddingStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [serviceAvailable, setServiceAvailable] = useState<boolean>(false);

  // Charger les modèles disponibles
  useEffect(() => {
    const loadModels = async () => {
      try {
        const available = await EmbeddingService.isEmbeddingServiceAvailable();
        setServiceAvailable(available);
        
        if (available) {
          const models = await EmbeddingService.getEmbeddingModels();
          setAvailableModels(models);
          if (models.length > 0) {
            setSelectedModel(models[0].id);
          }
        }
      } catch (error) {
        console.error('Erreur chargement modèles:', error);
        setServiceAvailable(false);
      }
    };

    loadModels();
  }, []);

  // Charger les statistiques du modèle sélectionné
  useEffect(() => {
    if (selectedModel && serviceAvailable) {
      loadModelStats();
    }
  }, [selectedModel, serviceAvailable]);

  const loadModelStats = async () => {
    try {
      const modelStats = await EmbeddingService.getEmbeddingStats(selectedModel);
      setStats(modelStats);
    } catch (error) {
      console.error('Erreur chargement statistiques:', error);
    }
  };

  const visualizeEmbeddings = async () => {
    if (!words.trim()) {
      setError('Veuillez entrer des mots à visualiser');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const wordList = words.split(',').map(w => w.trim()).filter(w => w.length > 0);
      
      if (wordList.length === 0) {
        throw new Error('Aucun mot valide trouvé');
      }

      // Envoyer les mots individuels comme des "texts" pour la visualisation
      const viz = await EmbeddingService.visualizeEmbeddings(
        wordList,  // Les mots individuels
        undefined, // pas de labels
        visualizationMethod
      );

      setVisualization(viz);

      // Injecter le graphique Plotly
      if (viz.plot) {
        const plotDiv = document.getElementById('embedding-plot');
        if (plotDiv && (window as any).Plotly) {
          const plotData = JSON.parse(viz.plot);
          (window as any).Plotly.newPlot('embedding-plot', plotData.data, plotData.layout, {
            responsive: true,
            displayModeBar: true
          });
        }
      }

    } catch (error) {
      setError(error instanceof Error ? error.message : 'Erreur lors de la visualisation');
    } finally {
      setIsLoading(false);
    }
  };

  const getMethodDescription = (method: string) => {
    switch (method) {
      case 'pca':
        return 'Analyse en Composantes Principales - Linéaire, rapide';
      case 'tsne':
        return 't-SNE - Non-linéaire, préserve les structures locales';
      case 'umap':
        return 'UMAP - Non-linéaire, préserve structures locales et globales';
      default:
        return '';
    }
  };

  if (!serviceAvailable) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
        <div className="max-w-4xl mx-auto">
          <div className="bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-red-500/30">
            <div className="text-center">
              <AlertCircle className="h-16 w-16 text-red-400 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-white mb-4">Service d'Embedding Non Disponible</h2>
              <p className="text-white/70 mb-6">
                Le service d'embedding n'est pas accessible. Assurez-vous que le backend est démarré 
                et que les dépendances d'embedding sont installées.
              </p>
              <div className="bg-slate-700/50 p-4 rounded-xl text-left">
                <p className="text-cyan-400 font-medium mb-2">Pour activer les embeddings :</p>
                <ul className="text-white/60 space-y-1 text-sm">
                  <li>• Installez les dépendances : pip install gensim sentence-transformers</li>
                  <li>• Redémarrez le backend</li>
                  <li>• Entraînez d'abord un modèle Word2Vec</li>
                </ul>
              </div>
              {onClose && (
                <button
                  onClick={onClose}
                  className="mt-6 px-6 py-3 bg-slate-600 text-white rounded-xl hover:bg-slate-500 transition-colors"
                >
                  Retour
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">Visualiseur d'Embeddings</h1>
            <p className="text-slate-300 text-lg">Explorez les représentations vectorielles des mots</p>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="px-6 py-3 bg-slate-600 text-white rounded-xl hover:bg-slate-500 transition-colors flex items-center space-x-2"
            >
              <span>Retour</span>
            </button>
          )}
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Panneau de contrôle */}
          <div className="lg:col-span-1 space-y-6">
            {/* Configuration */}
            <div className="bg-slate-800/90 backdrop-blur-xl p-6 rounded-2xl border border-white/10">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
                <Settings className="h-6 w-6 text-cyan-400" />
                <span className="text-white">Configuration</span>
              </h3>

              {/* Sélection du modèle */}
              <div className="mb-4">
                <label className="block text-white text-sm font-medium mb-2">
                  Modèle d'embedding
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white focus:outline-none focus:border-cyan-400 focus:bg-slate-600"
                  style={{ color: 'white', backgroundColor: '#334155' }}
                >
                  {availableModels.length === 0 ? (
                    <option value="" style={{ color: 'white', backgroundColor: '#334155' }}>Aucun modèle disponible</option>
                  ) : (
                    availableModels.map(model => (
                      <option key={model.id} value={model.id} style={{ color: 'white', backgroundColor: '#334155' }}>
                        {model.type} - {model.id} ({model.vocabulary_size} mots)
                      </option>
                    ))
                  )}
                </select>
              </div>

              {/* Méthode de visualisation */}
              <div className="mb-4">
                <label className="block text-white text-sm font-medium mb-2">
                  Méthode de réduction
                </label>
                <select
                  value={visualizationMethod}
                  onChange={(e) => setVisualizationMethod(e.target.value)}
                  className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white focus:outline-none focus:border-cyan-400 focus:bg-slate-600"
                  style={{ color: 'white', backgroundColor: '#334155' }}
                >
                  <option value="tsne" style={{ color: 'white', backgroundColor: '#334155' }}>t-SNE</option>
                  <option value="pca" style={{ color: 'white', backgroundColor: '#334155' }}>PCA</option>
                  <option value="umap" style={{ color: 'white', backgroundColor: '#334155' }}>UMAP</option>
                </select>
                <p className="text-slate-300 text-xs mt-1">
                  {getMethodDescription(visualizationMethod)}
                </p>
              </div>

              {/* Mots à visualiser */}
              <div className="mb-6">
                <label className="block text-white text-sm font-medium mb-2">
                  Mots à visualiser (séparés par des virgules)
                </label>
                <textarea
                  value={words}
                  onChange={(e) => setWords(e.target.value)}
                  placeholder="good,bad,excellent,terrible,amazing..."
                  className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:border-cyan-400 focus:bg-slate-600 resize-none"
                  style={{ color: 'white', backgroundColor: '#334155' }}
                  rows={4}
                />
              </div>

              {/* Bouton de visualisation */}
              <button
                onClick={visualizeEmbeddings}
                disabled={isLoading || !selectedModel}
                className="w-full px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="h-5 w-5 animate-spin" />
                    <span>Génération...</span>
                  </>
                ) : (
                  <>
                    <Eye className="h-5 w-5" />
                    <span>Visualiser</span>
                  </>
                )}
              </button>
            </div>

            {/* Statistiques du modèle */}
            {stats && (
              <div className="bg-slate-800/90 backdrop-blur-xl p-6 rounded-2xl border border-white/10">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
                  <BarChart3 className="h-6 w-6 text-green-400" />
                  <span className="text-white">Statistiques</span>
                </h3>

                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Vocabulaire</span>
                    <span className="text-white font-medium">{stats.vocabulary_size.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Dimensions</span>
                    <span className="text-white font-medium">{stats.vector_size}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Type</span>
                    <span className="text-white font-medium">{stats.model_type}</span>
                  </div>
                </div>

                {stats.most_frequent_words && stats.most_frequent_words.length > 0 && (
                  <div className="mt-4">
                    <p className="text-slate-300 text-sm mb-2">Mots les plus fréquents :</p>
                    <div className="space-y-1">
                      {stats.most_frequent_words.slice(0, 5).map(([word, count], index) => (
                        <div key={index} className="flex justify-between text-sm">
                          <span className="text-cyan-400">{word}</span>
                          <span className="text-slate-400">{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Zone de visualisation */}
          <div className="lg:col-span-2">
            <div className="bg-slate-800/90 backdrop-blur-xl p-6 rounded-2xl border border-white/10 h-full">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
                <Target className="h-6 w-6 text-purple-400" />
                <span className="text-white">Visualisation des Embeddings</span>
              </h3>

              {error && (
                <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-4 mb-4">
                  <div className="flex items-center space-x-2">
                    <AlertCircle className="h-5 w-5 text-red-400" />
                    <span className="text-red-400 font-medium">Erreur</span>
                  </div>
                  <p className="text-red-300 mt-1">{error}</p>
                </div>
              )}

              {!visualization && !isLoading && (
                <div className="flex flex-col items-center justify-center h-96 text-center">
                  <Layers className="h-16 w-16 text-slate-500 mb-4" />
                  <p className="text-slate-300 text-lg">Sélectionnez des mots et cliquez sur "Visualiser"</p>
                  <p className="text-slate-400 text-sm mt-2">
                    La visualisation apparaîtra ici avec les embeddings en 2D
                  </p>
                </div>
              )}

              {isLoading && (
                <div className="flex flex-col items-center justify-center h-96">
                  <RefreshCw className="h-12 w-12 text-cyan-400 animate-spin mb-4" />
                  <p className="text-slate-300 text-lg">Génération de la visualisation...</p>
                  <p className="text-slate-400 text-sm mt-2">
                    Calcul des projections {visualizationMethod.toUpperCase()}
                  </p>
                </div>
              )}

              {/* Plotly container */}
              <div id="embedding-plot" className="w-full h-96"></div>

              {/* Informations sur la visualisation */}
              {visualization && (
                <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-green-500/20 p-3 rounded-xl text-center border border-green-500/30">
                    <CheckCircle className="h-6 w-6 text-green-400 mx-auto mb-1" />
                    <p className="text-white font-bold text-lg">{visualization.words_count}</p>
                    <p className="text-green-200 text-xs">Mots trouvés</p>
                  </div>
                  <div className="bg-blue-500/20 p-3 rounded-xl text-center border border-blue-500/30">
                    <Zap className="h-6 w-6 text-blue-400 mx-auto mb-1" />
                    <p className="text-white font-bold text-lg">{visualization.method.toUpperCase()}</p>
                    <p className="text-blue-200 text-xs">Méthode</p>
                  </div>
                  <div className="bg-purple-500/20 p-3 rounded-xl text-center border border-purple-500/30">
                    <Layers className="h-6 w-6 text-purple-400 mx-auto mb-1" />
                    <p className="text-white font-bold text-lg">2D</p>
                    <p className="text-purple-200 text-xs">Dimensions</p>
                  </div>
                  <div className="bg-red-500/20 p-3 rounded-xl text-center border border-red-500/30">
                    <AlertCircle className="h-6 w-6 text-red-400 mx-auto mb-1" />
                    <p className="text-white font-bold text-lg">{visualization.words_not_found.length}</p>
                    <p className="text-red-200 text-xs">Mots manqués</p>
                  </div>
                </div>
              )}

              {visualization && visualization.words_not_found.length > 0 && (
                <div className="mt-4 bg-yellow-500/20 border border-yellow-500/50 rounded-xl p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <AlertCircle className="h-5 w-5 text-yellow-400" />
                    <p className="text-yellow-200 font-bold">Mots non trouvés dans le vocabulaire</p>
                  </div>
                  <p className="text-white bg-slate-700 p-2 rounded text-sm">
                    {visualization.words_not_found.join(', ')}
                  </p>
                  <p className="text-yellow-100 text-xs mt-2">
                    💡 Ces mots n'apparaissent pas dans le dataset d'entraînement Amazon
                  </p>
                </div>
              )}

              {/* Suggestions d'utilisation */}
              {!visualization && !isLoading && (
                <div className="mt-6 bg-cyan-500/20 border border-cyan-500/30 rounded-xl p-4">
                  <h4 className="text-white font-bold mb-3 flex items-center space-x-2">
                    <Eye className="h-5 w-5 text-cyan-400" />
                    <span>Exemples à essayer :</span>
                  </h4>
                  <div className="space-y-2 text-slate-200 text-sm">
                    <div className="bg-slate-700 p-2 rounded">
                      <strong className="text-cyan-300">Sentiment :</strong> happy,sad,joy,anger,love,hate,excited,disappointed
                    </div>
                    <div className="bg-slate-700 p-2 rounded">
                      <strong className="text-green-300">Qualité :</strong> good,bad,excellent,terrible,amazing,awful,perfect,horrible
                    </div>
                    <div className="bg-slate-700 p-2 rounded">
                      <strong className="text-purple-300">Produits :</strong> phone,laptop,camera,headphones,speaker,tablet,watch,charger
                    </div>
                  </div>
                </div>
              )}

              {/* Explication de la visualisation */}
              {visualization && (
                <div className="mt-4 bg-blue-500/20 border border-blue-500/30 rounded-xl p-4">
                  <h4 className="text-white font-bold mb-2 flex items-center space-x-2">
                    <Target className="h-5 w-5 text-blue-400" />
                    <span>Comment interpréter cette visualisation ?</span>
                  </h4>
                  <div className="space-y-2 text-slate-200 text-sm">
                    <p>📊 <strong>Chaque point</strong> représente un mot dans l'espace vectoriel TF-IDF</p>
                    <p>📍 <strong>Distance</strong> : Plus les mots sont proches, plus ils sont sémantiquement similaires</p>
                    <p>🎯 <strong>Clusters</strong> : Les groupes de mots partagent des contextes similaires</p>
                    <p>🔍 <strong>Méthode {visualization.method.toUpperCase()}</strong> : Réduction de {visualization.words_count} dimensions vers 2D</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Script Plotly */}
      <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </div>
  );
}; 