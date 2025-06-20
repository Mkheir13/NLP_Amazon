import React, { useState, useEffect, useRef } from 'react';
import { Brain, Target, Search, Settings, RefreshCw, CheckCircle, AlertCircle, BarChart3, Zap, Database, Lightbulb, Eye, Layers, Maximize2, Palette, Grid3X3, TrendingUp } from 'lucide-react';
import { EmbeddingService, EmbeddingModel, EmbeddingVisualization } from '../services/EmbeddingService';
import { DatasetLoader, Review } from '../services/DatasetLoader';

interface EmbeddingHubProps {
  onClose?: () => void;
}

export const EmbeddingHub: React.FC<EmbeddingHubProps> = ({ onClose }) => {
  // États pour l'entraînement
  const [dataSource, setDataSource] = useState<'amazon' | 'custom'>('amazon');
  const [customTexts, setCustomTexts] = useState<string>('');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStats, setTrainingStats] = useState<any>(null);
  const [trainingSuccess, setTrainingSuccess] = useState(false);

  // États pour la visualisation
  const [words, setWords] = useState<string>('good,bad,excellent,terrible,amazing,awful,love,hate,beautiful,ugly');
  const [visualizationMethod, setVisualizationMethod] = useState<string>('tsne');
  const [visualization, setVisualization] = useState<EmbeddingVisualization | null>(null);
  const [isVisualizing, setIsVisualizing] = useState(false);

  // États pour la recherche
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchStats, setSearchStats] = useState<any>(null);

  // États généraux
  const [error, setError] = useState<string | null>(null);
  const [serviceAvailable, setServiceAvailable] = useState<boolean>(false);
  const [availableModels, setAvailableModels] = useState<EmbeddingModel[]>([]);

  // Nouveaux états pour la visualisation améliorée
  const [visualizationSize, setVisualizationSize] = useState<'small' | 'large'>('large');
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [colorScheme, setColorScheme] = useState('viridis');
  const [pointSize, setPointSize] = useState(8);
  const [showLabels, setShowLabels] = useState(true);
  const [clusterAnalysis, setClusterAnalysis] = useState<any>(null);
  const [wordSimilarities, setWordSimilarities] = useState<any[]>([]);
  const plotRef = useRef<HTMLDivElement>(null);

  // Vérifier la disponibilité du service
  useEffect(() => {
    const checkService = async () => {
      try {
        const available = await EmbeddingService.isEmbeddingServiceAvailable();
        setServiceAvailable(available);
        
        if (available) {
          const models = await EmbeddingService.getEmbeddingModels();
          setAvailableModels(models);
        }
      } catch (error) {
        console.error('Erreur service:', error);
        setServiceAvailable(false);
      }
    };

    checkService();
  }, []);

  // Fonction d'entraînement
  const handleTraining = async () => {
    setIsTraining(true);
    setError(null);
    setTrainingSuccess(false);

    try {
      let texts: string[] = [];

      if (dataSource === 'amazon') {
        const reviews = await DatasetLoader.loadAmazonPolarityDataset(1000);
        texts = reviews.map(review => review.text);
      } else {
        texts = customTexts.split('\n').filter(text => text.trim().length > 0);
      }

      if (texts.length === 0) {
        throw new Error('Aucun texte à traiter');
      }

      const response = await EmbeddingService.trainTFIDF(texts);
      setTrainingStats(response.stats);
      setTrainingSuccess(true);

      // Recharger les modèles disponibles
      const models = await EmbeddingService.getEmbeddingModels();
      setAvailableModels(models);

    } catch (error) {
      setError(error instanceof Error ? error.message : 'Erreur lors de l\'entraînement');
    } finally {
      setIsTraining(false);
    }
  };

  // Fonction de visualisation améliorée
  const handleVisualization = async () => {
    if (!words.trim()) {
      setError('Veuillez entrer des mots à visualiser');
      return;
    }

    if (availableModels.length === 0) {
      setError('Aucun modèle disponible. Entraînez d\'abord un modèle TF-IDF.');
      return;
    }

    setIsVisualizing(true);
    setError(null);
    setVisualization(null);
    setClusterAnalysis(null);
    setWordSimilarities([]);

    try {
      const wordList = words.split(',').map(word => word.trim()).filter(word => word.length > 0);
      
      const viz = await EmbeddingService.visualizeEmbeddings(
        wordList,
        undefined,
        visualizationMethod
      );

      setVisualization(viz);

      // Analyse des clusters et similarités
      if (viz.coordinates && wordList.length > 1) {
        await performClusterAnalysis(wordList, viz.coordinates);
        await calculateWordSimilarities(wordList);
      }

      // Injecter le graphique Plotly avec configuration avancée
      setTimeout(() => {
        const plotDiv = document.getElementById('embedding-plot');
        if (plotDiv && (window as any).Plotly && viz.plot) {
          const plotData = JSON.parse(viz.plot);
          
          // Configuration avancée du graphique
          const enhancedLayout = {
            ...plotData.layout,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: 'white', size: 12 },
            showlegend: false,
            hovermode: 'closest',
            xaxis: {
              ...plotData.layout.xaxis,
              gridcolor: 'rgba(255,255,255,0.1)',
              zerolinecolor: 'rgba(255,255,255,0.2)',
              tickfont: { color: 'white' }
            },
            yaxis: {
              ...plotData.layout.yaxis,
              gridcolor: 'rgba(255,255,255,0.1)',
              zerolinecolor: 'rgba(255,255,255,0.2)',
              tickfont: { color: 'white' }
            },
            margin: { l: 40, r: 40, t: 40, b: 40 }
          };

          // Configuration avancée des points
          const enhancedData = plotData.data.map((trace: any) => ({
            ...trace,
            marker: {
              ...trace.marker,
              size: pointSize,
              color: trace.marker.color,
              colorscale: colorScheme,
              line: { width: 1, color: 'rgba(255,255,255,0.5)' },
              opacity: 0.8
            },
            textfont: { color: 'white', size: 10 },
            mode: showLabels ? 'markers+text' : 'markers'
          }));

          (window as any).Plotly.newPlot('embedding-plot', enhancedData, enhancedLayout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToAdd: [{
              name: 'Télécharger PNG',
              icon: (window as any).Plotly.Icons.camera,
              click: function(gd: any) {
                (window as any).Plotly.downloadImage(gd, {
                  format: 'png',
                  width: 1200,
                  height: 800,
                  filename: 'embedding_visualization'
                });
              }
            }],
            displaylogo: false
          });
        }
      }, 100);

    } catch (error) {
      setError(error instanceof Error ? error.message : 'Erreur lors de la visualisation');
    } finally {
      setIsVisualizing(false);
    }
  };

  // Analyse des clusters
  const performClusterAnalysis = async (words: string[], coordinates: number[][]) => {
    try {
      // Calcul simple des distances entre points
      const distances: { [key: string]: number } = {};
      const center = coordinates.reduce((acc, coord) => [
        acc[0] + coord[0] / coordinates.length,
        acc[1] + coord[1] / coordinates.length
      ], [0, 0]);

      words.forEach((word, i) => {
        const coord = coordinates[i];
        const distance = Math.sqrt(
          Math.pow(coord[0] - center[0], 2) + Math.pow(coord[1] - center[1], 2)
        );
        distances[word] = distance;
      });

      const sortedByDistance = Object.entries(distances)
        .sort(([,a], [,b]) => a - b);

      setClusterAnalysis({
        center_words: sortedByDistance.slice(0, 3).map(([word]) => word),
        outlier_words: sortedByDistance.slice(-2).map(([word]) => word),
        total_spread: Math.max(...Object.values(distances)) - Math.min(...Object.values(distances))
      });
    } catch (error) {
      console.error('Erreur analyse cluster:', error);
    }
  };

  // Calcul des similarités entre mots
  const calculateWordSimilarities = async (words: string[]) => {
    try {
      const similarities = [];
      for (let i = 0; i < words.length - 1; i++) {
        for (let j = i + 1; j < words.length; j++) {
          try {
            const result = await EmbeddingService.compareTexts(words[i], words[j]);
            similarities.push({
              word1: words[i],
              word2: words[j],
              similarity: result.similarity
            });
          } catch (error) {
            // Ignorer les erreurs de comparaison individuelle
          }
        }
      }
      
      similarities.sort((a, b) => b.similarity - a.similarity);
      setWordSimilarities(similarities.slice(0, 5)); // Top 5 similarités
    } catch (error) {
      console.error('Erreur calcul similarités:', error);
    }
  };

  // Fonction pour changer la taille de visualisation
  const toggleVisualizationSize = () => {
    setVisualizationSize(prev => prev === 'small' ? 'large' : 'small');
    // Redessiner le graphique après changement de taille
    setTimeout(() => {
      if ((window as any).Plotly) {
        (window as any).Plotly.Plots.resize('embedding-plot');
      }
    }, 100);
  };

  // Fonction de recherche
  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setError('Veuillez entrer une requête de recherche');
      return;
    }

    setIsSearching(true);
    setError(null);

    try {
      const reviews = await DatasetLoader.loadAmazonPolarityDataset(500);
      const texts = reviews.map(review => review.text);

      const results = await EmbeddingService.semanticSearch(searchQuery, texts, 10);
      setSearchResults(results);
      setSearchStats({
        query: searchQuery,
        total_searched: texts.length,
        results_found: results.length
      });

    } catch (error) {
      setError(error instanceof Error ? error.message : 'Erreur lors de la recherche');
    } finally {
      setIsSearching(false);
    }
  };

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
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
              <p className="text-slate-300 mb-6">
                Le service d'embedding n'est pas accessible. Assurez-vous que le backend est démarré.
              </p>
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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header fixe */}
      <div className="sticky top-0 z-50 bg-slate-900/95 backdrop-blur-xl border-b border-white/10 p-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Hub des Embeddings</h1>
            <p className="text-slate-300">Entraînement, Visualisation et Recherche sémantique</p>
          </div>
          
          {/* Navigation rapide */}
          <div className="flex items-center space-x-4">
            <button
              onClick={() => scrollToSection('training')}
              className="px-4 py-2 bg-blue-500/20 text-blue-300 rounded-lg hover:bg-blue-500/30 transition-colors flex items-center space-x-2"
            >
              <Brain className="h-4 w-4" />
              <span>Entraînement</span>
            </button>
            <button
              onClick={() => scrollToSection('visualization')}
              className="px-4 py-2 bg-purple-500/20 text-purple-300 rounded-lg hover:bg-purple-500/30 transition-colors flex items-center space-x-2"
            >
              <Target className="h-4 w-4" />
              <span>Visualisation</span>
            </button>
            <button
              onClick={() => scrollToSection('search')}
              className="px-4 py-2 bg-green-500/20 text-green-300 rounded-lg hover:bg-green-500/30 transition-colors flex items-center space-x-2"
            >
              <Search className="h-4 w-4" />
              <span>Recherche</span>
            </button>
            {onClose && (
              <button
                onClick={onClose}
                className="px-6 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-500 transition-colors"
              >
                Retour
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-8 space-y-16">
        {/* Guide d'utilisation */}
        <div className="bg-blue-500/20 border border-blue-500/30 rounded-2xl p-6">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
            <Lightbulb className="h-6 w-6 text-blue-400" />
            <span>Guide d'utilisation - Ordre recommandé</span>
          </h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-slate-700 p-4 rounded-xl">
              <p className="text-white font-bold">1️⃣ Entraînement TF-IDF</p>
              <p className="text-slate-300 text-sm">Entraîner d'abord le modèle sur le dataset Amazon</p>
            </div>
            <div className="bg-slate-700 p-4 rounded-xl">
              <p className="text-white font-bold">2️⃣ Visualisation</p>
              <p className="text-slate-300 text-sm">Voir les mots dans l'espace vectoriel 2D</p>
            </div>
            <div className="bg-slate-700 p-4 rounded-xl">
              <p className="text-white font-bold">3️⃣ Recherche sémantique</p>
              <p className="text-slate-300 text-sm">Trouver des reviews similaires</p>
            </div>
          </div>
        </div>

        {/* Section 1: Entraînement TF-IDF */}
        <section id="training" className="scroll-mt-24">
          <div className="bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-white/10">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center space-x-3">
              <Brain className="h-8 w-8 text-blue-400" />
              <span>Entraînement TF-IDF</span>
            </h2>

            <div className="grid lg:grid-cols-2 gap-8">
              {/* Configuration */}
              <div className="space-y-6">
                <div>
                  <label className="block text-white text-sm font-medium mb-3">
                    Source des données
                  </label>
                  <select
                    value={dataSource}
                    onChange={(e) => setDataSource(e.target.value as 'amazon' | 'custom')}
                    className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white focus:outline-none focus:border-blue-400"
                    style={{ color: 'white', backgroundColor: '#334155' }}
                  >
                    <option value="amazon" style={{ color: 'white', backgroundColor: '#334155' }}>
                      Amazon Dataset (1000 reviews)
                    </option>
                    <option value="custom" style={{ color: 'white', backgroundColor: '#334155' }}>
                      Textes personnalisés
                    </option>
                  </select>
                </div>

                {dataSource === 'custom' && (
                  <div>
                    <label className="block text-white text-sm font-medium mb-2">
                      Textes personnalisés (un par ligne)
                    </label>
                    <textarea
                      value={customTexts}
                      onChange={(e) => setCustomTexts(e.target.value)}
                      placeholder="Entrez vos textes, un par ligne..."
                      className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:border-blue-400 resize-none"
                      style={{ color: 'white', backgroundColor: '#334155' }}
                      rows={6}
                    />
                  </div>
                )}

                <button
                  onClick={handleTraining}
                  disabled={isTraining}
                  className="w-full px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  {isTraining ? (
                    <>
                      <RefreshCw className="h-5 w-5 animate-spin" />
                      <span>Entraînement...</span>
                    </>
                  ) : (
                    <>
                      <Brain className="h-5 w-5" />
                      <span>Entraîner le modèle TF-IDF</span>
                    </>
                  )}
                </button>
              </div>

              {/* Résultats */}
              <div className="space-y-6">
                {trainingSuccess && trainingStats && (
                  <div className="bg-green-500/20 border border-green-500/30 rounded-xl p-6">
                    <div className="flex items-center space-x-2 mb-4">
                      <CheckCircle className="h-6 w-6 text-green-400" />
                      <h3 className="text-white font-bold">Modèle entraîné avec succès !</h3>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-slate-700 p-3 rounded-lg text-center">
                        <p className="text-white font-bold text-xl">{trainingStats.vocabulary_size}</p>
                        <p className="text-slate-300 text-sm">Termes dans le vocabulaire</p>
                      </div>
                      <div className="bg-slate-700 p-3 rounded-lg text-center">
                        <p className="text-white font-bold text-xl">{trainingStats.corpus_size}</p>
                        <p className="text-slate-300 text-sm">Textes traités</p>
                      </div>
                    </div>
                  </div>
                )}

                {availableModels.length > 0 && (
                  <div className="bg-slate-700/50 p-4 rounded-xl">
                    <h4 className="text-white font-medium mb-2">Modèles disponibles :</h4>
                    {availableModels.map((model, index) => (
                      <div key={index} className="flex items-center justify-between text-sm">
                        <span className="text-slate-300">{model.name}</span>
                        <span className="text-green-400">✓ Prêt</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: Visualisation */}
        <section id="visualization" className="scroll-mt-24">
          <div className="bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-white/10">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
                <Target className="h-8 w-8 text-purple-400" />
                <span>Visualisation des Embeddings</span>
              </h2>
              
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
                  className={`px-4 py-2 rounded-lg transition-colors flex items-center space-x-2 ${
                    showAdvancedOptions 
                      ? 'bg-purple-500/30 text-purple-300' 
                      : 'bg-slate-600 text-slate-300 hover:bg-slate-500'
                  }`}
                >
                  <Settings className="h-4 w-4" />
                  <span>Options</span>
                </button>
                
                <button
                  onClick={toggleVisualizationSize}
                  className="px-4 py-2 bg-slate-600 text-slate-300 rounded-lg hover:bg-slate-500 transition-colors flex items-center space-x-2"
                >
                  <Maximize2 className="h-4 w-4" />
                  <span>{visualizationSize === 'large' ? 'Réduire' : 'Agrandir'}</span>
                </button>
              </div>
            </div>

                         <div className={`grid gap-6 ${visualizationSize === 'large' ? 'lg:grid-cols-4' : 'lg:grid-cols-3'}`}>
               {/* Configuration */}
               <div className="space-y-4">
                <div>
                  <label className="block text-white text-sm font-medium mb-2">
                    Méthode de réduction
                  </label>
                  <select
                    value={visualizationMethod}
                    onChange={(e) => setVisualizationMethod(e.target.value)}
                    className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white focus:outline-none focus:border-purple-400"
                    style={{ color: 'white', backgroundColor: '#334155' }}
                  >
                    <option value="tsne" style={{ color: 'white', backgroundColor: '#334155' }}>t-SNE (recommandé)</option>
                    <option value="pca" style={{ color: 'white', backgroundColor: '#334155' }}>PCA (rapide)</option>
                    <option value="umap" style={{ color: 'white', backgroundColor: '#334155' }}>UMAP (expérimental)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-white text-sm font-medium mb-2">
                    Mots à visualiser (séparés par des virgules)
                  </label>
                  <textarea
                    value={words}
                    onChange={(e) => setWords(e.target.value)}
                    placeholder="good,bad,excellent,terrible,amazing..."
                    className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:border-purple-400 resize-none"
                    style={{ color: 'white', backgroundColor: '#334155' }}
                    rows={4}
                  />
                                     <div className="flex justify-between items-center mt-2">
                     <p className="text-slate-400 text-xs">
                       {words.split(',').filter(w => w.trim()).length} mots sélectionnés
                     </p>
                     <button
                       onClick={() => setWords('')}
                       className="text-slate-400 hover:text-white text-xs underline"
                     >
                       Effacer
                     </button>
                   </div>
                 </div>

                {/* Options avancées */}
                {showAdvancedOptions && (
                  <div className="space-y-4 bg-slate-700/30 p-4 rounded-xl border border-purple-500/20">
                    <h4 className="text-white font-medium flex items-center space-x-2">
                      <Palette className="h-4 w-4" />
                      <span>Options avancées</span>
                    </h4>
                    
                    <div>
                      <label className="block text-white text-sm mb-2">Palette de couleurs</label>
                      <select
                        value={colorScheme}
                        onChange={(e) => setColorScheme(e.target.value)}
                        className="w-full p-2 bg-slate-600 border border-slate-500 rounded-lg text-white text-sm"
                        style={{ color: 'white', backgroundColor: '#475569' }}
                      >
                        <option value="viridis">Viridis</option>
                        <option value="plasma">Plasma</option>
                        <option value="rainbow">Arc-en-ciel</option>
                        <option value="blues">Bleus</option>
                        <option value="reds">Rouges</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-white text-sm mb-2">
                        Taille des points: {pointSize}px
                      </label>
                      <input
                        type="range"
                        min="4"
                        max="16"
                        value={pointSize}
                        onChange={(e) => setPointSize(Number(e.target.value))}
                        className="w-full"
                      />
                    </div>

                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="showLabels"
                        checked={showLabels}
                        onChange={(e) => setShowLabels(e.target.checked)}
                        className="rounded"
                      />
                      <label htmlFor="showLabels" className="text-white text-sm">
                        Afficher les étiquettes
                      </label>
                    </div>
                  </div>
                )}

                                 <button
                   onClick={handleVisualization}
                   disabled={isVisualizing || availableModels.length === 0 || !words.trim()}
                   className="w-full px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                 >
                   {isVisualizing ? (
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

                 {/* Instructions d'utilisation */}
                 {availableModels.length === 0 && (
                   <div className="bg-yellow-500/20 border border-yellow-500/30 rounded-xl p-3">
                     <div className="flex items-center space-x-2 mb-2">
                       <AlertCircle className="h-4 w-4 text-yellow-400" />
                       <span className="text-yellow-300 font-medium text-sm">Modèle requis</span>
                     </div>
                     <p className="text-yellow-200 text-xs">
                       Entraînez d'abord un modèle TF-IDF dans la section ci-dessus.
                     </p>
                   </div>
                 )}

                 {availableModels.length > 0 && !words.trim() && (
                   <div className="bg-blue-500/20 border border-blue-500/30 rounded-xl p-3">
                     <div className="flex items-center space-x-2 mb-2">
                       <Lightbulb className="h-4 w-4 text-blue-400" />
                       <span className="text-blue-300 font-medium text-sm">Comment commencer</span>
                     </div>
                     <p className="text-blue-200 text-xs">
                       1. Entrez des mots séparés par des virgules<br/>
                       2. Ou cliquez sur un exemple ci-dessous<br/>
                       3. Cliquez sur "Visualiser"
                     </p>
                   </div>
                 )}

                                 {/* Exemples améliorés */}
                 <div className="bg-cyan-500/20 border border-cyan-500/30 rounded-xl p-4">
                   <h4 className="text-white font-bold mb-3 flex items-center space-x-2">
                     <Lightbulb className="h-4 w-4" />
                     <span>Exemples à essayer :</span>
                   </h4>
                   <div className="space-y-3 text-sm">
                     <div className="bg-slate-700 p-3 rounded-lg cursor-pointer hover:bg-slate-600 transition-colors" 
                          onClick={() => setWords('happy,sad,joy,anger,love,hate')}>
                       <div className="flex items-center space-x-2 mb-1">
                         <span className="text-xl">😊</span>
                         <strong className="text-cyan-300">Sentiments</strong>
                       </div>
                       <div className="text-slate-300 text-xs leading-relaxed">
                         happy, sad, joy, anger, love, hate
                       </div>
                     </div>
                     
                     <div className="bg-slate-700 p-3 rounded-lg cursor-pointer hover:bg-slate-600 transition-colors"
                          onClick={() => setWords('good,bad,excellent,terrible,amazing,awful')}>
                       <div className="flex items-center space-x-2 mb-1">
                         <span className="text-xl">⭐</span>
                         <strong className="text-green-300">Qualité</strong>
                       </div>
                       <div className="text-slate-300 text-xs leading-relaxed">
                         good, bad, excellent, terrible, amazing, awful
                       </div>
                     </div>
                     
                     <div className="bg-slate-700 p-3 rounded-lg cursor-pointer hover:bg-slate-600 transition-colors"
                          onClick={() => setWords('fast,slow,quick,rapid,delayed,speedy')}>
                       <div className="flex items-center space-x-2 mb-1">
                         <span className="text-xl">⚡</span>
                         <strong className="text-yellow-300">Vitesse</strong>
                       </div>
                       <div className="text-slate-300 text-xs leading-relaxed">
                         fast, slow, quick, rapid, delayed, speedy
                       </div>
                     </div>
                     
                     <div className="bg-slate-700 p-3 rounded-lg cursor-pointer hover:bg-slate-600 transition-colors"
                          onClick={() => setWords('big,small,huge,tiny,massive,giant')}>
                       <div className="flex items-center space-x-2 mb-1">
                         <span className="text-xl">📏</span>
                         <strong className="text-purple-300">Taille</strong>
                       </div>
                       <div className="text-slate-300 text-xs leading-relaxed">
                         big, small, huge, tiny, massive, giant
                       </div>
                     </div>
                   </div>
                 </div>
              </div>

              {/* Visualisation */}
              <div className={visualizationSize === 'large' ? 'lg:col-span-2' : 'lg:col-span-1'}>
                <div className={`bg-slate-700/50 p-6 rounded-xl ${visualizationSize === 'large' ? 'h-[500px]' : 'h-96'}`}>
                  <div ref={plotRef} id="embedding-plot" className="w-full h-full"></div>
                  
                  {!visualization && !isVisualizing && (
                    <div className="flex flex-col items-center justify-center h-full text-center">
                      <Layers className="h-16 w-16 text-slate-500 mb-4" />
                      <p className="text-slate-300 text-lg">La visualisation apparaîtra ici</p>
                      <p className="text-slate-400 text-sm mt-2">Entrez des mots et cliquez sur "Visualiser"</p>
                    </div>
                  )}
                </div>

                {/* Statistiques de visualisation améliorées */}
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
                      <p className="text-white font-bold text-lg">{visualization.words_not_found?.length || 0}</p>
                      <p className="text-red-200 text-xs">Mots manqués</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Analyses avancées */}
              {visualizationSize === 'large' && (visualization || clusterAnalysis || wordSimilarities.length > 0) && (
                <div className="space-y-6">
                  {/* Analyse des clusters */}
                  {clusterAnalysis && (
                    <div className="bg-slate-700/30 p-4 rounded-xl border border-indigo-500/30">
                      <h4 className="text-white font-bold mb-3 flex items-center space-x-2">
                        <Grid3X3 className="h-4 w-4 text-indigo-400" />
                        <span>Analyse des Clusters</span>
                      </h4>
                      <div className="space-y-3 text-sm">
                        <div>
                          <p className="text-indigo-300 font-medium">Mots centraux :</p>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {clusterAnalysis.center_words.map((word: string, i: number) => (
                              <span key={i} className="px-2 py-1 bg-indigo-500/20 text-indigo-200 rounded text-xs">
                                {word}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div>
                          <p className="text-orange-300 font-medium">Mots périphériques :</p>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {clusterAnalysis.outlier_words.map((word: string, i: number) => (
                              <span key={i} className="px-2 py-1 bg-orange-500/20 text-orange-200 rounded text-xs">
                                {word}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div className="bg-slate-600 p-2 rounded">
                          <p className="text-slate-300">
                            <strong>Dispersion :</strong> {clusterAnalysis.total_spread.toFixed(2)}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Similarités entre mots */}
                  {wordSimilarities.length > 0 && (
                    <div className="bg-slate-700/30 p-4 rounded-xl border border-yellow-500/30">
                      <h4 className="text-white font-bold mb-3 flex items-center space-x-2">
                        <TrendingUp className="h-4 w-4 text-yellow-400" />
                        <span>Top Similarités</span>
                      </h4>
                      <div className="space-y-2">
                        {wordSimilarities.map((sim, i) => (
                          <div key={i} className="flex items-center justify-between bg-slate-600 p-2 rounded text-sm">
                            <span className="text-white">
                              <strong>{sim.word1}</strong> ↔ <strong>{sim.word2}</strong>
                            </span>
                            <div className="flex items-center space-x-2">
                              <span className="text-yellow-400 font-bold">
                                {(sim.similarity * 100).toFixed(1)}%
                              </span>
                              <div className="w-12 h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div 
                                  className="h-full bg-gradient-to-r from-yellow-500 to-orange-500"
                                  style={{ width: `${sim.similarity * 100}%` }}
                                ></div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Mots non trouvés */}
                  {visualization?.words_not_found && visualization.words_not_found.length > 0 && (
                    <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-4">
                      <h4 className="text-white font-bold mb-2 flex items-center space-x-2">
                        <AlertCircle className="h-4 w-4 text-red-400" />
                        <span>Mots non trouvés</span>
                      </h4>
                      <div className="flex flex-wrap gap-1">
                        {visualization.words_not_found.map((word: string, i: number) => (
                          <span key={i} className="px-2 py-1 bg-red-500/20 text-red-200 rounded text-xs">
                            {word}
                          </span>
                        ))}
                      </div>
                      <p className="text-red-300 text-xs mt-2">
                        Ces mots n'ont pas été trouvés dans le vocabulaire du modèle.
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Section 3: Recherche sémantique */}
        <section id="search" className="scroll-mt-24">
          <div className="bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-white/10">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center space-x-3">
              <Search className="h-8 w-8 text-green-400" />
              <span>Recherche Sémantique</span>
            </h2>

            {/* Explication des pourcentages */}
            <div className="bg-blue-500/20 border border-blue-500/30 rounded-xl p-4 mb-6">
              <h3 className="text-white font-bold mb-2 flex items-center space-x-2">
                <Lightbulb className="h-5 w-5 text-blue-400" />
                <span>Comment interpréter les pourcentages de similarité ?</span>
              </h3>
              <div className="grid md:grid-cols-4 gap-3 text-sm">
                <div className="bg-green-500/20 p-3 rounded-lg text-center border border-green-500/30">
                  <div className="text-green-400 font-bold text-lg">80-100%</div>
                  <div className="text-green-300 text-xs">Très similaire</div>
                  <div className="text-green-200 text-xs mt-1">Même sujet/sentiment</div>
                </div>
                <div className="bg-yellow-500/20 p-3 rounded-lg text-center border border-yellow-500/30">
                  <div className="text-yellow-400 font-bold text-lg">50-80%</div>
                  <div className="text-yellow-300 text-xs">Assez similaire</div>
                  <div className="text-yellow-200 text-xs mt-1">Contexte proche</div>
                </div>
                <div className="bg-orange-500/20 p-3 rounded-lg text-center border border-orange-500/30">
                  <div className="text-orange-400 font-bold text-lg">20-50%</div>
                  <div className="text-orange-300 text-xs">Peu similaire</div>
                  <div className="text-orange-200 text-xs mt-1">Quelques mots communs</div>
                </div>
                <div className="bg-red-500/20 p-3 rounded-lg text-center border border-red-500/30">
                  <div className="text-red-400 font-bold text-lg">0-20%</div>
                  <div className="text-red-300 text-xs">Très différent</div>
                  <div className="text-red-200 text-xs mt-1">Pas de relation</div>
                </div>
              </div>
            </div>

            <div className="grid lg:grid-cols-3 gap-8">
              {/* Configuration de recherche */}
              <div className="space-y-6">
                <div>
                  <label className="block text-white text-sm font-medium mb-2">
                    Requête de recherche
                  </label>
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Ex: great product, battery life..."
                    className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:border-green-400"
                    style={{ color: 'white', backgroundColor: '#334155' }}
                    onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  />
                  <p className="text-slate-400 text-xs mt-1">
                    Tapez des mots-clés pour trouver des reviews similaires
                  </p>
                </div>

                {/* Exemples de recherche */}
                <div className="bg-cyan-500/20 border border-cyan-500/30 rounded-xl p-4">
                  <h4 className="text-white font-bold mb-3 flex items-center space-x-2">
                    <Lightbulb className="h-4 w-4" />
                    <span>Exemples de recherche :</span>
                  </h4>
                  <div className="space-y-2 text-sm">
                    <div className="bg-slate-700 p-2 rounded cursor-pointer hover:bg-slate-600 transition-colors" 
                         onClick={() => setSearchQuery('great product quality')}>
                      <strong className="text-green-300">👍 Positif :</strong> great product quality
                    </div>
                    <div className="bg-slate-700 p-2 rounded cursor-pointer hover:bg-slate-600 transition-colors"
                         onClick={() => setSearchQuery('terrible experience disappointed')}>
                      <strong className="text-red-300">👎 Négatif :</strong> terrible experience disappointed
                    </div>
                    <div className="bg-slate-700 p-2 rounded cursor-pointer hover:bg-slate-600 transition-colors"
                         onClick={() => setSearchQuery('battery life duration')}>
                      <strong className="text-blue-300">🔋 Technique :</strong> battery life duration
                    </div>
                    <div className="bg-slate-700 p-2 rounded cursor-pointer hover:bg-slate-600 transition-colors"
                         onClick={() => setSearchQuery('fast shipping delivery')}>
                      <strong className="text-purple-300">📦 Livraison :</strong> fast shipping delivery
                    </div>
                  </div>
                </div>

                <button
                  onClick={handleSearch}
                  disabled={isSearching || availableModels.length === 0 || !searchQuery.trim()}
                  className="w-full px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  {isSearching ? (
                    <>
                      <RefreshCw className="h-5 w-5 animate-spin" />
                      <span>Recherche...</span>
                    </>
                  ) : (
                    <>
                      <Search className="h-5 w-5" />
                      <span>Rechercher</span>
                    </>
                  )}
                </button>

                {/* Instructions conditionnelles */}
                {availableModels.length === 0 && (
                  <div className="bg-yellow-500/20 border border-yellow-500/30 rounded-xl p-3">
                    <div className="flex items-center space-x-2 mb-2">
                      <AlertCircle className="h-4 w-4 text-yellow-400" />
                      <span className="text-yellow-300 font-medium text-sm">Modèle requis</span>
                    </div>
                    <p className="text-yellow-200 text-xs">
                      Entraînez d'abord un modèle TF-IDF pour activer la recherche.
                    </p>
                  </div>
                )}

                {searchStats && (
                  <div className="bg-slate-700/50 p-4 rounded-xl">
                    <div className="flex items-center space-x-2 mb-2">
                      <CheckCircle className="h-5 w-5 text-green-400" />
                      <span className="text-white font-medium">Recherche terminée</span>
                    </div>
                    <div className="text-sm space-y-1">
                      <p className="text-slate-300">
                        <strong className="text-white">{searchStats.results_found}</strong> résultats trouvés
                      </p>
                      <p className="text-slate-300">
                        sur <strong className="text-white">{searchStats.total_searched}</strong> reviews analysées
                      </p>
                      <div className="mt-2 p-2 bg-slate-600 rounded text-xs">
                        <strong className="text-white">Requête :</strong> "{searchStats.query}"
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Résultats de recherche */}
              <div className="lg:col-span-2">
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {searchResults.length === 0 && !isSearching && (
                    <div className="flex flex-col items-center justify-center h-64 text-center">
                      <Search className="h-16 w-16 text-slate-500 mb-4" />
                      <p className="text-slate-300 text-lg">Aucun résultat</p>
                      <p className="text-slate-400 text-sm mt-2">Entrez une requête et cliquez sur "Rechercher"</p>
                    </div>
                  )}

                  {searchResults.map((result, index) => {
                    const similarityPercent = result.similarity * 100;
                    let similarityColor = 'text-red-300';
                    let similarityBg = 'bg-slate-700/80';
                    let similarityLabel = 'Très différent';
                    let gradientColor = 'from-red-400 to-red-500';
                    let borderColor = 'border-red-400/50';

                    if (similarityPercent >= 80) {
                      similarityColor = 'text-green-300';
                      similarityLabel = 'Très similaire';
                      gradientColor = 'from-green-400 to-green-500';
                      borderColor = 'border-green-400/50';
                    } else if (similarityPercent >= 50) {
                      similarityColor = 'text-yellow-300';
                      similarityLabel = 'Assez similaire';
                      gradientColor = 'from-yellow-400 to-yellow-500';
                      borderColor = 'border-yellow-400/50';
                    } else if (similarityPercent >= 20) {
                      similarityColor = 'text-orange-300';
                      similarityLabel = 'Peu similaire';
                      gradientColor = 'from-orange-400 to-orange-500';
                      borderColor = 'border-orange-400/50';
                    }

                    return (
                      <div key={index} className={`${similarityBg} p-4 rounded-xl border ${borderColor}`}>
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center space-x-3">
                            <span className="text-white text-sm font-medium bg-slate-600 px-2 py-1 rounded">
                              #{index + 1}
                            </span>
                            <span className={`text-xs px-3 py-1 rounded-full bg-slate-600 ${similarityColor} font-medium`}>
                              {similarityLabel}
                            </span>
                          </div>
                          <div className="flex items-center space-x-3">
                            <span className={`text-lg font-bold ${similarityColor}`}>
                              {similarityPercent.toFixed(1)}%
                            </span>
                            <div className="w-24 h-4 bg-slate-600 rounded-full overflow-hidden">
                              <div 
                                className={`h-full bg-gradient-to-r ${gradientColor} transition-all duration-500`}
                                style={{ width: `${similarityPercent}%` }}
                              ></div>
                            </div>
                          </div>
                        </div>
                        
                        <div className="bg-slate-600/50 p-3 rounded-lg mb-3">
                          <p className="text-white text-sm leading-relaxed">
                            {result.text_preview || result.text}
                          </p>
                        </div>
                        
                                                 <div className="bg-slate-800/80 p-3 rounded-lg border border-slate-600">
                           <p className="text-slate-100 text-xs leading-relaxed">
                             <strong className="text-white">💡 Interprétation :</strong> Cette review partage{' '}
                             <span className={similarityColor}>
                               {similarityPercent >= 80 ? 'beaucoup de mots et concepts' :
                                similarityPercent >= 50 ? 'plusieurs mots et le contexte' :
                                similarityPercent >= 20 ? 'quelques mots communs' :
                                'très peu de mots communs'}
                             </span>
                             {' '}avec votre recherche "{searchQuery}".
                           </p>
                           <div className="mt-2 flex items-center justify-between">
                             <span className="text-slate-300 text-xs">
                               {result.word_count || 'N/A'} mots • {result.char_count || 'N/A'} caractères
                             </span>
                             <span className={`text-xs px-2 py-1 rounded ${
                               similarityPercent >= 80 ? 'bg-green-500/20 text-green-300' :
                               similarityPercent >= 50 ? 'bg-yellow-500/20 text-yellow-300' :
                               similarityPercent >= 20 ? 'bg-orange-500/20 text-orange-300' :
                               'bg-red-500/20 text-red-300'
                             }`}>
                               Rang #{index + 1}
                             </span>
                           </div>
                         </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Messages d'erreur globaux */}
        {error && (
          <div className="fixed bottom-6 right-6 bg-red-500/20 border border-red-500/30 rounded-xl p-4 max-w-md">
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-5 w-5 text-red-400" />
              <span className="text-red-400 font-medium">Erreur</span>
            </div>
            <p className="text-red-300 mt-1 text-sm">{error}</p>
            <button
              onClick={() => setError(null)}
              className="mt-2 text-red-400 hover:text-red-300 text-sm underline"
            >
              Fermer
            </button>
          </div>
        )}
      </div>

      {/* Script Plotly */}
      <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </div>
  );
}; 