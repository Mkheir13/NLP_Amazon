import React, { useState, useEffect } from 'react';
import { Target, Settings, RefreshCw, CheckCircle, AlertCircle, BarChart3, Download, Zap, Brain, Database } from 'lucide-react';
import { EmbeddingService, EmbeddingModel } from '../services/EmbeddingService';
import { DatasetLoader, Review } from '../services/DatasetLoader';

interface EmbeddingTrainingProps {
  onClose?: () => void;
}

export const EmbeddingTraining: React.FC<EmbeddingTrainingProps> = ({ onClose }) => {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingData, setTrainingData] = useState<string>('amazon');
  const [customTexts, setCustomTexts] = useState<string>('');
  const [config, setConfig] = useState({
    vector_size: 100,
    window: 5,
    min_count: 2,
    workers: 4,
    epochs: 10,
    sg: 1 // Skip-gram
  });
  const [trainedModel, setTrainedModel] = useState<EmbeddingModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [isLoadingDataset, setIsLoadingDataset] = useState(false);
  const [serviceAvailable, setServiceAvailable] = useState<boolean>(false);
  const [existingModels, setExistingModels] = useState<EmbeddingModel[]>([]);

  const [datasetSize, setDatasetSize] = useState<number>(5000);
  const [maxDatasetSize, setMaxDatasetSize] = useState<number>(3600000); // 3.6M échantillons disponibles
  const [useRandomSample, setUseRandomSample] = useState<boolean>(true);

  // Vérifier la disponibilité du service
  useEffect(() => {
    const checkService = async () => {
      const available = await EmbeddingService.isEmbeddingServiceAvailable();
      setServiceAvailable(available);
      
      if (available) {
        // Charger les modèles existants
        try {
          const models = await EmbeddingService.getEmbeddingModels();
          setExistingModels(models);
        } catch (error) {
          console.error('Erreur chargement modèles:', error);
        }
      }
    };
    checkService();
  }, []);

  // Charger le dataset Amazon si sélectionné
  useEffect(() => {
    if (serviceAvailable && trainingData === 'amazon') {
      loadAmazonDataset();
    }
  }, [serviceAvailable, trainingData]);

  const loadAmazonDataset = async () => {
    setIsLoadingDataset(true);
    try {
      const loadedReviews = await DatasetLoader.loadAmazonPolarityDataset(2000);
      setReviews(loadedReviews);
    } catch (error) {
      console.error('Erreur chargement dataset:', error);
    } finally {
      setIsLoadingDataset(false);
    }
  };

  const startTraining = async () => {
    setIsTraining(true);
    setError(null);
    setTrainedModel(null);

    try {
      let texts: string[] = [];

      if (trainingData === 'amazon') {
        // Charger le dataset avec la taille spécifiée
        console.log(`🚀 Chargement de ${datasetSize} avis Amazon...`);
        const loadedReviews = await DatasetLoader.loadAmazonPolarityDataset(datasetSize, useRandomSample);
        texts = loadedReviews.map(review => review.text);
        console.log(`✅ ${texts.length} avis chargés pour l'entraînement`);
      } else if (trainingData === 'custom') {
        if (!customTexts.trim()) {
          throw new Error('Veuillez entrer des textes personnalisés');
        }
        texts = customTexts.split('\n').filter(text => text.trim().length > 0);
      }

      if (texts.length === 0) {
        throw new Error('Aucun texte disponible pour l\'entraînement');
      }

      console.log(`🚀 Démarrage entraînement Word2Vec sur ${texts.length} textes...`);
      
      const model = await EmbeddingService.trainWord2Vec(texts, {
        ...config,
        dataset_size: datasetSize,
        random_sample: useRandomSample
      });
      setTrainedModel(model);
      
      // Recharger la liste des modèles
      const models = await EmbeddingService.getEmbeddingModels();
      setExistingModels(models);

    } catch (error) {
      setError(error instanceof Error ? error.message : 'Erreur lors de l\'entraînement');
    } finally {
      setIsTraining(false);
    }
  };

  const getConfigDescription = () => {
    return {
      vector_size: `Taille des vecteurs d'embedding (${config.vector_size} dimensions)`,
      window: `Taille de la fenêtre contextuelle (${config.window} mots)`,
      min_count: `Fréquence minimale des mots (${config.min_count} occurrences)`,
      workers: `Nombre de threads parallèles (${config.workers})`,
      epochs: `Nombre d'époques d'entraînement (${config.epochs})`,
      sg: config.sg === 1 ? 'Skip-gram (recommandé)' : 'CBOW (plus rapide)'
    };
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
                Le service d'embedding n'est pas accessible. Assurez-vous que le backend est démarré.
              </p>
              {onClose && (
                <button
                  onClick={onClose}
                  className="px-6 py-3 bg-slate-600 text-white rounded-xl hover:bg-slate-500 transition-colors"
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
            <h1 className="text-4xl font-bold text-white mb-2">Entraînement d'Embeddings</h1>
            <p className="text-white/70 text-lg">Créez vos propres modèles Word2Vec personnalisés</p>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="px-6 py-3 bg-slate-600 text-white rounded-xl hover:bg-slate-500 transition-colors"
            >
              Retour
            </button>
          )}
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Configuration d'entraînement */}
          <div className="lg:col-span-2 space-y-6">
            {/* Source de données */}
            <div className="bg-slate-800/90 backdrop-blur-xl p-6 rounded-2xl border border-white/10">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
                <Database className="h-6 w-6 text-cyan-400" />
                <span>Source de Données</span>
              </h3>

              <div className="mb-4">
                <label className="block text-white/80 text-sm font-medium mb-2">
                  Données d'entraînement
                </label>
                <select
                  value={trainingData}
                  onChange={(e) => setTrainingData(e.target.value)}
                  className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-cyan-400"
                >
                  <option value="amazon">Dataset Amazon (jusqu'à {maxDatasetSize.toLocaleString()} avis)</option>
                  <option value="custom">Textes personnalisés</option>
                </select>
              </div>

              {/* Configuration du Dataset Amazon */}
              {trainingData === 'amazon' && (
                <div className="mb-4 p-4 bg-slate-700/30 rounded-xl">
                  <h4 className="text-white/90 font-medium mb-3 flex items-center">
                    <Settings className="h-4 w-4 mr-2" />
                    Configuration du Dataset
                  </h4>
                  
                  <div className="space-y-4">
                    {/* Sélection de la taille */}
                    <div>
                      <label className="block text-white/80 text-sm font-medium mb-2">
                        Nombre d'avis: {datasetSize.toLocaleString()}
                      </label>
                      <input
                        type="range"
                        min="50"
                        max={maxDatasetSize}
                        step="50"
                        value={datasetSize}
                        onChange={(e) => setDatasetSize(parseInt(e.target.value))}
                        className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-white/50 mt-1">
                        <span>50</span>
                        <span>1K</span>
                        <span>3K</span>
                        <span>{maxDatasetSize.toLocaleString()}</span>
                      </div>
                    </div>

                    {/* Boutons de sélection rapide */}
                    <div className="flex flex-wrap gap-2">
                      {[100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, maxDatasetSize].map(size => (
                        <button
                          key={size}
                          onClick={() => setDatasetSize(size)}
                          className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                            datasetSize === size
                              ? 'bg-cyan-500 text-white'
                              : 'bg-slate-600 text-white/70 hover:bg-slate-500'
                          }`}
                        >
                          {size >= 1000000 ? `${(size/1000000).toFixed(1)}M` : size >= 1000 ? `${(size/1000)}K` : size.toString()}
                        </button>
                      ))}
                    </div>

                    {/* Options d'échantillonnage */}
                    <div>
                      <label className="flex items-center text-white/80 text-sm">
                        <input
                          type="checkbox"
                          checked={useRandomSample}
                          onChange={(e) => setUseRandomSample(e.target.checked)}
                          className="rounded border-slate-400 text-cyan-500 focus:ring-cyan-500 mr-2"
                        />
                        Échantillonnage aléatoire
                      </label>
                      <p className="text-white/50 text-xs mt-1">
                        Sélectionne aléatoirement les avis pour éviter le biais
                      </p>
                    </div>

                    {/* Conseils */}
                    <div className="text-xs text-white/60 bg-slate-600/30 p-3 rounded-lg">
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
              )}

              {trainingData === 'amazon' && (
                <div className="p-4 bg-slate-700/50 rounded-xl">
                  <div className="flex items-center space-x-2 mb-2">
                    {isLoadingDataset ? (
                      <>
                        <RefreshCw className="h-4 w-4 text-cyan-400 animate-spin" />
                        <span className="text-cyan-400 text-sm">Chargement du dataset...</span>
                      </>
                    ) : reviews.length > 0 ? (
                      <>
                        <CheckCircle className="h-4 w-4 text-green-400" />
                        <span className="text-green-400 text-sm">{reviews.length} avis Amazon chargés</span>
                      </>
                    ) : (
                      <>
                        <AlertCircle className="h-4 w-4 text-orange-400" />
                        <span className="text-orange-400 text-sm">Dataset non chargé</span>
                      </>
                    )}
                  </div>
                  <p className="text-white/60 text-xs">
                    Avis clients Amazon avec sentiments positifs et négatifs
                  </p>
                </div>
              )}

              {trainingData === 'custom' && (
                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    Textes personnalisés (un par ligne)
                  </label>
                  <textarea
                    value={customTexts}
                    onChange={(e) => setCustomTexts(e.target.value)}
                    placeholder="Entrez vos textes ici, un par ligne...&#10;Exemple:&#10;Ce produit est fantastique&#10;Je recommande vivement&#10;Qualité décevante"
                    className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/50 focus:outline-none focus:border-cyan-400 resize-none"
                    rows={8}
                  />
                  <p className="text-white/50 text-xs mt-1">
                    {customTexts.split('\n').filter(line => line.trim()).length} textes
                  </p>
                </div>
              )}
            </div>

            {/* Configuration du modèle */}
            <div className="bg-slate-800/90 backdrop-blur-xl p-6 rounded-2xl border border-white/10">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
                <Settings className="h-6 w-6 text-purple-400" />
                <span>Configuration du Modèle</span>
              </h3>

              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    Taille des vecteurs
                  </label>
                  <select
                    value={config.vector_size}
                    onChange={(e) => setConfig({...config, vector_size: parseInt(e.target.value)})}
                    className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-purple-400"
                  >
                    <option value={50}>50 dimensions</option>
                    <option value={100}>100 dimensions</option>
                    <option value={200}>200 dimensions</option>
                    <option value={300}>300 dimensions</option>
                  </select>
                </div>

                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    Fenêtre contextuelle
                  </label>
                  <select
                    value={config.window}
                    onChange={(e) => setConfig({...config, window: parseInt(e.target.value)})}
                    className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-purple-400"
                  >
                    <option value={3}>3 mots</option>
                    <option value={5}>5 mots</option>
                    <option value={7}>7 mots</option>
                    <option value={10}>10 mots</option>
                  </select>
                </div>

                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    Fréquence minimale
                  </label>
                  <select
                    value={config.min_count}
                    onChange={(e) => setConfig({...config, min_count: parseInt(e.target.value)})}
                    className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-purple-400"
                  >
                    <option value={1}>1 occurrence</option>
                    <option value={2}>2 occurrences</option>
                    <option value={5}>5 occurrences</option>
                    <option value={10}>10 occurrences</option>
                  </select>
                </div>

                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    Époques d'entraînement
                  </label>
                  <select
                    value={config.epochs}
                    onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
                    className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-purple-400"
                  >
                    <option value={5}>5 époques</option>
                    <option value={10}>10 époques</option>
                    <option value={15}>15 époques</option>
                    <option value={20}>20 époques</option>
                  </select>
                </div>

                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    Algorithme
                  </label>
                  <select
                    value={config.sg}
                    onChange={(e) => setConfig({...config, sg: parseInt(e.target.value)})}
                    className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-purple-400"
                  >
                    <option value={1}>Skip-gram (recommandé)</option>
                    <option value={0}>CBOW (plus rapide)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-white/80 text-sm font-medium mb-2">
                    Threads parallèles
                  </label>
                  <select
                    value={config.workers}
                    onChange={(e) => setConfig({...config, workers: parseInt(e.target.value)})}
                    className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-purple-400"
                  >
                    <option value={1}>1 thread</option>
                    <option value={2}>2 threads</option>
                    <option value={4}>4 threads</option>
                    <option value={8}>8 threads</option>
                  </select>
                </div>
              </div>

              {/* Résumé de la configuration */}
              <div className="mt-6 p-4 bg-slate-700/30 rounded-xl">
                <h4 className="text-white/80 font-medium mb-3">📋 Résumé de la configuration :</h4>
                <div className="space-y-1">
                  {Object.entries(getConfigDescription()).map(([key, description]) => (
                    <div key={key} className="text-white/60 text-sm">
                      • {description}
                    </div>
                  ))}
                </div>
              </div>

              {/* Bouton d'entraînement */}
              <button
                onClick={startTraining}
                disabled={isTraining || isLoadingDataset || (trainingData === 'custom' && !customTexts.trim())}
                className="w-full mt-6 px-6 py-4 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
              >
                {isTraining ? (
                  <>
                    <RefreshCw className="h-5 w-5 animate-spin" />
                    <span>Entraînement en cours...</span>
                  </>
                ) : (
                  <>
                    <Brain className="h-5 w-5" />
                    <span>Démarrer l'Entraînement</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Panneau de résultats */}
          <div className="lg:col-span-1 space-y-6">
            {/* Résultat de l'entraînement */}
            <div className="bg-slate-800/90 backdrop-blur-xl p-6 rounded-2xl border border-white/10">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
                <Target className="h-6 w-6 text-green-400" />
                <span>Résultat</span>
              </h3>

              {error && (
                <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-4 mb-4">
                  <div className="flex items-center space-x-2">
                    <AlertCircle className="h-5 w-5 text-red-400" />
                    <span className="text-red-400 font-medium">Erreur</span>
                  </div>
                  <p className="text-red-300 mt-1 text-sm">{error}</p>
                </div>
              )}

              {trainedModel && (
                <div className="bg-green-500/20 border border-green-500/30 rounded-xl p-4 mb-4">
                  <div className="flex items-center space-x-2 mb-3">
                    <CheckCircle className="h-5 w-5 text-green-400" />
                    <span className="text-green-400 font-medium">Modèle entraîné avec succès !</span>
                  </div>
                  
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-white/70">ID :</span>
                      <span className="text-white font-mono">{trainedModel.id}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white/70">Vocabulaire :</span>
                      <span className="text-white">{trainedModel?.vocabulary_size?.toLocaleString() || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white/70">Textes :</span>
                      <span className="text-white">{trainedModel?.trained_on?.toLocaleString() || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white/70">Dimensions :</span>
                      <span className="text-white">{trainedModel?.config?.vector_size || 'N/A'}</span>
                    </div>
                  </div>
                </div>
              )}

              {!trainedModel && !error && !isTraining && (
                <div className="text-center py-8">
                  <Brain className="h-12 w-12 text-white/30 mx-auto mb-3" />
                  <p className="text-white/60">Aucun modèle entraîné</p>
                  <p className="text-white/40 text-sm mt-1">
                    Configurez et lancez l'entraînement
                  </p>
                </div>
              )}

              {isTraining && (
                <div className="text-center py-8">
                  <RefreshCw className="h-12 w-12 text-purple-400 animate-spin mx-auto mb-3" />
                  <p className="text-white/70">Entraînement en cours...</p>
                  <p className="text-white/50 text-sm mt-1">
                    Cela peut prendre plusieurs minutes
                  </p>
                </div>
              )}
            </div>

            {/* Modèles existants */}
            <div className="bg-slate-800/90 backdrop-blur-xl p-6 rounded-2xl border border-white/10">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
                <BarChart3 className="h-6 w-6 text-orange-400" />
                <span>Modèles Existants</span>
              </h3>

              {!existingModels || existingModels.length === 0 ? (
                <div className="text-center py-6">
                  <Database className="h-8 w-8 text-white/30 mx-auto mb-2" />
                  <p className="text-white/60 text-sm">Aucun modèle disponible</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {existingModels.slice(0, 5).filter(model => model).map((model, index) => (
                    <div key={index} className="bg-slate-700/50 p-3 rounded-xl">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-white font-medium text-sm">{model?.type || 'N/A'}</span>
                        <span className="text-white/60 text-xs">{model?.id || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-white/50">{model?.vocabulary_size || 0} mots</span>
                        <span className="text-white/50">{model?.config?.vector_size || 0}D</span>
                      </div>
                    </div>
                  ))}
                  {existingModels.length > 5 && (
                    <p className="text-white/50 text-xs text-center">
                      +{existingModels.length - 5} autres modèles
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Guide d'utilisation */}
            <div className="bg-slate-800/90 backdrop-blur-xl p-6 rounded-2xl border border-white/10">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
                <Zap className="h-6 w-6 text-yellow-400" />
                <span>Guide Rapide</span>
              </h3>

              <div className="space-y-3 text-sm">
                <div className="flex items-start space-x-2">
                  <div className="w-6 h-6 bg-cyan-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-cyan-400 text-xs font-bold">1</span>
                  </div>
                  <p className="text-white/70">Choisissez vos données d'entraînement</p>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="w-6 h-6 bg-purple-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-purple-400 text-xs font-bold">2</span>
                  </div>
                  <p className="text-white/70">Ajustez la configuration du modèle</p>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="w-6 h-6 bg-green-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-green-400 text-xs font-bold">3</span>
                  </div>
                  <p className="text-white/70">Lancez l'entraînement et attendez</p>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="w-6 h-6 bg-orange-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-orange-400 text-xs font-bold">4</span>
                  </div>
                  <p className="text-white/70">Utilisez votre modèle pour les embeddings</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 