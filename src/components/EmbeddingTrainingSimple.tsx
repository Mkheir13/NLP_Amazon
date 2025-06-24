import React, { useState, useEffect } from 'react';
import { Target, Settings, RefreshCw, CheckCircle, AlertCircle, BarChart3, Zap, Brain, Database, Lightbulb } from 'lucide-react';
import { DatasetLoader, Review } from '../services/DatasetLoader';
import ConfigManager from '../config/AppConfig';

interface EmbeddingTrainingSimpleProps {
  onClose?: () => void;
}

export const EmbeddingTrainingSimple: React.FC<EmbeddingTrainingSimpleProps> = ({ onClose }) => {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingData, setTrainingData] = useState<string>('amazon');
  const [customTexts, setCustomTexts] = useState<string>('');
  const [trainedStats, setTrainedStats] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [isLoadingDataset, setIsLoadingDataset] = useState(false);
  const [serviceAvailable, setServiceAvailable] = useState<boolean>(false);

  // Vérifier la disponibilité du service
  useEffect(() => {
    const checkService = async () => {
      try {
        const response = await fetch(`${ConfigManager.getApiUrl('embeddings')}/status`);
        const data = await response.json();
        setServiceAvailable(data.available);
      } catch (error) {
        console.error('Erreur vérification service:', error);
        setServiceAvailable(false);
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
      const loadedReviews = await DatasetLoader.loadAmazonPolarityDataset(1000);
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
    setTrainedStats(null);

    try {
      let texts: string[] = [];

      if (trainingData === 'amazon') {
        if (reviews.length === 0) {
          throw new Error('Dataset Amazon non chargé');
        }
        texts = reviews.map(review => review.text);
      } else if (trainingData === 'custom') {
        if (!customTexts.trim()) {
          throw new Error('Veuillez entrer des textes personnalisés');
        }
        texts = customTexts.split('\n').filter(text => text.trim().length > 0);
      }

      if (texts.length === 0) {
        throw new Error('Aucun texte disponible pour l\'entraînement');
      }

      console.log(`🚀 Démarrage entraînement TF-IDF sur ${texts.length} textes...`);
      
      const response = await fetch(`${ConfigManager.getApiUrl('embeddings')}/train/tfidf`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ texts }),
      });

      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Erreur lors de l\'entraînement');
      }

      setTrainedStats(data.stats);

    } catch (error) {
      setError(error instanceof Error ? error.message : 'Erreur lors de l\'entraînement');
    } finally {
      setIsTraining(false);
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
        {/* Header avec guide d'utilisation */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">Entraînement TF-IDF</h1>
            <p className="text-slate-300 text-lg">Étape 1 : Entraîner le modèle avant la visualisation</p>
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

        {/* Guide d'utilisation */}
        <div className="bg-blue-500/20 border border-blue-500/30 rounded-2xl p-6 mb-8">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
            <Lightbulb className="h-6 w-6 text-blue-400" />
            <span>Guide d'utilisation - Ordre obligatoire</span>
          </h3>
          <div className="space-y-3">
            <div className="bg-slate-700 p-3 rounded-xl">
              <p className="text-white font-bold">📋 Étape 1 : Entraînement TF-IDF (ICI)</p>
              <p className="text-slate-300 text-sm">Cliquez sur "Entraîner le modèle" avec le dataset Amazon</p>
            </div>
            <div className="bg-slate-700 p-3 rounded-xl">
              <p className="text-white font-bold">📊 Étape 2 : Visualisation</p>
              <p className="text-slate-300 text-sm">Allez sur "Visualize" → Entrez des mots → Voyez le graphique 2D</p>
            </div>
            <div className="bg-slate-700 p-3 rounded-xl">
              <p className="text-white font-bold">🔍 Étape 3 : Recherche sémantique</p>
              <p className="text-slate-300 text-sm">Allez sur "Search" → Recherchez des produits similaires</p>
            </div>
            <div className="bg-red-500/20 border border-red-500/30 p-3 rounded-xl">
              <p className="text-red-200 font-bold text-sm">⚠️ IMPORTANT : Sans entraînement, vous aurez des erreurs HTTP 400</p>
            </div>
          </div>
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
                  <option value="amazon">Dataset Amazon (1000 avis)</option>
                  <option value="custom">Textes personnalisés</option>
                </select>
              </div>

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

            {/* Information TF-IDF */}
            <div className="bg-slate-800/90 backdrop-blur-xl p-6 rounded-2xl border border-white/10">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
                <Settings className="h-6 w-6 text-purple-400" />
                <span>À propos de TF-IDF</span>
              </h3>

              <div className="space-y-4">
                <div className="p-4 bg-slate-700/30 rounded-xl">
                  <h4 className="text-white/80 font-medium mb-3">🔍 Fonctionnalités TF-IDF :</h4>
                  <div className="space-y-2 text-sm">
                    <div className="text-white/60">• <strong>TF-IDF :</strong> Term Frequency-Inverse Document Frequency</div>
                    <div className="text-white/60">• <strong>Vocabulaire :</strong> Maximum 5000 termes les plus importants</div>
                    <div className="text-white/60">• <strong>N-grammes :</strong> Mots individuels et bigrammes</div>
                    <div className="text-white/60">• <strong>Stop words :</strong> Filtrage automatique des mots vides</div>
                    <div className="text-white/60">• <strong>Fréquence :</strong> Minimum 2 occurrences par terme</div>
                  </div>
                </div>

                <div className="p-4 bg-blue-500/20 rounded-xl border border-blue-500/30">
                  <h4 className="text-blue-400 font-medium mb-2">💡 Avantages :</h4>
                  <ul className="text-blue-300 text-sm space-y-1">
                    <li>• Rapide et efficace</li>
                    <li>• Pas de compilation nécessaire</li>
                    <li>• Fonctionne avec scikit-learn</li>
                    <li>• Idéal pour la recherche sémantique</li>
                  </ul>
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
                    <span>Entraîner TF-IDF</span>
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

              {trainedStats && (
                <div className="bg-green-500/20 border border-green-500/30 rounded-xl p-4 mb-4">
                  <div className="flex items-center space-x-2 mb-3">
                    <CheckCircle className="h-5 w-5 text-green-400" />
                    <span className="text-green-400 font-medium">TF-IDF entraîné avec succès !</span>
                  </div>
                  
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-white/70">Vocabulaire :</span>
                      <span className="text-white font-medium">{trainedStats.vocabulary_size.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white/70">Corpus :</span>
                      <span className="text-white">{trainedStats.corpus_size.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white/70">Features :</span>
                      <span className="text-white">{trainedStats.features}</span>
                    </div>
                  </div>
                </div>
              )}

              {!trainedStats && !error && !isTraining && (
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
                    Vectorisation TF-IDF...
                  </p>
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
                  <p className="text-white/70">Lancez l'entraînement TF-IDF</p>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="w-6 h-6 bg-green-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-green-400 text-xs font-bold">3</span>
                  </div>
                  <p className="text-white/70">Utilisez pour la recherche sémantique</p>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="w-6 h-6 bg-orange-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-orange-400 text-xs font-bold">4</span>
                  </div>
                  <p className="text-white/70">Visualisez les embeddings</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 