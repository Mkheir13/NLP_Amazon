import React, { useState, useEffect } from 'react';
import { Search, Zap, Target, AlertCircle, CheckCircle, RefreshCw, FileText, TrendingUp, Hash, Clock, Star } from 'lucide-react';
import { EmbeddingService, SemanticSearchResult } from '../services/EmbeddingService';
import { DatasetLoader, Review } from '../services/DatasetLoader';

interface SemanticSearchProps {
  onClose?: () => void;
}

export const SemanticSearch: React.FC<SemanticSearchProps> = ({ onClose }) => {
  const [query, setQuery] = useState<string>('');
  const [results, setResults] = useState<SemanticSearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [serviceAvailable, setServiceAvailable] = useState<boolean>(false);
  const [searchCollection, setSearchCollection] = useState<string>('amazon');
  const [topK, setTopK] = useState<number>(10);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [isLoadingDataset, setIsLoadingDataset] = useState(false);
  const [customTexts, setCustomTexts] = useState<string>('');
  const [searchHistory, setSearchHistory] = useState<string[]>([]);

  // Vérifier la disponibilité du service
  useEffect(() => {
    const checkService = async () => {
      const available = await EmbeddingService.isEmbeddingServiceAvailable();
      setServiceAvailable(available);
    };
    checkService();
  }, []);

  // Charger le dataset Amazon
  useEffect(() => {
    if (serviceAvailable && searchCollection === 'amazon') {
      loadAmazonDataset();
    }
  }, [serviceAvailable, searchCollection]);

  const loadAmazonDataset = async () => {
    setIsLoadingDataset(true);
    try {
      const loadedReviews = await DatasetLoader.loadAmazonPolarityDataset(500);
      setReviews(loadedReviews);
    } catch (error) {
      console.error('Erreur chargement dataset:', error);
    } finally {
      setIsLoadingDataset(false);
    }
  };

  const performSemanticSearch = async () => {
    if (!query.trim()) {
      setError('Veuillez entrer une requête de recherche');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      let textsToSearch: string[] = [];

      if (searchCollection === 'amazon') {
        if (reviews.length === 0) {
          throw new Error('Dataset Amazon non chargé');
        }
        textsToSearch = reviews.map(review => review.text);
      } else if (searchCollection === 'custom') {
        if (!customTexts.trim()) {
          throw new Error('Veuillez entrer des textes personnalisés');
        }
        textsToSearch = customTexts.split('\n').filter(text => text.trim().length > 0);
      }

      if (textsToSearch.length === 0) {
        throw new Error('Aucun texte disponible pour la recherche');
      }

      const searchResults = await EmbeddingService.semanticSearch(query, textsToSearch, topK);
      setResults(searchResults);

      // Ajouter à l'historique
      if (!searchHistory.includes(query)) {
        setSearchHistory(prev => [query, ...prev.slice(0, 9)]); // Garder les 10 dernières recherches
      }

    } catch (error) {
      setError(error instanceof Error ? error.message : 'Erreur lors de la recherche');
    } finally {
      setIsLoading(false);
    }
  };

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 0.8) return 'text-green-400 bg-green-500/20 border-green-500/30';
    if (similarity >= 0.6) return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30';
    if (similarity >= 0.4) return 'text-orange-400 bg-orange-500/20 border-orange-500/30';
    return 'text-red-400 bg-red-500/20 border-red-500/30';
  };

  const getSimilarityLabel = (similarity: number) => {
    if (similarity >= 0.8) return 'Très similaire';
    if (similarity >= 0.6) return 'Similaire';
    if (similarity >= 0.4) return 'Moyennement similaire';
    return 'Peu similaire';
  };

  const handleHistoryClick = (historicQuery: string) => {
    setQuery(historicQuery);
  };

  if (!serviceAvailable) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
        <div className="max-w-4xl mx-auto">
          <div className="bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-red-500/30">
            <div className="text-center">
              <AlertCircle className="h-16 w-16 text-red-400 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-white mb-4">Service de Recherche Sémantique Non Disponible</h2>
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
            <h1 className="text-4xl font-bold text-white mb-2">Recherche Sémantique</h1>
            <p className="text-white/70 text-lg">Trouvez des textes similaires par le sens, pas seulement par les mots</p>
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

        <div className="grid lg:grid-cols-4 gap-8">
          {/* Panneau de recherche */}
          <div className="lg:col-span-1 space-y-6">
            {/* Configuration de recherche */}
            <div className="bg-slate-800/90 backdrop-blur-xl p-6 rounded-2xl border border-white/10">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
                <Search className="h-6 w-6 text-cyan-400" />
                <span>Recherche</span>
              </h3>

              {/* Collection de recherche */}
              <div className="mb-4">
                <label className="block text-white text-sm font-medium mb-2">
                  Collection de textes
                </label>
                <select
                  value={searchCollection}
                  onChange={(e) => setSearchCollection(e.target.value)}
                  className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white focus:outline-none focus:border-cyan-400"
                  style={{ color: 'white', backgroundColor: '#334155' }}
                >
                  <option value="amazon" style={{ color: 'white', backgroundColor: '#334155' }}>Avis Amazon (500)</option>
                  <option value="custom" style={{ color: 'white', backgroundColor: '#334155' }}>Textes personnalisés</option>
                </select>
              </div>

              {/* Nombre de résultats */}
              <div className="mb-4">
                <label className="block text-white text-sm font-medium mb-2">
                  Nombre de résultats
                </label>
                <select
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white focus:outline-none focus:border-cyan-400"
                  style={{ color: 'white', backgroundColor: '#334155' }}
                >
                  <option value={5} style={{ color: 'white', backgroundColor: '#334155' }}>5 résultats</option>
                  <option value={10} style={{ color: 'white', backgroundColor: '#334155' }}>10 résultats</option>
                  <option value={20} style={{ color: 'white', backgroundColor: '#334155' }}>20 résultats</option>
                  <option value={50} style={{ color: 'white', backgroundColor: '#334155' }}>50 résultats</option>
                </select>
              </div>

              {/* Textes personnalisés */}
              {searchCollection === 'custom' && (
                <div className="mb-4">
                  <label className="block text-white text-sm font-medium mb-2">
                    Textes personnalisés (un par ligne)
                  </label>
                  <textarea
                    value={customTexts}
                    onChange={(e) => setCustomTexts(e.target.value)}
                    placeholder="Entrez vos textes ici, un par ligne..."
                    className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:border-cyan-400 resize-none"
                    style={{ color: 'white', backgroundColor: '#334155' }}
                    rows={6}
                  />
                </div>
              )}

              {/* Requête de recherche */}
              <div className="mb-4">
                <label className="block text-white text-sm font-medium mb-2">
                  Requête de recherche
                </label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Décrivez ce que vous cherchez..."
                  className="w-full p-3 bg-slate-700 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:border-cyan-400 resize-none"
                  style={{ color: 'white', backgroundColor: '#334155' }}
                  rows={3}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && e.ctrlKey) {
                      performSemanticSearch();
                    }
                  }}
                />
                <p className="text-slate-400 text-xs mt-1">Ctrl+Entrée pour rechercher</p>
              </div>

              {/* Bouton de recherche */}
              <button
                onClick={performSemanticSearch}
                disabled={isLoading || isLoadingDataset}
                className="w-full px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="h-5 w-5 animate-spin" />
                    <span>Recherche...</span>
                  </>
                ) : (
                  <>
                    <Zap className="h-5 w-5" />
                    <span>Rechercher</span>
                  </>
                )}
              </button>

              {/* Statut du dataset */}
              {searchCollection === 'amazon' && (
                <div className="mt-4 p-3 bg-slate-700/50 rounded-xl">
                  <div className="flex items-center space-x-2">
                    {isLoadingDataset ? (
                      <>
                        <RefreshCw className="h-4 w-4 text-cyan-400 animate-spin" />
                        <span className="text-cyan-400 text-sm">Chargement...</span>
                      </>
                    ) : reviews.length > 0 ? (
                      <>
                        <CheckCircle className="h-4 w-4 text-green-400" />
                        <span className="text-green-400 text-sm">{reviews.length} avis chargés</span>
                      </>
                    ) : (
                      <>
                        <AlertCircle className="h-4 w-4 text-orange-400" />
                        <span className="text-orange-400 text-sm">Dataset non chargé</span>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Historique de recherche */}
            {searchHistory.length > 0 && (
              <div className="bg-slate-800/90 backdrop-blur-xl p-6 rounded-2xl border border-white/10">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
                  <Clock className="h-6 w-6 text-purple-400" />
                  <span>Historique</span>
                </h3>
                <div className="space-y-2">
                  {searchHistory.slice(0, 5).map((historicQuery, index) => (
                    <button
                      key={index}
                      onClick={() => handleHistoryClick(historicQuery)}
                      className="w-full text-left p-2 bg-white/5 hover:bg-white/10 rounded-lg text-white/70 hover:text-white transition-colors text-sm"
                    >
                      {historicQuery}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Résultats de recherche */}
          <div className="lg:col-span-3">
            <div className="bg-slate-800/90 backdrop-blur-xl p-6 rounded-2xl border border-white/10">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-white flex items-center space-x-2">
                  <Target className="h-6 w-6 text-green-400" />
                  <span>Résultats de Recherche</span>
                </h3>
                {results.length > 0 && (
                  <div className="text-white/60 text-sm">
                    {results.length} résultat{results.length > 1 ? 's' : ''} trouvé{results.length > 1 ? 's' : ''}
                  </div>
                )}
              </div>

              {error && (
                <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-4 mb-6">
                  <div className="flex items-center space-x-2">
                    <AlertCircle className="h-5 w-5 text-red-400" />
                    <span className="text-red-400 font-medium">Erreur</span>
                  </div>
                  <p className="text-red-300 mt-1">{error}</p>
                </div>
              )}

              {!results.length && !isLoading && !error && (
                <div className="flex flex-col items-center justify-center py-16 text-center">
                  <Search className="h-16 w-16 text-white/30 mb-4" />
                  <p className="text-white/60 text-lg mb-2">Aucune recherche effectuée</p>
                  <p className="text-white/40 text-sm">
                    Entrez une requête et cliquez sur "Rechercher" pour trouver des textes similaires
                  </p>
                </div>
              )}

              {isLoading && (
                <div className="flex flex-col items-center justify-center py-16">
                  <RefreshCw className="h-12 w-12 text-cyan-400 animate-spin mb-4" />
                  <p className="text-white/70 text-lg">Recherche sémantique en cours...</p>
                  <p className="text-white/50 text-sm mt-2">
                    Analyse des embeddings et calcul des similarités
                  </p>
                </div>
              )}

              {/* Résultats */}
              <div className="space-y-4">
                {results.map((result, index) => (
                  <div
                    key={index}
                    className="bg-slate-700/50 rounded-xl p-4 border border-white/10 hover:border-white/20 transition-colors"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <div className="flex items-center space-x-2">
                          <Hash className="h-4 w-4 text-white/40" />
                          <span className="text-white/60 text-sm">#{index + 1}</span>
                        </div>
                        <div className={`px-3 py-1 rounded-full text-xs font-medium border ${getSimilarityColor(result.similarity)}`}>
                          {getSimilarityLabel(result.similarity)}
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Star className="h-4 w-4 text-yellow-400" />
                        <span className="text-white font-medium">
                          {(result.similarity * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <div className="text-white/90 leading-relaxed">
                      {result.text_preview}
                    </div>

                    {result.text.length > result.text_preview.length && (
                      <button className="mt-2 text-cyan-400 hover:text-cyan-300 text-sm transition-colors">
                        Voir le texte complet...
                      </button>
                    )}
                  </div>
                ))}
              </div>

              {/* Suggestions de recherche */}
              {!results.length && !isLoading && !error && query && (
                <div className="mt-8 p-4 bg-slate-700/30 rounded-xl">
                  <h4 className="text-white/80 font-medium mb-3">💡 Conseils pour améliorer votre recherche :</h4>
                  <ul className="text-white/60 text-sm space-y-1">
                    <li>• Utilisez des phrases complètes plutôt que des mots-clés</li>
                    <li>• Décrivez le concept ou l'émotion recherchée</li>
                    <li>• Essayez des synonymes ou des formulations différentes</li>
                    <li>• La recherche sémantique comprend le contexte et les nuances</li>
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 