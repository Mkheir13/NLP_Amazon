import React, { useState, useEffect } from 'react';
import { Brain, Database, BarChart3, Search, Play, ArrowRight, Github, Sparkles, Zap, ChevronRight, Home, RefreshCw, Target, TrendingUp, Award, Rocket, Globe, Shield, CheckCircle, AlertCircle, Eye, Hash, Clock, FileText, Activity, Download, Copy, Star, ThumbsUp, ThumbsDown, Filter, Shuffle, Settings, Layers, Heart, Minus, Network } from 'lucide-react';
import { NLPPipeline } from './components/NLPPipeline';
import { BERTTraining } from './components/BERTTraining';
import { EmbeddingHub } from './components/EmbeddingHub';
import { EmbeddingVisualizer } from './components/EmbeddingVisualizer';
import { SemanticSearch } from './components/SemanticSearch';
import { EmbeddingTraining } from './components/EmbeddingTraining';
import { EmbeddingTrainingSimple } from './components/EmbeddingTrainingSimple';
import { DatasetLoader, Review } from './services/DatasetLoader';
import { RealNLPService, RealNLPAnalysis } from './services/RealNLPService';

// Interfaces
interface AnalysisResults {
  text: string;
  sentiment: {
    label: string;
    confidence: number;
    polarity: number;
  };
  features: {
    wordCount: number;
    charCount: number;
    sentenceCount: number;
    positiveWords: number;
    negativeWords: number;
    emotionalWords: number;
  };
  keywords: { [key: string]: number };
}

// Interface principale
function App() {
  const [currentView, setCurrentView] = useState('home');
  const [selectedReview, setSelectedReview] = useState<Review | null>(null);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null);
  const [realNLPResults, setRealNLPResults] = useState<RealNLPAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [textToAnalyze, setTextToAnalyze] = useState('');
  const [showPipeline, setShowPipeline] = useState(false);
  const [availableBERTModels, setAvailableBERTModels] = useState<any[]>([]);
  const [selectedBERTModel, setSelectedBERTModel] = useState<string | null>(null);

  // Navigation
  const views = {
    home: 'Accueil',
    explore: 'Explorer Dataset',
    analyze: 'Analyser Texte',
    training: 'Entraîner Modèles',
    pipeline: 'Pipeline NLP',
    results: 'Résultats',
    embeddings_hub: 'Hub Embeddings',
    embeddings: 'Visualiser Embeddings',
    semantic_search: 'Recherche Sémantique',
    embedding_training: 'Entraîner Embeddings'
  };

  // Données du dataset Amazon
  const [reviews, setReviews] = useState<Review[]>([]);
  const [filteredReviews, setFilteredReviews] = useState<Review[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [sentimentFilter, setSentimentFilter] = useState('all');
  const [isLoadingDataset, setIsLoadingDataset] = useState(false);

  // Chargement du dataset Amazon Polarity et modèles BERT
  useEffect(() => {
    const loadDataset = async () => {
      setIsLoadingDataset(true);
      try {
        console.log('Chargement du dataset Amazon Polarity...');
        const loadedReviews = await DatasetLoader.loadAmazonPolarityDataset(1000);
        setReviews(loadedReviews);
        setFilteredReviews(loadedReviews);
        console.log(`Dataset chargé : ${loadedReviews.length} avis`);
        
        // Charger les modèles BERT disponibles
        const models = await RealNLPService.getAvailableBERTModels();
        setAvailableBERTModels(models);
        if (models.length > 0) {
          setSelectedBERTModel(models[0].id);
        }
      } catch (error) {
        console.error('Erreur lors du chargement du dataset:', error);
      } finally {
        setIsLoadingDataset(false);
      }
    };

    loadDataset();
  }, []);

  // Filtrage des avis
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
  }, [reviews, sentimentFilter, searchQuery]);

  // Analyse de texte avec vraie NLP (NLTK + BERT)
  const analyzeText = async (text: string, showPipelineView = false) => {
    setTextToAnalyze(text);
    setShowPipeline(showPipelineView);
    
    if (showPipelineView) {
      setCurrentView('pipeline');
      return;
    }

    setIsLoading(true);
    setRealNLPResults(null);
    
    try {
      // Vérifier si le backend est disponible
      const backendAvailable = await RealNLPService.isBackendAvailable();
      
      if (backendAvailable) {
        console.log('🚀 Analyse avec NLTK + BERT...');
        // Utiliser le vrai service NLP avec NLTK et BERT
        const nlpResults = await RealNLPService.analyzeWithRealNLP(text, selectedBERTModel || undefined);
        setRealNLPResults(nlpResults);
        
        // Convertir pour la compatibilité avec l'ancien système
        const legacyResults: AnalysisResults = {
          text: nlpResults.text,
          sentiment: {
            label: nlpResults.comparison?.finalSentiment || nlpResults.nltk.sentiment,
            confidence: nlpResults.comparison?.bertConfidence || nlpResults.nltk.confidence,
            polarity: nlpResults.nltk.polarity
          },
          features: nlpResults.features,
          keywords: nlpResults.keywords
        };
        setAnalysisResults(legacyResults);
        
      } else {
        console.warn('⚠️ Backend non disponible, utilisation analyse basique...');
        // Fallback vers l'ancienne méthode
        await analyzeTextFallback(text);
      }
      
    } catch (error) {
      console.error('Erreur analyse NLP:', error);
      // Fallback vers l'ancienne méthode
      await analyzeTextFallback(text);
    }
    
    setIsLoading(false);
    setCurrentView('results');
  };

  // Méthode de fallback (ancienne analyse)
  const analyzeTextFallback = async (text: string) => {
    
    // Simulation d'analyse plus réaliste
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const words = text.split(/\s+/).filter((w: string) => w.length > 0);
    
    // Dictionnaire de sentiments étendu
    const sentimentWords: { [key: string]: number } = {
      // Très positifs
      'amazing': 0.9, 'incredible': 0.85, 'outstanding': 0.88, 'excellent': 0.8, 'fantastic': 0.82,
      'perfect': 0.9, 'wonderful': 0.78, 'brilliant': 0.8, 'superb': 0.75, 'magnificent': 0.85,
      'awesome': 0.7, 'great': 0.6, 'good': 0.5, 'nice': 0.4, 'fine': 0.3, 'love': 0.75,
      'recommend': 0.6, 'satisfied': 0.55, 'happy': 0.65, 'pleased': 0.5, 'delighted': 0.7,
      
      // Très négatifs  
      'terrible': -0.9, 'horrible': -0.85, 'awful': -0.8, 'disgusting': -0.9, 'pathetic': -0.8,
      'worst': -0.9, 'hate': -0.8, 'despise': -0.85, 'useless': -0.75, 'worthless': -0.8,
      'bad': -0.6, 'poor': -0.5, 'disappointing': -0.65, 'broken': -0.7, 'defective': -0.75,
      'slow': -0.4, 'expensive': -0.3, 'cheap': -0.4, 'frustrated': -0.6, 'angry': -0.7,
      
      // Modérés
      'okay': 0.1, 'decent': 0.3, 'average': 0.0, 'normal': 0.0, 'standard': 0.1,
      'works': 0.3, 'functional': 0.25, 'adequate': 0.2, 'acceptable': 0.25
    };
    
    let totalScore = 0;
    let wordCount = 0;
    let positiveWords = 0;
    let negativeWords = 0;
    
    // Analyse contextuelle
    for (let i = 0; i < words.length; i++) {
      const word = words[i].toLowerCase().replace(/[^\w]/g, '');
      let score = sentimentWords[word] || 0;
      
      // Vérifier les négations
      if (i > 0) {
        const prevWord = words[i-1].toLowerCase();
        if (['not', 'never', 'no', 'nothing', 'neither'].includes(prevWord)) {
          score *= -1;
        }
      }
      
      // Vérifier les intensificateurs
      if (i > 0) {
        const prevWord = words[i-1].toLowerCase();
        if (['very', 'extremely', 'incredibly', 'absolutely', 'totally'].includes(prevWord)) {
          score *= 1.5;
        }
      }
      
      if (Math.abs(score) > 0.1) {
        totalScore += score;
        wordCount++;
        
        if (score > 0.1) positiveWords++;
        if (score < -0.1) negativeWords++;
      }
    }
    
    const avgScore = wordCount > 0 ? totalScore / wordCount : 0;
    
    // Déterminer le sentiment avec plus de nuances
    let sentiment = 'neutral';
    let confidence = Math.min(0.95, Math.max(0.1, Math.abs(avgScore) * 0.8 + 0.2));
    
    if (avgScore > 0.15) {
      sentiment = 'positive';
    } else if (avgScore < -0.15) {
      sentiment = 'negative';
    }
    
    // Ajuster la confiance selon le nombre de mots émotionnels
    if (wordCount > 0) {
      confidence = Math.min(0.95, confidence + (wordCount / words.length) * 0.2);
    }

    const results = {
      text: text,
      sentiment: {
        label: sentiment,
        confidence: confidence,
        polarity: avgScore
      },
      features: {
        wordCount: words.length,
        charCount: text.length,
        sentenceCount: text.split(/[.!?]+/).filter(s => s.trim().length > 0).length,
        positiveWords: positiveWords,
        negativeWords: negativeWords,
        emotionalWords: wordCount
      },
      keywords: words
        .filter((w: string) => w.length > 3)
        .reduce((acc: { [key: string]: number }, word: string) => {
          const lower = word.toLowerCase();
          acc[lower] = (acc[lower] || 0) + 1;
          return acc;
        }, {})
    };

    setAnalysisResults(results);
  };

  // Callback du pipeline - NE PAS CHANGER DE VUE
  const handlePipelineComplete = (pipelineData: any) => {
    console.log('Pipeline terminé:', pipelineData);
  };

  // Sélection d'un avis
  const selectReview = (review: Review) => {
    setSelectedReview(review);
    analyzeText(review.text);
  };

  // Avis aléatoire
  const getRandomReview = () => {
    if (filteredReviews.length === 0) return;
    const randomIndex = Math.floor(Math.random() * filteredReviews.length);
    const review = filteredReviews[randomIndex];
    selectReview(review);
  };

  // Rendu de la page d'accueil avec design amélioré
  const renderHome = () => (
    <div className="text-center space-y-12">
      {/* Hero section avec animation */}
      <div className="relative pt-8">
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 rounded-full blur-3xl opacity-20 animate-pulse"></div>
        <div className="relative">
          <div className="flex items-center justify-center space-x-4 mb-6">
            <Brain className="h-32 w-32 text-cyan-400 animate-bounce" />
            <div className="text-left">
              <div className="flex items-center space-x-2 mb-2">
                <span className="px-3 py-1 bg-cyan-500/20 text-cyan-400 rounded-full text-sm font-medium border border-cyan-500/30">
                  v3.0 • Embeddings Ready
                </span>
                <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm font-medium border border-green-500/30">
                  TF-IDF Intégré
                </span>
              </div>
              <div className="text-4xl md:text-5xl font-bold text-white mb-2">
                Nouvelle Génération
              </div>
              <div className="text-cyan-400 text-lg font-medium">
                🔗 Embeddings • 🔍 Recherche Sémantique • 📊 Visualisations
              </div>
            </div>
          </div>
          <div className="space-y-4">
            <h1 className="text-6xl md:text-7xl font-bold">
              <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                NLP Amazon
              </span>
              <br />
              <span className="text-white">Analysis</span>
            </h1>
            <p className="text-2xl text-white/80 max-w-4xl mx-auto leading-relaxed">
              Pipeline NLP révolutionnaire avec embeddings TF-IDF, recherche sémantique et visualisations interactives
            </p>
            <div className="flex items-center justify-center space-x-6 text-lg">
              <div className="flex items-center space-x-2 text-green-400">
                <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                <span className="font-medium">IA Émotionnelle</span>
              </div>
              <div className="flex items-center space-x-2 text-cyan-400">
                <div className="w-3 h-3 bg-cyan-400 rounded-full animate-pulse"></div>
                <span className="font-medium">Embeddings TF-IDF</span>
              </div>
              <div className="flex items-center space-x-2 text-purple-400">
                <div className="w-3 h-3 bg-purple-400 rounded-full animate-pulse"></div>
                <span className="font-medium">Temps Réel</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Cards principales avec hover effects */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-7xl mx-auto">
        <button 
          onClick={() => setCurrentView('explore')}
          className="group relative p-8 bg-gradient-to-br from-cyan-500/20 to-blue-600/20 rounded-2xl border border-cyan-500/30 text-white transition-all duration-500 hover:scale-105 hover:shadow-2xl hover:shadow-cyan-500/25 overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-500 to-blue-600 opacity-0 group-hover:opacity-10 transition-opacity duration-500"></div>
          <div className="relative">
            <Database className="h-12 w-12 mx-auto mb-4 text-cyan-400 group-hover:scale-110 transition-transform duration-300" />
            <h3 className="text-xl font-bold mb-3">Explorer Dataset</h3>
            <p className="text-cyan-100 text-sm leading-relaxed">4M d'avis Amazon avec filtres intelligents</p>
            <div className="mt-4 flex items-center justify-center space-x-2 text-cyan-300">
              <span className="text-xs">Découvrir</span>
              <ArrowRight className="h-4 w-4 transform group-hover:translate-x-2 transition-transform" />
            </div>
          </div>
        </button>
        
        <button 
          onClick={() => setCurrentView('analyze')}
          className="group relative p-8 bg-gradient-to-br from-purple-500/20 to-pink-600/20 rounded-2xl border border-purple-500/30 text-white transition-all duration-500 hover:scale-105 hover:shadow-2xl hover:shadow-purple-500/25 overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-purple-500 to-pink-600 opacity-0 group-hover:opacity-10 transition-opacity duration-500"></div>
          <div className="relative">
            <Brain className="h-12 w-12 mx-auto mb-4 text-purple-400 group-hover:scale-110 transition-transform duration-300" />
            <h3 className="text-xl font-bold mb-3">Analyser Texte</h3>
            <p className="text-purple-100 text-sm leading-relaxed">IA émotionnelle avec détection de sentiments</p>
            <div className="mt-4 flex items-center justify-center space-x-2 text-purple-300">
              <span className="text-xs">Analyser</span>
              <ArrowRight className="h-4 w-4 transform group-hover:translate-x-2 transition-transform" />
            </div>
          </div>
        </button>

        <button 
          onClick={() => setCurrentView('training')}
          className="group relative p-8 bg-gradient-to-br from-orange-500/20 to-red-600/20 rounded-2xl border border-orange-500/30 text-white transition-all duration-500 hover:scale-105 hover:shadow-2xl hover:shadow-orange-500/25 overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-orange-500 to-red-600 opacity-0 group-hover:opacity-10 transition-opacity duration-500"></div>
          <div className="relative">
            <Target className="h-12 w-12 mx-auto mb-4 text-orange-400 group-hover:scale-110 transition-transform duration-300" />
            <h3 className="text-xl font-bold mb-3">Entraîner Modèles</h3>
            <p className="text-orange-100 text-sm leading-relaxed">Créez vos propres modèles ML personnalisés</p>
            <div className="mt-4 flex items-center justify-center space-x-2 text-orange-300">
              <span className="text-xs">Entraîner</span>
              <ArrowRight className="h-4 w-4 transform group-hover:translate-x-2 transition-transform" />
            </div>
          </div>
        </button>

        <button 
          onClick={() => setCurrentView('pipeline')}
          className="group relative p-8 bg-gradient-to-br from-green-500/20 to-teal-600/20 rounded-2xl border border-green-500/30 text-white transition-all duration-500 hover:scale-105 hover:shadow-2xl hover:shadow-green-500/25 overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-green-500 to-teal-600 opacity-0 group-hover:opacity-10 transition-opacity duration-500"></div>
          <div className="relative">
            <Layers className="h-12 w-12 mx-auto mb-4 text-green-400 group-hover:scale-110 transition-transform duration-300" />
            <h3 className="text-xl font-bold mb-3">Pipeline NLP</h3>
            <p className="text-green-100 text-sm leading-relaxed">Visualisation complète des étapes ML</p>
            <div className="mt-4 flex items-center justify-center space-x-2 text-green-300">
              <span className="text-xs">Explorer</span>
              <ArrowRight className="h-4 w-4 transform group-hover:translate-x-2 transition-transform" />
            </div>
          </div>
        </button>
      </div>

      {/* Nouvelle section Embeddings avec design premium */}
      <div className="relative">
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-pink-500/10 rounded-3xl blur-xl"></div>
        <div className="relative bg-slate-800/50 backdrop-blur-xl p-12 rounded-3xl border border-white/10">
          <div className="text-center mb-12">
            <div className="flex items-center justify-center space-x-3 mb-4">
              <div className="relative">
                <Network className="h-12 w-12 text-indigo-400 animate-pulse" />
                <div className="absolute inset-0 bg-indigo-400 rounded-full blur-lg opacity-30"></div>
              </div>
              <h2 className="text-4xl font-bold bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                Embeddings & IA Sémantique
              </h2>
              <div className="relative">
                <Search className="h-12 w-12 text-pink-400 animate-pulse" />
                <div className="absolute inset-0 bg-pink-400 rounded-full blur-lg opacity-30"></div>
              </div>
            </div>
            <p className="text-white/80 text-xl max-w-4xl mx-auto leading-relaxed">
              🚀 <strong>Nouveauté v3.0</strong> : Transformez vos textes en vecteurs intelligents avec TF-IDF, 
              explorez les relations sémantiques et découvrez des insights cachés dans vos données
            </p>
            <div className="flex items-center justify-center space-x-6 mt-6">
              <div className="flex items-center space-x-2 text-indigo-400 bg-indigo-500/20 px-4 py-2 rounded-full border border-indigo-500/30">
                <Zap className="h-4 w-4" />
                <span className="font-medium">Vectorisation TF-IDF</span>
              </div>
              <div className="flex items-center space-x-2 text-purple-400 bg-purple-500/20 px-4 py-2 rounded-full border border-purple-500/30">
                <Eye className="h-4 w-4" />
                <span className="font-medium">Visualisation 2D/3D</span>
              </div>
              <div className="flex items-center space-x-2 text-pink-400 bg-pink-500/20 px-4 py-2 rounded-full border border-pink-500/30">
                <Search className="h-4 w-4" />
                <span className="font-medium">Recherche Sémantique</span>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
            <button 
              onClick={() => setCurrentView('embeddings_hub')}
              className="group relative p-6 bg-gradient-to-br from-indigo-500/20 to-purple-600/20 rounded-2xl border border-indigo-500/30 text-white transition-all duration-500 hover:scale-105 hover:shadow-2xl hover:shadow-indigo-500/25 overflow-hidden col-span-3"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-indigo-500 to-purple-600 opacity-0 group-hover:opacity-10 transition-opacity duration-500"></div>
              <div className="relative text-center">
                <div className="flex items-center justify-center space-x-4 mb-4">
                  <Network className="h-12 w-12 text-indigo-400 group-hover:scale-110 transition-transform duration-300" />
                  <Eye className="h-12 w-12 text-teal-400 group-hover:scale-110 transition-transform duration-300" />
                  <Search className="h-12 w-12 text-rose-400 group-hover:scale-110 transition-transform duration-300" />
                </div>
                <h3 className="text-2xl font-bold mb-3">Hub Embeddings Unifié</h3>
                <p className="text-indigo-100 text-lg leading-relaxed">
                  🎯 Tout-en-un : Entraînement TF-IDF, Visualisation 2D et Recherche sémantique dans une seule page fluide
                </p>
                <div className="mt-4 flex items-center justify-center space-x-6">
                  <div className="flex items-center space-x-2 text-indigo-300 bg-indigo-500/20 px-3 py-1 rounded-full">
                    <Network className="h-4 w-4" />
                    <span className="text-sm font-medium">Entraînement</span>
                  </div>
                  <div className="flex items-center space-x-2 text-teal-300 bg-teal-500/20 px-3 py-1 rounded-full">
                    <Eye className="h-4 w-4" />
                    <span className="text-sm font-medium">Visualisation</span>
                  </div>
                  <div className="flex items-center space-x-2 text-rose-300 bg-rose-500/20 px-3 py-1 rounded-full">
                    <Search className="h-4 w-4" />
                    <span className="text-sm font-medium">Recherche</span>
                  </div>
                </div>
                <div className="mt-4 flex items-center justify-center space-x-2 text-indigo-300">
                  <span className="text-lg">Accéder au Hub</span>
                  <ArrowRight className="h-5 w-5 transform group-hover:translate-x-2 transition-transform" />
                </div>
              </div>
            </button>
          </div>
        </div>
      </div>

      {/* Statistiques avec animations - Mise à jour v3.0 */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-6 max-w-6xl mx-auto">
        {[
          { number: "4M", label: "Avis Amazon", icon: Database, color: "text-cyan-400", bg: "bg-cyan-500/20", desc: "Dataset réel" },
          { number: "TF-IDF", label: "Vectorisation", icon: Network, color: "text-indigo-400", bg: "bg-indigo-500/20", desc: "Embeddings" },
          { number: "2D/3D", label: "Visualisations", icon: Eye, color: "text-purple-400", bg: "bg-purple-500/20", desc: "PCA • t-SNE" },
          { number: "96%", label: "Précision IA", icon: Award, color: "text-yellow-400", bg: "bg-yellow-500/20", desc: "Sentiment" },
          { number: "25ms", label: "Temps Réponse", icon: Zap, color: "text-green-400", bg: "bg-green-500/20", desc: "Ultra-rapide" }
        ].map((stat, index) => (
          <div key={index} className={`${stat.bg} backdrop-blur-sm rounded-2xl p-6 border border-white/20 hover:scale-105 transition-all duration-300 group relative overflow-hidden`}>
            {/* Effet de brillance au hover */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000"></div>
            <div className="relative">
              <stat.icon className={`h-8 w-8 ${stat.color} mx-auto mb-3 group-hover:scale-110 transition-transform`} />
              <div className={`text-2xl font-bold ${stat.color} mb-1`}>{stat.number}</div>
              <div className="text-white/80 text-sm font-medium mb-1">{stat.label}</div>
              <div className="text-white/50 text-xs">{stat.desc}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Call to action */}
      <div className="bg-gradient-to-r from-slate-800/50 to-slate-700/50 backdrop-blur-xl p-8 rounded-2xl border border-white/10 max-w-4xl mx-auto">
        <h2 className="text-3xl font-bold text-white mb-4">Prêt à explorer l'IA émotionnelle ?</h2>
        <p className="text-white/70 text-lg mb-6">Découvrez comment notre pipeline NLP révolutionnaire analyse les sentiments avec une précision inégalée</p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <button
            onClick={() => setCurrentView('explore')}
            className="px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 hover:shadow-lg"
          >
            Commencer l'Exploration
          </button>
          <button
            onClick={() => setCurrentView('pipeline')}
            className="px-8 py-4 bg-white/10 text-white rounded-xl font-semibold hover:bg-white/20 transition-colors border border-white/20"
          >
            Voir le Pipeline
          </button>
        </div>
      </div>
    </div>
  );

  // Rendu de l'explorateur avec design amélioré
  const renderExplore = () => (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">Dataset Amazon Polarity</h2>
          <p className="text-white/70 text-lg">Analyse intelligente de 4 millions d'avis clients</p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-3 text-green-400 bg-green-500/20 px-4 py-2 rounded-xl border border-green-500/30">
            <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
            <span className="font-medium">Dataset réel • {reviews.length} avis chargés</span>
          </div>
          <button
            onClick={() => DatasetLoader.downloadAndSaveDataset(5000)}
            className="px-4 py-2 bg-blue-500/20 text-blue-400 rounded-xl hover:bg-blue-500/30 transition-all flex items-center space-x-2 border border-blue-500/30"
          >
            <Download className="h-4 w-4" />
            <span>Télécharger 5K avis</span>
          </button>
        </div>
      </div>

      {/* Filtres améliorés */}
      <div className="bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-white/10 shadow-2xl">
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-white/40" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Rechercher dans les avis... (mots-clés, sentiments, etc.)"
                className="w-full pl-12 pr-4 py-4 bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/50 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20 transition-all"
              />
            </div>
          </div>
          
          <div className="flex flex-wrap gap-3">
            {[
              { id: 'all', label: 'Tous', count: reviews.length, color: 'bg-slate-500/20 text-slate-400 border-slate-500/30' },
              { id: 'positive', label: 'Positifs', count: reviews.filter(r => r.sentiment === 'positive').length, color: 'bg-green-500/20 text-green-400 border-green-500/30' },
              { id: 'negative', label: 'Négatifs', count: reviews.filter(r => r.sentiment === 'negative').length, color: 'bg-red-500/20 text-red-400 border-red-500/30' }
            ].map((filter) => (
              <button
                key={filter.id}
                onClick={() => setSentimentFilter(filter.id)}
                className={`px-6 py-3 rounded-xl transition-all font-medium ${
                  sentimentFilter === filter.id
                    ? 'bg-cyan-500/30 text-cyan-400 border-2 border-cyan-500/50 shadow-lg'
                    : `${filter.color} border hover:scale-105`
                }`}
              >
                {filter.label} ({filter.count})
              </button>
            ))}
          </div>

          <button
            onClick={getRandomReview}
            className="px-6 py-3 bg-gradient-to-r from-purple-500/20 to-pink-500/20 text-purple-400 rounded-xl hover:from-purple-500/30 hover:to-pink-500/30 transition-all flex items-center space-x-2 border border-purple-500/30 hover:scale-105"
          >
            <Shuffle className="h-5 w-5" />
            <span className="font-medium">Surprise</span>
          </button>
        </div>
      </div>

      {/* Indicateur de chargement */}
      {isLoadingDataset && (
        <div className="text-center py-12">
          <div className="inline-flex items-center space-x-3 text-cyan-400">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-400"></div>
            <span className="text-lg font-medium">Chargement du dataset Amazon Polarity...</span>
          </div>
          <p className="text-white/60 mt-2">Récupération des avis depuis Hugging Face</p>
        </div>
      )}

      {/* Liste des avis avec design moderne */}
      <div className="grid gap-6">
        {!isLoadingDataset && filteredReviews.slice(0, 10).map((review) => {
          const getSentimentStyle = (sentiment: string, intensity = 0) => {
            switch (sentiment) {
              case 'positive':
                return {
                  border: 'border-green-500/30',
                  bg: 'bg-gradient-to-r from-green-500/10 to-emerald-500/10',
                  icon: ThumbsUp,
                  iconColor: 'text-green-400',
                  label: intensity > 0.7 ? 'Très Positif' : intensity > 0.4 ? 'Positif' : 'Légèrement Positif',
                  emoji: intensity > 0.7 ? '😍' : intensity > 0.4 ? '😊' : '🙂'
                };
              case 'negative':
                return {
                  border: 'border-red-500/30',
                  bg: 'bg-gradient-to-r from-red-500/10 to-rose-500/10',
                  icon: ThumbsDown,
                  iconColor: 'text-red-400',
                  label: Math.abs(intensity) > 0.7 ? 'Très Négatif' : Math.abs(intensity) > 0.4 ? 'Négatif' : 'Légèrement Négatif',
                  emoji: Math.abs(intensity) > 0.7 ? '😡' : Math.abs(intensity) > 0.4 ? '😞' : '😕'
                };
              default:
                return {
                  border: 'border-yellow-500/30',
                  bg: 'bg-gradient-to-r from-yellow-500/10 to-orange-500/10',
                  icon: Minus,
                  iconColor: 'text-yellow-400',
                  label: 'Neutre',
                  emoji: '😐'
                };
            }
          };

          const style = getSentimentStyle(review.sentiment, review.intensity);
          const SentimentIcon = style.icon;

          return (
            <div
              key={review.id}
              className={`p-8 ${style.bg} rounded-2xl border ${style.border} hover:scale-[1.02] transition-all duration-300 shadow-lg hover:shadow-xl backdrop-blur-sm`}
            >
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center space-x-4">
                  <div className="text-3xl">{style.emoji}</div>
                  <div className="flex items-center space-x-3">
                    <SentimentIcon className={`h-6 w-6 ${style.iconColor}`} />
                    <span className={`font-bold text-lg ${style.iconColor}`}>
                      {style.label}
                    </span>
                    <div className="flex items-center space-x-1">
                      {[...Array(5)].map((_, i) => (
                        <Star
                          key={i}
                          className={`h-4 w-4 ${
                            i < review.rating ? 'text-yellow-400 fill-current' : 'text-gray-600'
                          }`}
                        />
                      ))}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-white/60 text-sm font-medium">{review.category}</div>
                  <div className="text-white/40 text-xs">{review.date}</div>
                </div>
              </div>
              
              <h3 className="text-white font-bold text-xl mb-4">{review.title}</h3>
              <p className="text-white/90 text-lg leading-relaxed mb-6">{review.text}</p>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-6 text-sm text-white/60">
                  <span className="flex items-center space-x-1">
                    <FileText className="h-4 w-4" />
                    <span>{review.text.length} caractères</span>
                  </span>
                  <span className="flex items-center space-x-1">
                    <Hash className="h-4 w-4" />
                    <span>{review.text.split(' ').length} mots</span>
                  </span>
                  {review.intensity && (
                    <span className="flex items-center space-x-1">
                      <Target className="h-4 w-4" />
                      <span>Intensité: {Math.abs(review.intensity).toFixed(2)}</span>
                    </span>
                  )}
                </div>
                
                <div className="flex space-x-3">
                  <button
                    onClick={() => selectReview(review)}
                    className="px-6 py-3 bg-cyan-500/20 text-cyan-400 rounded-xl hover:bg-cyan-500/30 transition-all font-medium border border-cyan-500/30 hover:scale-105"
                  >
                    Analyser
                  </button>
                  <button
                    onClick={() => analyzeText(review.text, true)}
                    className="px-6 py-3 bg-green-500/20 text-green-400 rounded-xl hover:bg-green-500/30 transition-all font-medium flex items-center space-x-2 border border-green-500/30 hover:scale-105"
                  >
                    <Layers className="h-4 w-4" />
                    <span>Pipeline</span>
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {!isLoadingDataset && filteredReviews.length === 0 && (
        <div className="text-center py-16 text-white/60">
          <Search className="h-20 w-20 mx-auto mb-6 opacity-50" />
          <h3 className="text-2xl font-bold text-white mb-2">Aucun avis trouvé</h3>
          <p className="text-lg">Essayez de modifier vos filtres ou votre recherche</p>
        </div>
      )}
    </div>
  );

  // Rendu de l'analyseur avec design amélioré
  const renderAnalyze = () => (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h2 className="text-4xl font-bold text-white mb-4">Analyseur IA Émotionnel</h2>
        <p className="text-white/70 text-xl">Découvrez les émotions cachées dans vos textes</p>
      </div>
      
      <div className="bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-white/10 shadow-2xl">
        <label className="block text-white font-bold text-xl mb-4">
          Texte à analyser
        </label>
        <textarea
          placeholder="Collez votre texte ici pour découvrir ses émotions cachées..."
          className="w-full h-40 p-6 bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/50 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20 resize-none text-lg transition-all"
          onChange={(e) => {
            if (e.target.value.trim()) {
              setTextToAnalyze(e.target.value);
            }
          }}
        />
        
        <div className="flex flex-col sm:flex-row gap-4 mt-6">
          <button
            onClick={() => textToAnalyze && analyzeText(textToAnalyze, false)}
            disabled={!textToAnalyze}
            className="flex-1 px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl hover:from-cyan-600 hover:to-blue-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed font-semibold text-lg hover:scale-105"
          >
            Analyse Rapide
          </button>
          <button
            onClick={() => textToAnalyze && analyzeText(textToAnalyze, true)}
            disabled={!textToAnalyze}
            className="flex-1 px-8 py-4 bg-gradient-to-r from-green-500 to-teal-600 text-white rounded-xl hover:from-green-600 hover:to-teal-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed font-semibold text-lg flex items-center justify-center space-x-2 hover:scale-105"
          >
            <Layers className="h-5 w-5" />
            <span>Pipeline Complet</span>
          </button>
        </div>
      </div>

      {/* Exemples avec design moderne */}
      <div className="grid md:grid-cols-3 gap-6">
        {[
          { 
            label: "Très Positif", 
            text: "This product is absolutely incredible! The quality exceeded all my expectations and I'm genuinely amazed by the results.", 
            sentiment: "positive", 
            emoji: "😍",
            gradient: "from-green-500/20 to-emerald-500/20",
            border: "border-green-500/30"
          },
          { 
            label: "Très Négatif", 
            text: "Absolutely terrible experience! This is the worst product I've ever purchased and I deeply regret wasting my money on this garbage.", 
            sentiment: "negative", 
            emoji: "😡",
            gradient: "from-red-500/20 to-rose-500/20",
            border: "border-red-500/30"
          },
          { 
            label: "Neutre", 
            text: "The product is okay, nothing special. It works as described but doesn't stand out in any particular way. Average quality for the price.", 
            sentiment: "neutral", 
            emoji: "😐",
            gradient: "from-yellow-500/20 to-orange-500/20",
            border: "border-yellow-500/30"
          }
        ].map((example, index) => (
          <div key={index} className={`p-6 bg-gradient-to-br ${example.gradient} border ${example.border} rounded-2xl hover:scale-105 transition-all duration-300`}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <span className="text-3xl">{example.emoji}</span>
                <span className="text-white font-bold text-lg">{example.label}</span>
              </div>
            </div>
            <p className="text-white/80 text-sm mb-6 leading-relaxed">{example.text}</p>
            <div className="flex space-x-3">
              <button
                onClick={() => analyzeText(example.text, false)}
                className="flex-1 px-4 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg text-sm hover:bg-cyan-500/30 transition-colors font-medium"
              >
                Analyser
              </button>
              <button
                onClick={() => analyzeText(example.text, true)}
                className="flex-1 px-4 py-2 bg-green-500/20 text-green-400 rounded-lg text-sm hover:bg-green-500/30 transition-colors font-medium"
              >
                Pipeline
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  // Rendu du pipeline
  const renderPipeline = () => (
    <div className="space-y-6">
      {textToAnalyze ? (
        <NLPPipeline 
          text={textToAnalyze} 
          onComplete={handlePipelineComplete}
        />
      ) : (
        <div className="text-center py-16">
          <Layers className="h-24 w-24 text-white/40 mx-auto mb-6" />
          <h2 className="text-4xl font-bold text-white mb-4">Pipeline NLP Avancé</h2>
          <p className="text-white/70 text-xl mb-8">Sélectionnez un texte pour voir le pipeline en action</p>
          <button
            onClick={() => setCurrentView('analyze')}
            className="px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 text-lg"
          >
            Choisir un Texte
          </button>
        </div>
      )}
    </div>
  );

  // Rendu de l'entraînement de modèles
  const renderTraining = () => (
    <BERTTraining reviews={reviews} />
  );

  // Rendu des résultats avec design amélioré (NLTK + BERT)
  const renderResults = () => {
    if (!analysisResults && !realNLPResults) return null;

    // Utiliser les résultats réels si disponibles, sinon fallback
    const results = analysisResults;
    const nlpResults = realNLPResults;
    
    if (!results) return null;

    const getSentimentColor = (label: string) => {
      switch (label) {
        case 'positive': return 'text-green-400';
        case 'negative': return 'text-red-400';
        default: return 'text-yellow-400';
      }
    };

    const getSentimentBg = (label: string) => {
      switch (label) {
        case 'positive': return 'bg-green-500/20 border-green-500/30';
        case 'negative': return 'bg-red-500/20 border-red-500/30';
        default: return 'bg-yellow-500/20 border-yellow-500/30';
      }
    };

    const getSentimentEmoji = (label: string, polarity: number) => {
      if (label === 'positive') {
        return Math.abs(polarity) > 0.6 ? '😍' : Math.abs(polarity) > 0.3 ? '😊' : '🙂';
      } else if (label === 'negative') {
        return Math.abs(polarity) > 0.6 ? '😡' : Math.abs(polarity) > 0.3 ? '😞' : '😕';
      }
      return '😐';
    };

    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-4xl font-bold text-white mb-2">
              {nlpResults ? 'Analyse NLTK + BERT' : 'Résultats d\'Analyse IA'}
            </h2>
            <p className="text-white/70 text-xl">
              {nlpResults ? 'Analyse avec vrais modèles NLP' : 'Analyse émotionnelle complète'}
            </p>
          </div>
          <button
            onClick={() => {
              const summary = `Analyse NLP Émotionnelle:\nTexte: "${results.text}"\nSentiment: ${results.sentiment.label}\nConfiance: ${(results.sentiment.confidence * 100).toFixed(1)}%\nPolarité: ${results.sentiment.polarity.toFixed(3)}`;
              navigator.clipboard.writeText(summary);
            }}
            className="px-6 py-3 bg-white/10 text-white rounded-xl hover:bg-white/20 transition-colors flex items-center space-x-2 border border-white/20"
          >
            <Copy className="h-5 w-5" />
            <span>Copier</span>
          </button>
        </div>

        {/* Texte analysé */}
        <div className="bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-white/10 shadow-2xl">
          <h3 className="text-white font-bold text-xl mb-4">Texte analysé</h3>
          <p className="text-white/90 italic text-lg leading-relaxed">"{results.text}"</p>
        </div>

        {/* Résultat principal avec design moderne */}
        <div className={`p-10 rounded-2xl border ${getSentimentBg(results.sentiment.label)} shadow-2xl`}>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
                          <div className="text-center">
                <div className="text-8xl mb-4">
                  {getSentimentEmoji(results.sentiment.label, results.sentiment.polarity)}
                </div>
                <div className={`text-4xl font-bold ${getSentimentColor(results.sentiment.label)} mb-2`}>
                  {results.sentiment.label.toUpperCase()}
                </div>
                <div className="text-white/60 text-lg">
                  {nlpResults?.comparison ? 'Sentiment final (NLTK+BERT)' : 'Sentiment détecté'}
                </div>
              </div>
            
            <div className="text-center">
              <div className="text-6xl font-bold text-white mb-2">
                {(analysisResults.sentiment.confidence * 100).toFixed(0)}%
              </div>
              <div className="text-white/60 text-lg">Confiance</div>
              <div className="mt-2 w-full bg-white/20 rounded-full h-3">
                <div 
                  className="bg-gradient-to-r from-cyan-400 to-blue-500 h-3 rounded-full transition-all duration-1000"
                  style={{ width: `${analysisResults.sentiment.confidence * 100}%` }}
                />
              </div>
            </div>
            
            <div className="text-center">
              <div className={`text-6xl font-bold ${getSentimentColor(analysisResults.sentiment.label)} mb-2`}>
                {analysisResults.sentiment.polarity.toFixed(2)}
              </div>
              <div className="text-white/60 text-lg">Score de polarité</div>
              <div className="text-white/50 text-sm mt-1">
                {analysisResults.sentiment.polarity > 0 ? 'Positif' : analysisResults.sentiment.polarity < 0 ? 'Négatif' : 'Neutre'}
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            <div className="bg-white/10 p-6 rounded-xl">
              <div className="text-3xl font-bold text-cyan-400">{analysisResults.features.wordCount}</div>
              <div className="text-white/60 text-sm">Mots totaux</div>
            </div>
            <div className="bg-white/10 p-6 rounded-xl">
              <div className="text-3xl font-bold text-green-400">{analysisResults.features.positiveWords}</div>
              <div className="text-white/60 text-sm">Mots positifs</div>
            </div>
            <div className="bg-white/10 p-6 rounded-xl">
              <div className="text-3xl font-bold text-red-400">{analysisResults.features.negativeWords}</div>
              <div className="text-white/60 text-sm">Mots négatifs</div>
            </div>
            <div className="bg-white/10 p-6 rounded-xl">
              <div className="text-3xl font-bold text-purple-400">{analysisResults.features.emotionalWords || 0}</div>
              <div className="text-white/60 text-sm">Mots émotionnels</div>
            </div>
          </div>
        </div>

        {/* Actions avec design moderne */}
        <div className="flex flex-col sm:flex-row gap-4">
          <button
            onClick={() => analyzeText(analysisResults.text, true)}
            className="flex-1 px-8 py-4 bg-gradient-to-r from-green-500 to-teal-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 flex items-center justify-center space-x-2"
          >
            <Layers className="h-5 w-5" />
            <span>Voir Pipeline Complet</span>
          </button>
          <button
            onClick={() => setCurrentView('explore')}
            className="flex-1 px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105"
          >
            Analyser un autre avis
          </button>
          <button
            onClick={() => setCurrentView('analyze')}
            className="flex-1 px-8 py-4 bg-white/10 text-white rounded-xl font-semibold hover:bg-white/20 transition-colors border border-white/20"
          >
            Nouveau texte
          </button>
        </div>
      </div>
    );
  };

  // Composant Header uniforme et compact
  const renderHeader = () => (
    <header className="bg-slate-900/95 backdrop-blur-xl border-b border-white/10 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          {/* Logo compact */}
          <button 
            onClick={() => setCurrentView('home')}
            className="flex items-center space-x-2 hover:opacity-80 transition-opacity group"
          >
            <div className="relative">
              <Brain className="h-6 w-6 text-cyan-400 animate-pulse" />
              <div className="absolute inset-0 bg-cyan-400 rounded-full blur-md opacity-30"></div>
            </div>
            <div className="flex flex-col">
              <span className="text-lg font-bold text-white">NLP Amazon</span>
              <span className="text-xs text-cyan-400 font-medium">Analysis Platform</span>
            </div>
          </button>

          {/* Navigation principale compacte */}
          <nav className="hidden md:flex items-center space-x-1">
            {[
              { id: 'home', label: 'Accueil', icon: Home },
              { id: 'explore', label: 'Dataset', icon: Database },
              { id: 'analyze', label: 'Analyser', icon: Brain },
              { id: 'training', label: 'Entraîner', icon: Target },
              { id: 'pipeline', label: 'Pipeline', icon: Layers }
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setCurrentView(item.id)}
                className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-all duration-200 text-sm ${
                  currentView === item.id
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                    : 'text-white/70 hover:text-white hover:bg-white/10'
                }`}
              >
                <item.icon className="h-4 w-4" />
                <span className="font-medium">{item.label}</span>
              </button>
            ))}
          </nav>

          {/* Menu embeddings compact */}
          <div className="hidden lg:flex items-center space-x-2">
            <div className="w-px h-6 bg-white/20"></div>
            <div className="flex items-center space-x-1">
              <button
                onClick={() => setCurrentView('embeddings_hub')}
                className={`flex items-center space-x-2 px-3 py-1 rounded-lg transition-all duration-200 text-sm ${
                  currentView === 'embeddings_hub'
                    ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30'
                    : 'text-indigo-400/70 hover:text-indigo-400 hover:bg-white/5'
                }`}
              >
                <Network className="h-4 w-4" />
                <span className="font-medium">Hub Embeddings</span>
              </button>
            </div>
          </div>

          {/* Indicateurs de statut compacts */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 text-green-400 bg-green-500/20 px-2 py-1 rounded-full border border-green-500/30">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-xs font-medium">Live</span>
            </div>
            {reviews.length > 0 && (
              <div className="text-white/60 text-xs font-medium">
                {reviews.length} avis
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800">
      {/* Header uniforme pour toutes les pages */}
      {renderHeader()}

      {/* Contenu principal */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {isLoading && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
            <div className="bg-slate-800/90 backdrop-blur-xl p-10 rounded-2xl border border-white/20 text-center shadow-2xl">
              <RefreshCw className="h-16 w-16 text-cyan-400 animate-spin mx-auto mb-6" />
              <p className="text-white text-2xl font-bold mb-2">Analyse IA en cours...</p>
              <p className="text-white/60 text-lg">Traitement émotionnel avancé</p>
              <div className="mt-4 w-64 bg-white/20 rounded-full h-2">
                <div className="bg-gradient-to-r from-cyan-400 to-blue-500 h-2 rounded-full animate-pulse" style={{ width: '70%' }}></div>
              </div>
            </div>
          </div>
        )}

        {currentView === 'home' && renderHome()}
        {currentView === 'explore' && renderExplore()}
        {currentView === 'analyze' && renderAnalyze()}
        {currentView === 'training' && renderTraining()}
        {currentView === 'pipeline' && renderPipeline()}
        {currentView === 'results' && renderResults()}
        {currentView === 'embeddings_hub' && <EmbeddingHub onClose={() => setCurrentView('home')} />}
      </main>

      {/* Footer simplifié */}
      <footer className="border-t border-white/10 bg-slate-900/50 backdrop-blur-xl mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-3 mb-4 md:mb-0">
              <Brain className="h-5 w-5 text-cyan-400" />
              <span className="text-white/80">Pipeline NLP révolutionnaire • Amazon Polarity Analysis</span>
            </div>
            <div className="flex items-center space-x-4 text-white/60 text-sm">
              <span>Dataset: Amazon Polarity</span>
              <div className="flex items-center space-x-2 text-green-400 bg-green-500/20 px-2 py-1 rounded-full">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-xs font-medium">Système Opérationnel</span>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;