import React, { useState, useEffect } from 'react';
import { 
  Brain, Database, BarChart3, Search, Play, ArrowRight, Github, Sparkles, Zap, 
  ChevronRight, Home, RefreshCw, Target, TrendingUp, Award, Rocket, Globe, Shield, 
  CheckCircle, AlertCircle, Eye, Hash, Clock, FileText, Activity, Download, Copy, 
  Star, ThumbsUp, ThumbsDown, Filter, Shuffle, Settings, Layers, Heart, Minus, 
  Network, Code2, BookOpen, Lightbulb, GraduationCap, Beaker, Microscope,
  PieChart, LineChart, Radar, Map, Compass, Route, Flag, Trophy, Medal,
  Menu, X, ChevronDown, ChevronUp, Folder, FolderOpen, Terminal, Cpu, Zap as Lightning,
  GitBranch, Package, Wrench, Monitor, Gauge, BarChart, TrendingUp as Trending, Cloud,
  Info
} from 'lucide-react';

import { NLPPipeline } from './components/NLPPipeline';
import { BERTTraining } from './components/BERTTraining';
import RNNTraining from './components/RNNTraining';
import { EmbeddingHub } from './components/EmbeddingHub';
import { EmbeddingVisualizer } from './components/EmbeddingVisualizer';
import { SemanticSearch } from './components/SemanticSearch';
import { EmbeddingTraining } from './components/EmbeddingTraining';
import { EmbeddingTrainingSimple } from './components/EmbeddingTrainingSimple';
import AutoencoderTraining from './components/AutoencoderTraining';
import SimpleAutoencoder from './components/SimpleAutoencoder';
import CodeViewer from './components/CodeViewer';
import InfoPopup from './components/InfoPopup';

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



// Interface principale refontée
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
  const [trainingTab, setTrainingTab] = useState('bert');
  const [showCodeViewer, setShowCodeViewer] = useState(false);
  const [codeViewerStep, setCodeViewerStep] = useState('');
  
  // État pour la sidebar
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [expandedSections, setExpandedSections] = useState<string[]>(['exploration', 'analysis']);
  
  // État pour le popup d'information
  const [showInfoPopup, setShowInfoPopup] = useState(false);
  const [infoStepId, setInfoStepId] = useState('');



  // Données du dataset Amazon
  const [reviews, setReviews] = useState<Review[]>([]);
  const [filteredReviews, setFilteredReviews] = useState<Review[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [sentimentFilter, setSentimentFilter] = useState('all');
  const [isLoadingDataset, setIsLoadingDataset] = useState(false);

  // Configuration de la sidebar avec fonctionnalités disponibles uniquement (SANS DOUBLONS)
  const sidebarSections = [
    {
      id: 'exploration',
      title: 'Exploration de Données',
      icon: Database,
      color: 'cyan',
      items: [
        { id: 'home', title: 'Accueil & Guide', icon: Home, description: 'Vue d\'ensemble du projet' },
        { id: 'explore', title: 'Dataset Amazon', icon: Database, description: 'Explorer les 1000+ avis' }
      ]
    },
    {
      id: 'analysis',
      title: 'Analyse & Sentiment',
      icon: Brain,
      color: 'purple',
      items: [
        { id: 'analyze', title: 'Analyseur NLTK+BERT', icon: Brain, description: 'Comparaison VADER vs BERT' },
        { id: 'pipeline', title: 'Pipeline Complet', icon: Layers, description: 'Workflow NLP intégré' },
        { id: 'results', title: 'Résultats Détaillés', icon: BarChart, description: 'Visualisation avancée' }
      ]
    },
    {
      id: 'training',
      title: 'Entraînement Modèles',
      icon: Target,
      color: 'orange',
      items: [
        { id: 'training', title: 'Hub Entraînement', icon: Target, description: 'BERT, RNN, Autoencoder' },
        { id: 'simple_autoencoder', title: 'Autoencoder Simple', icon: Package, description: 'Version simplifiée' }
      ]
    },
    {
      id: 'embeddings',
      title: 'Embeddings & Vectorisation',
      icon: Network,
      color: 'emerald',
      items: [
        { id: 'embedding_training', title: 'TF-IDF Training', icon: Wrench, description: 'Entraîner embeddings' },
        { id: 'embedding_visualizer', title: 'Visualisation 2D/3D', icon: Monitor, description: 'Explorer l\'espace vectoriel' }
      ]
    },
    {
      id: 'search',
      title: 'Recherche & Similarité',
      icon: Search,
      color: 'blue',
      items: [
        { id: 'semantic_search', title: 'Recherche Sémantique', icon: Search, description: 'Similarité vectorielle' }
      ]
    },
    {
      id: 'development',
      title: 'Code & API',
      icon: Code2,
      color: 'slate',
      items: [
        { id: 'code_viewer', title: 'Code Source', icon: FileText, description: 'Comprendre l\'implémentation' }
      ]
    }
      ];

  // Fonction pour basculer l'expansion des sections
  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev => 
      prev.includes(sectionId) 
        ? prev.filter(id => id !== sectionId)
        : [...prev, sectionId]
    );
  };



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

  const navigateWithData = (view: string, data?: any) => {
    setCurrentView(view);
  };

  // Fonction pour ouvrir le popup d'information
  const openInfoPopup = (stepId: string) => {
    setInfoStepId(stepId);
    setShowInfoPopup(true);
  };

  // Fonction pour fermer le popup d'information
  const closeInfoPopup = () => {
    setShowInfoPopup(false);
    setInfoStepId('');
  };

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

  // Rendu de la page d'accueil refontée
  const renderHome = () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-gray-900 to-black">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-400/10 via-blue-500/10 to-purple-600/10 animate-pulse"></div>
        
        {/* Header Hero */}
        <div className="relative pt-16 pb-12 text-center">
          <div className="flex items-center justify-center space-x-4 mb-8">
            <div className="relative">
              <Brain className="h-20 w-20 text-cyan-400 animate-bounce" />
              <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-400 rounded-full animate-ping"></div>
            </div>
            <div className="text-left">
              <div className="flex items-center space-x-2 mb-2">
                <span className="px-3 py-1 bg-cyan-500/20 text-cyan-400 rounded-full text-sm font-medium border border-cyan-500/30">
                  v3.0 • Refonte UX
                </span>
                <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm font-medium border border-green-500/30">
                  Navigation Optimisée
                </span>
              </div>
              <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                Analyse NLP Amazon
              </h1>
              <p className="text-gray-300 text-lg mt-2">
                Explorez les techniques NLP appliquées aux avis Amazon avec une interface moderne
              </p>
            </div>
          </div>

          {/* Stats du projet */}
          <div className="flex items-center justify-center space-x-8 mb-12">
            <div className="text-center">
              <div className="text-3xl font-bold text-cyan-400">{reviews.length}</div>
              <div className="text-gray-400 text-sm">Avis Amazon</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-400">{availableBERTModels.length}</div>
              <div className="text-gray-400 text-sm">Modèles BERT</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-400">30+</div>
              <div className="text-gray-400 text-sm">API Endpoints</div>
            </div>
          </div>

          {/* Description du projet */}
          <div className="max-w-4xl mx-auto text-center mb-12">
            <p className="text-gray-300 text-lg leading-relaxed">
              Ce projet démontre l'application de techniques NLP modernes sur le dataset Amazon Polarity. 
              Explorez les différentes approches d'analyse de sentiment, de l'entraînement de modèles deep learning 
              à la visualisation de données, en passant par la recherche sémantique et les embeddings.
            </p>
          </div>

          {/* Actions principales */}
          <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
            <button
              onClick={() => setCurrentView('explore')}
              className="group px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 flex items-center space-x-3 shadow-lg shadow-cyan-500/25"
            >
              <Database className="h-5 w-5 group-hover:animate-pulse" />
              <span>Explorer les Données</span>
              <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </button>
            
            <button
              onClick={() => setCurrentView('analyze')}
              className="group px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 flex items-center space-x-3 shadow-lg shadow-purple-500/25"
            >
              <Brain className="h-5 w-5 group-hover:animate-pulse" />
              <span>Analyser Sentiment</span>
              <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </button>
            
            <button
              onClick={() => setCurrentView('training')}
              className="group px-8 py-4 bg-gradient-to-r from-orange-500 to-red-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 flex items-center space-x-3 shadow-lg shadow-orange-500/25"
            >
              <Target className="h-5 w-5 group-hover:animate-pulse" />
              <span>Entraîner Modèles</span>
              <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>
        </div>

        {/* Fonctionnalités en grille */}
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
          <h2 className="text-3xl font-bold text-center text-white mb-12">Fonctionnalités Disponibles</h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {sidebarSections.map((section) => (
              <div key={section.id} className={`bg-gradient-to-br from-${section.color}-500/20 to-${section.color}-600/10 backdrop-blur-sm rounded-2xl p-6 border border-${section.color}-500/30 hover:scale-105 transition-all duration-300 group`}>
                <div className="flex items-center space-x-4 mb-4">
                  {section.icon && (
                    <div className={`p-3 rounded-xl bg-${section.color}-500/20 group-hover:scale-110 transition-transform`}>
                      <section.icon className={`h-8 w-8 text-${section.color}-400`} />
                    </div>
                  )}
                  <h3 className="text-xl font-semibold text-white">{section.title}</h3>
                </div>
                
                <div className="space-y-2">
                  {section.items.map((item) => (
                    <div
                      key={item.id}
                      onClick={() => setCurrentView(item.id)}
                      className="w-full text-left p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors border border-transparent hover:border-white/20 group/item cursor-pointer"
                    >
                      <div className="flex items-center space-x-3">
                        <item.icon className="h-4 w-4 text-white/60 group-hover/item:text-white/80" />
                        <div>
                          <div className="text-sm font-medium text-white/80 group-hover/item:text-white">{item.title}</div>
                          <div className="text-xs text-white/50">{item.description}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

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
        
        // Marquer l'analyse comme terminée
        console.log('Analyse terminée:', { analysisResults: legacyResults, realNLPResults: nlpResults });
        
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
    
    // Marquer l'exploration du dataset comme terminée
    console.log('Dataset exploré:', { selectedReview: review });
  };

  // Avis aléatoire
  const getRandomReview = () => {
    if (filteredReviews.length === 0) return;
    const randomIndex = Math.floor(Math.random() * filteredReviews.length);
    const review = filteredReviews[randomIndex];
    selectReview(review);
  };



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

  // Rendu de l'entraînement de modèles avec onglets
  const renderTraining = () => (
    <div className="space-y-6">
      {/* Header avec onglets */}
      <div className="bg-slate-800/90 backdrop-blur-xl rounded-xl p-6 border border-slate-600/30">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-3xl font-bold text-white mb-2">Entraînement de Modèles</h2>
            <p className="text-slate-300">Choisissez le type de modèle à entraîner</p>
          </div>
        </div>

        {/* Onglets de navigation */}
        <div className="flex space-x-2 mb-6">
          <button
            onClick={() => setTrainingTab('bert')}
            className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
              trainingTab === 'bert'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            🤖 BERT Training
          </button>
          <button
            onClick={() => setTrainingTab('rnn')}
            className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
              trainingTab === 'rnn'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            🧠 RNN from Scratch
          </button>
          <button
            onClick={() => setTrainingTab('autoencoder')}
            className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
              trainingTab === 'autoencoder'
                ? 'bg-green-600 text-white shadow-lg'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            🔄 Autoencoder Avancé
          </button>
        </div>

        {/* Description de l'onglet actuel */}
        <div className="bg-slate-700/50 rounded-lg p-4">
          {trainingTab === 'bert' && (
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
              <p className="text-slate-200">
                <strong>BERT Training:</strong> Entraînez des modèles BERT pour la classification de sentiments avec fine-tuning sur le dataset Amazon.
              </p>
            </div>
          )}
          {trainingTab === 'rnn' && (
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-purple-400 rounded-full"></div>
              <p className="text-slate-200">
                <strong>RNN from Scratch:</strong> Implémentation complète d'un RNN from scratch avec PyTorch.
              </p>
            </div>
          )}
          {trainingTab === 'autoencoder' && (
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-400 rounded-full"></div>
              <p className="text-slate-200">
                <strong>Autoencoder Avancé:</strong> Entraînement d'autoencodeurs avec régularisation, dropout et techniques avancées.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Contenu de l'onglet */}
      <div>
        {trainingTab === 'bert' && <BERTTraining reviews={reviews} />}
        {trainingTab === 'rnn' && <RNNTraining isVisible={true} />}
        {trainingTab === 'autoencoder' && <AutoencoderTraining />}
      </div>
    </div>
  );

  // Rendu des résultats avec design amélioré (NLTK + BERT)
  const renderResults = () => {
    if (!analysisResults && !realNLPResults) return null;

    // Utiliser les résultats réels si disponibles, sinon fallback
    const results = analysisResults;
    const nlpResults = realNLPResults;
    
    if (!results) return null;

    // Déterminer quelle source utiliser pour l'affichage
    const displaySentiment = nlpResults?.comparison?.finalSentiment || nlpResults?.nltk?.sentiment || results.sentiment.label;
    const displayConfidence = nlpResults?.comparison?.nltkConfidence || nlpResults?.nltk?.confidence || results.sentiment.confidence;
    const displayPolarity = nlpResults?.nltk?.polarity || results.sentiment.polarity;

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
                  {getSentimentEmoji(displaySentiment, displayPolarity)}
                </div>
                <div className={`text-4xl font-bold ${getSentimentColor(displaySentiment)} mb-2`}>
                  {displaySentiment.toUpperCase()}
                </div>
                <div className="text-white/60 text-lg">
                  {nlpResults?.comparison ? 'Sentiment final (NLTK+BERT)' : 'Sentiment détecté'}
                </div>
              </div>
            
            <div className="text-center">
              <div className="text-6xl font-bold text-white mb-2">
                {(displayConfidence * 100).toFixed(0)}%
              </div>
              <div className="text-white/60 text-lg">
                Confiance {nlpResults ? '(NLTK)' : ''}
              </div>
              <div className="mt-2 w-full bg-white/20 rounded-full h-3">
                <div 
                  className="bg-gradient-to-r from-cyan-400 to-blue-500 h-3 rounded-full transition-all duration-1000"
                  style={{ width: `${displayConfidence * 100}%` }}
                />
              </div>
            </div>
            
            <div className="text-center">
              <div className={`text-6xl font-bold ${getSentimentColor(displaySentiment)} mb-2`}>
                {displayPolarity.toFixed(2)}
              </div>
              <div className="text-white/60 text-lg">Score de polarité</div>
              <div className="text-white/50 text-sm mt-1">
                {displayPolarity > 0 ? 'Positif' : displayPolarity < 0 ? 'Négatif' : 'Neutre'}
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

        {/* Résultats détaillés NLTK + BERT */}
        {nlpResults && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Résultats NLTK */}
            <div className="bg-blue-500/20 border border-blue-500/30 rounded-xl p-6">
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
                <h3 className="text-blue-400 font-bold text-lg">Analyse NLTK VADER</h3>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-white/70">Sentiment:</span>
                  <span className={`font-bold ${getSentimentColor(nlpResults.nltk.sentiment)}`}>
                    {nlpResults.nltk.sentiment}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/70">Confiance:</span>
                  <span className="text-white font-bold">{(nlpResults.nltk.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/70">Polarité:</span>
                  <span className="text-white font-bold">{nlpResults.nltk.polarity.toFixed(3)}</span>
                </div>
                {nlpResults.nltk.scores && (
                  <div className="mt-4 space-y-2">
                    <div className="text-white/60 text-sm font-medium">Scores détaillés:</div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="flex justify-between">
                        <span className="text-green-400">Positif:</span>
                        <span className="text-white">{nlpResults.nltk.scores.pos.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-red-400">Négatif:</span>
                        <span className="text-white">{nlpResults.nltk.scores.neg.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-yellow-400">Neutre:</span>
                        <span className="text-white">{nlpResults.nltk.scores.neu.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-purple-400">Composé:</span>
                        <span className="text-white">{nlpResults.nltk.scores.compound.toFixed(3)}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Résultats BERT */}
            {nlpResults.bert ? (
              <div className="bg-purple-500/20 border border-purple-500/30 rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-3 h-3 bg-purple-400 rounded-full"></div>
                  <h3 className="text-purple-400 font-bold text-lg">Analyse BERT</h3>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-white/70">Sentiment:</span>
                    <span className={`font-bold ${getSentimentColor(nlpResults.bert.sentiment)}`}>
                      {nlpResults.bert.sentiment}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/70">Confiance:</span>
                    <span className="text-white font-bold">{(nlpResults.bert.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/70">Classe:</span>
                    <span className="text-white font-bold">{nlpResults.bert.class}</span>
                  </div>
                </div>

                {/* Comparaison */}
                {nlpResults.comparison && (
                  <div className="mt-4 p-3 bg-white/10 rounded-lg">
                    <div className="text-white/60 text-sm font-medium mb-2">Comparaison:</div>
                    <div className="space-y-1 text-xs">
                      <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${nlpResults.comparison.agreement ? 'bg-green-400' : 'bg-yellow-400'}`}></div>
                        <span className="text-white/80">
                          {nlpResults.comparison.agreement ? 'Accord' : 'Désaccord'} NLTK-BERT
                        </span>
                      </div>
                      <div className="text-white/60 mt-2">
                        {nlpResults.comparison.reasoning}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-gray-500/20 border border-gray-500/30 rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
                  <h3 className="text-gray-400 font-bold text-lg">BERT Non Disponible</h3>
                </div>
                <p className="text-white/60 text-sm">
                  Aucun modèle BERT n'est actuellement chargé. L'analyse se base uniquement sur NLTK VADER.
                </p>
                <button
                  onClick={() => setCurrentView('training')}
                  className="mt-3 px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg text-sm hover:bg-purple-500/30 transition-colors"
                >
                  Entraîner un modèle BERT
                </button>
              </div>
            )}
          </div>
        )}

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

  // Composant Sidebar avec navigation hiérarchique
  const renderSidebar = () => (
    <>
      {/* Overlay pour mobile */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      
      {/* Sidebar redimensionnable */}
      <div 
        className={`
          fixed top-0 left-0 h-full bg-gradient-to-b from-slate-900/98 via-slate-900/95 to-slate-800/98 backdrop-blur-xl border-r border-cyan-500/20 z-50 transform transition-all duration-500 overflow-hidden shadow-2xl shadow-cyan-500/10 flex flex-col
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
        style={{ width: '320px' }}
      >
        {/* Gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 via-transparent to-purple-500/5 pointer-events-none"></div>
        
        {/* Header sidebar */}
        <div className="relative p-6 border-b border-cyan-500/20 bg-gradient-to-r from-cyan-500/10 to-purple-500/10">
          <div className="flex items-center justify-between">
            <div className="text-center flex-1">
              <h2 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                NLP Platform
              </h2>
              <p className="text-sm text-cyan-300/80 font-medium">Navigation Complète</p>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden p-2 hover:bg-cyan-500/20 rounded-xl transition-all duration-300 border border-transparent hover:border-cyan-500/30"
            >
              <X className="h-5 w-5 text-cyan-300 hover:text-white transition-colors" />
            </button>
          </div>
        </div>

        {/* Navigation avec scrollbar stylisée */}
        <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-track-slate-800/50 scrollbar-thumb-cyan-500/30 hover:scrollbar-thumb-cyan-500/50 scrollbar-thumb-rounded-full scrollbar-track-rounded-full">
          <div className="p-4 space-y-2">
            {sidebarSections.map((section) => {
                const SectionIcon = section.icon;
                const isExpanded = expandedSections.includes(section.id);
                
                return (
                  <div key={section.id} className="space-y-1">
                    {/* Section Header */}
                    <button
                      onClick={() => toggleSection(section.id)}
                      className={`
                        w-full flex items-center justify-between p-4 rounded-xl transition-all duration-300 group relative overflow-hidden
                        ${isExpanded 
                          ? `bg-gradient-to-r from-${section.color}-500/20 to-${section.color}-400/10 border border-${section.color}-500/40 shadow-lg shadow-${section.color}-500/20` 
                          : 'hover:bg-gradient-to-r hover:from-white/5 hover:to-white/10 border border-transparent hover:border-white/20'
                        }
                      `}
                    >
                      {/* Effet de brillance */}
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
                      
                      <div className="relative flex items-center space-x-3">
                        {SectionIcon && (
                          <div className={`p-2 rounded-lg bg-${section.color}-500/20 group-hover:scale-110 transition-transform duration-300`}>
                            <SectionIcon className={`h-5 w-5 text-${section.color}-400 group-hover:text-${section.color}-300 transition-colors`} />
                          </div>
                        )}
                        <span className="text-white font-semibold text-sm group-hover:text-white/90 transition-colors">{section.title}</span>
                      </div>
                      <div className="relative">
                        {isExpanded ? (
                          <ChevronUp className={`h-4 w-4 text-${section.color}-400 group-hover:rotate-180 transition-all duration-300`} />
                        ) : (
                          <ChevronDown className="h-4 w-4 text-white/60 group-hover:text-white/80 transition-colors" />
                        )}
                      </div>
                    </button>

                    {/* Section Items */}
                    {isExpanded && (
                      <div className="ml-6 space-y-2 border-l-2 border-gradient-to-b from-cyan-500/30 to-purple-500/30 pl-4 relative">
                        {/* Ligne de connexion animée */}
                        <div className={`absolute left-0 top-0 w-0.5 h-full bg-gradient-to-b from-${section.color}-500/50 to-${section.color}-400/20 rounded-full`}></div>
                        
                        {section.items.map((item, itemIndex) => {
                          const ItemIcon = item.icon;
                          const isActive = currentView === item.id;
                          
                          return (
                            <button
                              key={item.id}
                              onClick={() => {
                                setCurrentView(item.id);
                                setSidebarOpen(false); // Fermer sur mobile
                              }}
                              className={`
                                w-full flex items-center space-x-3 p-3 rounded-xl transition-all duration-300 text-left group relative overflow-hidden
                                ${isActive 
                                  ? `bg-gradient-to-r from-${section.color}-500/25 to-${section.color}-400/15 border border-${section.color}-500/50 shadow-md shadow-${section.color}-500/20 scale-105` 
                                  : 'text-white/70 hover:text-white hover:bg-gradient-to-r hover:from-white/5 hover:to-white/10 hover:border-white/20 border border-transparent hover:scale-102'
                                }
                              `}
                              style={{
                                animationDelay: `${itemIndex * 100}ms`
                              }}
                            >
                              {/* Effet de brillance pour les éléments actifs */}
                              {isActive && (
                                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -skew-x-12 animate-pulse"></div>
                              )}
                              
                              <div className="relative flex items-center space-x-3 flex-1 min-w-0">
                                <div className={`p-2 rounded-lg transition-all duration-300 ${
                                  isActive 
                                    ? `bg-${section.color}-500/30 scale-110` 
                                    : 'bg-white/10 group-hover:bg-white/20 group-hover:scale-110'
                                }`}>
                                  <ItemIcon className={`h-4 w-4 transition-colors ${
                                    isActive 
                                      ? `text-${section.color}-300` 
                                      : 'text-white/60 group-hover:text-white/80'
                                  }`} />
                                </div>
                                <div className="flex-1 min-w-0">
                                  <div className={`text-sm font-semibold transition-colors ${
                                    isActive 
                                      ? `text-${section.color}-300` 
                                      : 'text-white/80 group-hover:text-white'
                                  }`}>
                                    {item.title}
                                  </div>
                                  <div className="text-xs text-white/50 truncate group-hover:text-white/60 transition-colors">
                                    {item.description}
                                  </div>
                                </div>
                              </div>
                              
                              {/* Bouton Info */}
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  openInfoPopup(item.id);
                                }}
                                className="p-2 rounded-lg hover:bg-white/10 transition-all duration-300 opacity-0 group-hover:opacity-100 hover:scale-110"
                                title="Plus d'informations"
                              >
                                <Info className="h-4 w-4 text-white/60 hover:text-blue-400 transition-colors" />
                              </button>
                              
                              {/* Indicateur d'activité */}
                              {isActive && (
                                <div className={`w-2 h-2 rounded-full bg-${section.color}-400 animate-pulse`}></div>
                              )}
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
          </div>
        </div>


      </div>
    </>
  );

  // Composant Header uniforme et compact
  const renderHeader = () => (
    <header className="lg:hidden bg-slate-900/95 backdrop-blur-xl border-b border-white/10 sticky top-0 z-40">
      <div className="px-4 sm:px-6">
        <div className="flex items-center justify-between h-14">
          {/* Menu burger pour mobile */}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
          >
            <Menu className="h-5 w-5 text-white/70" />
          </button>
          
          {/* Logo mobile */}
          <button 
            onClick={() => setCurrentView('home')}
            className="flex items-center space-x-2 hover:opacity-80 transition-opacity"
          >
            <div className="relative">
              <Brain className="h-6 w-6 text-cyan-400 animate-pulse" />
              <div className="absolute inset-0 bg-cyan-400 rounded-full blur-md opacity-30"></div>
            </div>
              <span className="text-lg font-bold text-white">NLP Amazon</span>
          </button>

          {/* Statut live */}
          <div className="flex items-center space-x-2 text-green-400 bg-green-500/20 px-2 py-1 rounded-full border border-green-500/30">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-xs font-medium">Live</span>
          </div>
        </div>
      </div>
    </header>
  );

  // Rendu de la vue Code & API Explorer
  const renderCodeExplorer = () => (
    <div className="space-y-8">
      <div className="text-center mb-12">
        <div className="flex items-center justify-center space-x-4 mb-6">
          <Code2 className="h-16 w-16 text-slate-400 animate-pulse" />
          <div className="text-left">
            <h1 className="text-4xl font-bold text-white mb-2">Code & API Explorer</h1>
            <p className="text-slate-300 text-lg">Explorez le code source et testez les 30+ endpoints API</p>
          </div>
        </div>
        
        <div className="flex items-center justify-center space-x-6 mt-6">
          <div className="flex items-center space-x-2 text-slate-400 bg-slate-500/20 px-4 py-2 rounded-full border border-slate-500/30">
            <FileText className="h-4 w-4" />
            <span className="font-medium">Code Source</span>
          </div>
          <div className="flex items-center space-x-2 text-blue-400 bg-blue-500/20 px-4 py-2 rounded-full border border-blue-500/30">
            <Globe className="h-4 w-4" />
            <span className="font-medium">API REST</span>
          </div>
          <div className="flex items-center space-x-2 text-green-400 bg-green-500/20 px-4 py-2 rounded-full border border-green-500/30">
            <CheckCircle className="h-4 w-4" />
            <span className="font-medium">Tests Interactifs</span>
          </div>
        </div>
      </div>

      {/* Sélecteur de module */}
      <div className="bg-slate-800/90 backdrop-blur-xl rounded-2xl p-6 border border-slate-600/30">
        <h3 className="text-xl font-semibold text-white mb-4">Sélectionner un Module</h3>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { id: 'rnn', title: 'RNN from Scratch', icon: Brain, color: 'purple', desc: 'PyTorch + TensorFlow' },
            { id: 'bert', title: 'BERT Training', icon: Target, color: 'blue', desc: 'Hugging Face' },
            { id: 'autoencoder', title: 'Autoencoder', icon: Layers, color: 'green', desc: 'Compression' },
            { id: 'embeddings', title: 'Embeddings', icon: Network, color: 'cyan', desc: 'TF-IDF + Search' }
          ].map((module) => (
              <button
              key={module.id}
              onClick={() => {
                setCodeViewerStep(module.id);
                setShowCodeViewer(true);
              }}
              className={`group p-4 rounded-xl border transition-all duration-300 hover:scale-105 bg-${module.color}-500/10 border-${module.color}-500/30 hover:bg-${module.color}-500/20`}
            >
              <module.icon className={`h-8 w-8 text-${module.color}-400 mx-auto mb-3 group-hover:scale-110 transition-transform`} />
              <h4 className="font-semibold text-white mb-1">{module.title}</h4>
              <p className="text-sm text-gray-400">{module.desc}</p>
              <div className="mt-3 flex items-center justify-center space-x-2 text-xs">
                <Code2 className="h-3 w-3" />
                <span className={`text-${module.color}-400`}>Voir le code</span>
              </div>
              </button>
            ))}
            </div>
          </div>

      {/* API Endpoints */}
      <div className="bg-slate-800/90 backdrop-blur-xl rounded-2xl p-6 border border-slate-600/30">
        <h3 className="text-xl font-semibold text-white mb-6">API Endpoints Disponibles</h3>
        
        <div className="grid gap-4">
          {/* Endpoints par catégorie */}
          {[
            {
              category: 'BERT Training',
              color: 'blue',
              endpoints: [
                { method: 'POST', path: '/api/train/bert', desc: 'Entraîner un modèle BERT' },
                { method: 'GET', path: '/api/models', desc: 'Liste des modèles disponibles' },
                { method: 'POST', path: '/api/predict/bert/{id}', desc: 'Prédiction avec BERT' }
              ]
            },
            {
              category: 'RNN from Scratch',
              color: 'purple',
              endpoints: [
                { method: 'POST', path: '/api/rnn/train', desc: 'Entraîner RNN PyTorch' },
                { method: 'POST', path: '/api/rnn/predict', desc: 'Prédiction RNN' },
                { method: 'GET', path: '/api/rnn/info', desc: 'Informations du modèle' }
              ]
            },
            {
              category: 'Embeddings & TF-IDF',
              color: 'cyan',
              endpoints: [
                { method: 'POST', path: '/api/embeddings/train/tfidf', desc: 'Entraîner TF-IDF' },
                { method: 'POST', path: '/api/embeddings/search', desc: 'Recherche sémantique' },
                { method: 'POST', path: '/api/embeddings/visualize', desc: 'Visualisation 2D/3D' }
              ]
            },
            {
              category: 'Autoencoder',
              color: 'green',
              endpoints: [
                { method: 'POST', path: '/api/autoencoder/train', desc: 'Entraîner autoencoder' },
                { method: 'POST', path: '/api/autoencoder/encode', desc: 'Encoder des textes' },
                { method: 'POST', path: '/api/autoencoder/clustering', desc: 'Clustering avancé' }
              ]
            }
          ].map((category, idx) => (
            <div key={idx} className={`bg-${category.color}-500/10 border border-${category.color}-500/30 rounded-xl p-4`}>
              <h4 className={`font-semibold text-${category.color}-400 mb-3`}>{category.category}</h4>
              <div className="space-y-2">
                {category.endpoints.map((endpoint, endIdx) => (
                  <div key={endIdx} className="flex items-center justify-between bg-slate-700/50 rounded-lg p-3">
          <div className="flex items-center space-x-3">
                      <span className={`px-2 py-1 text-xs font-mono rounded bg-${category.color}-500/20 text-${category.color}-400`}>
                        {endpoint.method}
                      </span>
                      <code className="text-white font-mono text-sm">{endpoint.path}</code>
            </div>
                    <span className="text-gray-400 text-sm">{endpoint.desc}</span>
              </div>
                ))}
          </div>
        </div>
          ))}
      </div>
      </div>

      {/* Statistiques du projet */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          { number: "30+", label: "Endpoints API", icon: Globe, color: "text-blue-400", bg: "bg-blue-500/20" },
          { number: "2000+", label: "Lignes de Code", icon: Code2, color: "text-green-400", bg: "bg-green-500/20" },
          { number: "5", label: "Services ML", icon: Brain, color: "text-purple-400", bg: "bg-purple-500/20" },
          { number: "React", label: "Frontend", icon: Sparkles, color: "text-cyan-400", bg: "bg-cyan-500/20" }
        ].map((stat, index) => (
          <div key={index} className={`${stat.bg} backdrop-blur-sm rounded-2xl p-6 border border-white/20 hover:scale-105 transition-all duration-300 group relative overflow-hidden`}>
            <div className="relative text-center">
              <stat.icon className={`h-8 w-8 ${stat.color} mx-auto mb-3 group-hover:scale-110 transition-transform`} />
              <div className={`text-2xl font-bold ${stat.color} mb-1`}>{stat.number}</div>
              <div className="text-white/80 text-sm font-medium">{stat.label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Actions */}
      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        <button
          onClick={() => {
            setCodeViewerStep('rnn');
            setShowCodeViewer(true);
          }}
          className="px-8 py-4 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 flex items-center justify-center space-x-2"
        >
          <Code2 className="h-5 w-5" />
          <span>Explorer le Code RNN</span>
        </button>
        <button
          onClick={() => setCurrentView('home')}
          className="px-8 py-4 bg-white/10 text-white rounded-xl font-semibold hover:bg-white/20 transition-colors border border-white/20"
        >
          Retour au Guide
        </button>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800">
      {/* Sidebar */}
      {renderSidebar()}
      
      {/* Contenu principal avec offset pour sidebar redimensionnable */}
      <div 
        className="transition-all duration-300"
        style={{ marginLeft: sidebarOpen || window.innerWidth >= 1024 ? '320px' : '0' }}
      >
      {/* Header uniforme pour toutes les pages */}
      {renderHeader()}

      {/* Contenu principal */}
        <main className="px-4 sm:px-6 lg:px-8 py-12">

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
        {currentView === 'simple_autoencoder' && <SimpleAutoencoder />}
        {currentView === 'explore' && renderExplore()}
        {currentView === 'analyze' && renderAnalyze()}
        {currentView === 'training' && renderTraining()}
        {currentView === 'pipeline' && renderPipeline()}
        {currentView === 'results' && renderResults()}
        {currentView === 'embeddings_hub' && <EmbeddingHub onClose={() => setCurrentView('home')} />}
        {currentView === 'autoencoder_training' && <AutoencoderTraining />}
        {currentView === 'code' && renderCodeExplorer()}
        
        {/* Nouvelles vues pour toutes les fonctionnalités */}
        {currentView === 'bert_training' && <BERTTraining reviews={reviews} />}
        {currentView === 'rnn_training' && <RNNTraining />}
        {currentView === 'embedding_training' && <EmbeddingTraining />}
        {currentView === 'embedding_simple' && <EmbeddingTrainingSimple />}
        {currentView === 'embedding_visualizer' && <EmbeddingVisualizer />}
        {currentView === 'semantic_search' && <SemanticSearch />}
        {currentView === 'nlp_pipeline' && <NLPPipeline text={textToAnalyze || "Entrez votre texte ici..."} onComplete={handlePipelineComplete} />}
        {currentView === 'code_viewer' && showCodeViewer && <CodeViewer stepId={codeViewerStep} isVisible={true} />}

        {/* Modal CodeViewer */}
        {showCodeViewer && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-slate-900/95 backdrop-blur-xl rounded-2xl border border-slate-600/30 max-w-6xl w-full max-h-[90vh] overflow-hidden">
              <div className="flex items-center justify-between p-6 border-b border-slate-600/30">
                <h2 className="text-2xl font-bold text-white">Code Source - {codeViewerStep}</h2>
                <button
                  onClick={() => setShowCodeViewer(false)}
                  className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
                >
                  <AlertCircle className="h-6 w-6 text-slate-400" />
                </button>
              </div>
                             <div className="overflow-y-auto max-h-[calc(90vh-100px)]">
                 <CodeViewer stepId={codeViewerStep} isVisible={true} />
               </div>
            </div>
          </div>
        )}



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
      
      {/* Popup d'information */}
      <InfoPopup 
        isOpen={showInfoPopup}
        onClose={closeInfoPopup}
        stepId={infoStepId}
      />
    </div>
  );
}

export default App;