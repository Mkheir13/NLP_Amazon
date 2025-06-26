import React, { useState, useEffect } from 'react';
import { Brain, Play, Download, Trash2, Eye, BarChart3, Target, Settings, Cpu, Zap, Award, RefreshCw, CheckCircle, AlertCircle, Database, Activity, GitBranch, Server } from 'lucide-react';
import { BERTTrainingService, BERTTrainingConfig, BERTModel, NLTKResult } from '../services/BERTTrainingService';
import { Review } from '../services/DatasetLoader';

interface BERTTrainingProps {
  reviews: Review[];
}

export const BERTTraining: React.FC<BERTTrainingProps> = ({ reviews }) => {
  const [bertModels, setBertModels] = useState<BERTModel[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState('');
  const [selectedModel, setSelectedModel] = useState<BERTModel | null>(null);
  const [testText, setTestText] = useState('');
  const [nltkResult, setNltkResult] = useState<NLTKResult | null>(null);
  const [bertPrediction, setBertPrediction] = useState<any>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [config, setConfig] = useState<BERTTrainingConfig>({
    model_name: 'distilbert-base-uncased',
    epochs: 3,
    batch_size: 8,
    learning_rate: 2e-5,
    test_size: 0.2
  });

  // Vérifier le statut du backend au chargement
  useEffect(() => {
    checkBackendStatus();
    loadBertModels();
  }, []);

  const checkBackendStatus = async () => {
    setBackendStatus('checking');
    const isHealthy = await BERTTrainingService.checkBackendHealth();
    setBackendStatus(isHealthy ? 'online' : 'offline');
  };

  const loadBertModels = async () => {
    try {
      const models = await BERTTrainingService.getBERTModels();
      setBertModels(models);
    } catch (error) {
      console.error('Erreur chargement modèles:', error);
    }
  };

  const handleTrainBERT = async () => {
    if (reviews.length === 0) {
      alert('Aucune donnée disponible pour l\'entraînement');
      return;
    }

    if (backendStatus !== 'online') {
      alert('Backend Python non disponible. Démarrez le serveur Flask d\'abord.');
      return;
    }

    setIsTraining(true);
    setTrainingStatus('Initialisation...');

    try {
      const model = await BERTTrainingService.trainBERTModel(
        reviews,
        config,
        (status) => setTrainingStatus(status)
      );

      setBertModels(prev => [...prev, model]);
      setSelectedModel(model);
      setTrainingStatus('Modèle BERT entraîné avec succès !');
    } catch (error) {
      console.error('Erreur entraînement BERT:', error);
      setTrainingStatus(`Erreur: ${error instanceof Error ? error.message : 'Erreur inconnue'}`);
    } finally {
      setIsTraining(false);
    }
  };

  const handleAnalyzeText = async () => {
    if (!testText.trim()) return;

    setIsAnalyzing(true);
    setNltkResult(null);
    setBertPrediction(null);

    try {
      if (selectedModel) {
        // Analyse avec BERT et NLTK
        const comparison = await BERTTrainingService.compareAnalysis(testText, selectedModel.id);
        setNltkResult(comparison.nltk);
        setBertPrediction(comparison.bert);
        
        // Afficher un avertissement si BERT a échoué
        if (comparison.error) {
          console.warn('Erreur BERT:', comparison.error);
          setTrainingStatus(`NLTK analysé avec succès. BERT: ${comparison.error}`);
        }
      } else {
        // Analyse avec NLTK seulement
        const result = await BERTTrainingService.analyzeWithNLTK(testText);
        setNltkResult(result);
      }
    } catch (error) {
      console.error('Erreur analyse:', error);
      alert(`Erreur lors de l'analyse: ${error instanceof Error ? error.message : 'Erreur inconnue'}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'text-green-400 bg-green-500/20 border-green-500/30';
      case 'offline': return 'text-red-400 bg-red-500/20 border-red-500/30';
      default: return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online': return <CheckCircle className="h-5 w-5" />;
      case 'offline': return <AlertCircle className="h-5 w-5" />;
      default: return <RefreshCw className="h-5 w-5 animate-spin" />;
    }
  };

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-4xl font-bold text-white mb-4">Entraînement BERT & Analyse NLTK</h2>
        <p className="text-white/70 text-xl">Utilisez de vrais modèles transformers et NLTK pour l'analyse de sentiment</p>
      </div>

      {/* Statut du backend */}
      <div className="bg-slate-800/90 backdrop-blur-xl p-6 rounded-2xl border border-white/10 shadow-2xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Server className="h-8 w-8 text-cyan-400" />
            <div>
              <h3 className="text-white font-bold text-xl">Backend Python</h3>
              <p className="text-white/60">Serveur Flask pour BERT et NLTK</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-3 px-4 py-2 rounded-xl border ${getStatusColor(backendStatus)}`}>
              {getStatusIcon(backendStatus)}
              <span className="font-medium">
                {backendStatus === 'online' ? 'En ligne' : 
                 backendStatus === 'offline' ? 'Hors ligne' : 'Vérification...'}
              </span>
            </div>
            <button
              onClick={checkBackendStatus}
              className="px-4 py-2 bg-blue-500/20 text-blue-400 rounded-xl hover:bg-blue-500/30 transition-colors border border-blue-500/30"
            >
              <RefreshCw className="h-5 w-5" />
            </button>
          </div>
        </div>

        {backendStatus === 'offline' && (
          <div className="mt-4 p-4 bg-red-500/20 rounded-xl border border-red-500/30">
            <h4 className="text-red-400 font-bold mb-2">Backend non disponible</h4>
            <p className="text-white/70 text-sm mb-3">Pour utiliser BERT et NLTK, démarrez le backend Python :</p>
            <div className="bg-black/30 p-3 rounded-lg font-mono text-sm text-white">
              <div>cd backend</div>
              <div>pip install -r requirements.txt</div>
              <div>python app.py</div>
            </div>
          </div>
        )}
      </div>

      {/* Configuration BERT */}
      {backendStatus === 'online' && (
        <div className="bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-white/10 shadow-2xl">
          <h3 className="text-2xl font-bold text-white mb-6 flex items-center space-x-3">
            <Brain className="h-7 w-7 text-purple-400" />
            <span>Configuration BERT</span>
          </h3>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Modèle de base */}
            <div>
              <label className="block text-white font-medium mb-3">Modèle de base</label>
                             <select
                 value={config.model_name}
                 onChange={(e) => setConfig({...config, model_name: e.target.value as any})}
                 className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-purple-400"
                 disabled={isTraining}
                 style={{ color: 'white', backgroundColor: 'rgba(255,255,255,0.1)' }}
               >
                 <option value="distilbert-base-uncased" style={{ color: 'black', backgroundColor: 'white' }}>DistilBERT (Rapide)</option>
                 <option value="bert-base-uncased" style={{ color: 'black', backgroundColor: 'white' }}>BERT Base</option>
                 <option value="roberta-base" style={{ color: 'black', backgroundColor: 'white' }}>RoBERTa Base</option>
               </select>
            </div>

            {/* Epochs */}
            <div>
              <label className="block text-white font-medium mb-3">Epochs</label>
              <input
                type="number"
                min="1"
                max="10"
                value={config.epochs}
                onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
                className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-purple-400"
                disabled={isTraining}
              />
            </div>

            {/* Batch size */}
            <div>
              <label className="block text-white font-medium mb-3">Batch Size</label>
              <input
                type="number"
                min="4"
                max="32"
                step="4"
                value={config.batch_size}
                onChange={(e) => setConfig({...config, batch_size: parseInt(e.target.value)})}
                className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-purple-400"
                disabled={isTraining}
              />
            </div>

            {/* Learning rate */}
            <div>
              <label className="block text-white font-medium mb-3">Learning Rate</label>
                             <select
                 value={config.learning_rate}
                 onChange={(e) => setConfig({...config, learning_rate: parseFloat(e.target.value)})}
                 className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-purple-400"
                 disabled={isTraining}
                 style={{ color: 'white', backgroundColor: 'rgba(255,255,255,0.1)' }}
               >
                 <option value={1e-5} style={{ color: 'black', backgroundColor: 'white' }}>1e-5 (Conservateur)</option>
                 <option value={2e-5} style={{ color: 'black', backgroundColor: 'white' }}>2e-5 (Recommandé)</option>
                 <option value={3e-5} style={{ color: 'black', backgroundColor: 'white' }}>3e-5 (Agressif)</option>
                 <option value={5e-5} style={{ color: 'black', backgroundColor: 'white' }}>5e-5 (Très agressif)</option>
               </select>
            </div>

            {/* Test size */}
            <div>
              <label className="block text-white font-medium mb-3">Taille test (%)</label>
              <input
                type="number"
                min="10"
                max="50"
                value={config.test_size * 100}
                onChange={(e) => setConfig({...config, test_size: parseInt(e.target.value) / 100})}
                className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-purple-400"
                disabled={isTraining}
              />
            </div>
          </div>

          {/* Statistiques du dataset */}
          <div className="mt-6 p-4 bg-white/5 rounded-xl">
            <div className="flex items-center space-x-4 text-white/70">
              <div className="flex items-center space-x-2">
                <Database className="h-5 w-5" />
                <span>{reviews.length} avis disponibles</span>
              </div>
              <div className="flex items-center space-x-2">
                <Target className="h-5 w-5" />
                <span>{Math.floor(reviews.length * (1 - config.test_size))} pour l'entraînement</span>
              </div>
              <div className="flex items-center space-x-2">
                <Eye className="h-5 w-5" />
                <span>{Math.floor(reviews.length * config.test_size)} pour le test</span>
              </div>
            </div>
          </div>

          {/* Bouton d'entraînement */}
          <div className="mt-6">
            <button
              onClick={handleTrainBERT}
              disabled={isTraining || reviews.length === 0}
              className="w-full px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-3"
            >
              {isTraining ? (
                <>
                  <RefreshCw className="h-6 w-6 animate-spin" />
                  <span>Entraînement BERT en cours...</span>
                </>
              ) : (
                <>
                  <Brain className="h-6 w-6" />
                  <span>Entraîner BERT</span>
                </>
              )}
            </button>

            {/* Statut d'entraînement */}
            {isTraining && (
              <div className="mt-4 p-4 bg-purple-500/20 rounded-xl border border-purple-500/30">
                <div className="text-purple-400 font-medium">{trainingStatus}</div>
                <div className="text-white/60 text-sm mt-1">
                  ⚠️ L'entraînement peut prendre plusieurs minutes selon votre matériel
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Modèles BERT entraînés */}
      <div className="bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-white/10 shadow-2xl">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-2xl font-bold text-white flex items-center space-x-3">
            <Cpu className="h-7 w-7 text-orange-400" />
            <span>Modèles BERT ({bertModels.length})</span>
          </h3>
          <button
            onClick={loadBertModels}
            className="px-4 py-2 bg-blue-500/20 text-blue-400 rounded-xl hover:bg-blue-500/30 transition-colors border border-blue-500/30"
          >
            <RefreshCw className="h-5 w-5" />
          </button>
        </div>

        {bertModels.length === 0 ? (
          <div className="text-center py-12 text-white/60">
            <Brain className="h-16 w-16 mx-auto mb-4 opacity-50" />
            <p className="text-lg">Aucun modèle BERT entraîné</p>
            <p className="text-sm">Entraînez votre premier modèle BERT ci-dessus</p>
          </div>
        ) : (
          <div className="grid gap-4">
            {bertModels.map((model) => (
              <div
                key={model.id}
                className={`p-6 rounded-xl border transition-all hover:scale-[1.02] cursor-pointer ${
                  selectedModel?.id === model.id
                    ? 'bg-purple-500/20 border-purple-500/50'
                    : 'bg-white/5 border-white/10 hover:border-white/20'
                }`}
                onClick={() => setSelectedModel(model)}
              >
                <div className="flex justify-between items-start mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 rounded-lg bg-purple-500/20 border border-purple-500/30">
                      <Brain className="h-5 w-5 text-purple-400" />
                    </div>
                    <div>
                      <h4 className="text-white font-bold text-lg">{model.name}</h4>
                      <p className="text-white/60 text-sm">
                        {model.model_name} • {new Date(model.created_at).toLocaleDateString()}
                      </p>
                      <p className="text-white/40 text-xs font-mono">ID: {model.id}</p>
                    </div>
                  </div>
                  {selectedModel?.id === model.id && (
                    <div className="px-3 py-1 bg-purple-500/30 text-purple-300 rounded-lg text-sm font-medium">
                      Sélectionné
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-400">
                      {(model.metrics.accuracy * 100).toFixed(1)}%
                    </div>
                    <div className="text-white/60 text-sm">Accuracy</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-400">
                      {(model.metrics.f1_score * 100).toFixed(1)}%
                    </div>
                    <div className="text-white/60 text-sm">F1-Score</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-400">
                      {model.trained_on}
                    </div>
                    <div className="text-white/60 text-sm">Exemples</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-400">
                      {model.config.epochs}
                    </div>
                    <div className="text-white/60 text-sm">Epochs</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Test et comparaison */}
      <div className="bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-white/10 shadow-2xl">
        <h3 className="text-2xl font-bold text-white mb-6 flex items-center space-x-3">
          <GitBranch className="h-7 w-7 text-cyan-400" />
          <span>Test et Comparaison NLTK vs BERT</span>
        </h3>

        <div className="space-y-6">
          <div>
            <label className="block text-white font-medium mb-3">Texte à analyser</label>
            <textarea
              value={testText}
              onChange={(e) => setTestText(e.target.value)}
              placeholder="Entrez un texte pour comparer NLTK et BERT..."
              className="w-full h-32 p-4 bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/50 focus:outline-none focus:border-cyan-400 resize-none"
            />
          </div>

          <button
            onClick={handleAnalyzeText}
            disabled={!testText.trim() || isAnalyzing || backendStatus !== 'online'}
            className="px-8 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold transition-all hover:scale-105 disabled:opacity-50 flex items-center space-x-2"
          >
            {isAnalyzing ? (
              <>
                <RefreshCw className="h-5 w-5 animate-spin" />
                <span>Analyse en cours...</span>
              </>
            ) : (
              <>
                <Activity className="h-5 w-5" />
                <span>Analyser avec {selectedModel ? 'NLTK + BERT' : 'NLTK'}</span>
              </>
            )}
          </button>

          {/* Avertissement si erreur BERT */}
          {trainingStatus.includes('BERT:') && (
            <div className="p-4 bg-yellow-500/20 rounded-xl border border-yellow-500/30">
              <div className="flex items-center space-x-3">
                <AlertCircle className="h-5 w-5 text-yellow-400" />
                <div>
                  <h4 className="text-yellow-400 font-bold">Avertissement</h4>
                  <p className="text-white/70 text-sm">{trainingStatus}</p>
                  <p className="text-white/60 text-xs mt-1">
                    Le modèle BERT sélectionné n'est peut-être plus disponible. Essayez d'en entraîner un nouveau.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Résultats */}
          {(nltkResult || bertPrediction) && (
            <div className="grid md:grid-cols-2 gap-6">
              {/* Résultat NLTK */}
              {nltkResult && (
                <div className="p-6 bg-green-500/20 rounded-xl border border-green-500/30">
                  <h4 className="text-green-400 font-bold text-lg mb-4 flex items-center space-x-2">
                    <Zap className="h-5 w-5" />
                    <span>NLTK VADER</span>
                  </h4>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-white/70">Sentiment:</span>
                      <span className="text-green-400 font-bold capitalize">{nltkResult.sentiment}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white/70">Confiance:</span>
                      <span className="text-white font-bold">{(nltkResult.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white/70">Score composé:</span>
                      <span className="text-white font-bold">{nltkResult.polarity.toFixed(3)}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Résultat BERT */}
              {bertPrediction && (
                <div className="p-6 bg-purple-500/20 rounded-xl border border-purple-500/30">
                  <h4 className="text-purple-400 font-bold text-lg mb-4 flex items-center space-x-2">
                    <Brain className="h-5 w-5" />
                    <span>BERT ({selectedModel?.model_name})</span>
                  </h4>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-white/70">Sentiment:</span>
                      <span className="text-purple-400 font-bold capitalize">{bertPrediction.sentiment}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white/70">Confiance:</span>
                      <span className="text-white font-bold">{(bertPrediction.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white/70">Classe:</span>
                      <span className="text-white font-bold">{bertPrediction.class}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}; 