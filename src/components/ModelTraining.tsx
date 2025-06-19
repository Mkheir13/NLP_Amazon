import React, { useState, useEffect } from 'react';
import { Brain, Play, Download, Upload, Trash2, Eye, BarChart3, Clock, Target, TrendingUp, Settings, Cpu, Zap, Award, RefreshCw, CheckCircle, AlertCircle, FileText, Database } from 'lucide-react';
import { ModelTrainer, TrainingConfig, TrainedModel, TrainingMetrics } from '../services/ModelTrainer';
import { Review } from '../services/DatasetLoader';

interface ModelTrainingProps {
  reviews: Review[];
}

export const ModelTraining: React.FC<ModelTrainingProps> = ({ reviews }) => {
  const [trainedModels, setTrainedModels] = useState<TrainedModel[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingStatus, setTrainingStatus] = useState('');
  const [selectedModel, setSelectedModel] = useState<TrainedModel | null>(null);
  const [testText, setTestText] = useState('');
  const [prediction, setPrediction] = useState<{ prediction: number; confidence: number } | null>(null);

  const [config, setConfig] = useState<TrainingConfig>({
    modelType: 'logistic_regression',
    testSize: 0.2,
    maxFeatures: 1000,
    ngrams: [1, 2] as [number, number],
    epochs: 100,
    learningRate: 0.01
  });

  useEffect(() => {
    setTrainedModels(ModelTrainer.getModels());
  }, []);

  const handleTrainModel = async () => {
    if (reviews.length === 0) {
      alert('Aucune donnée disponible pour l\'entraînement');
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingStatus('Initialisation...');

    try {
      const model = await ModelTrainer.trainModel(
        reviews,
        config,
        (progress, status) => {
          setTrainingProgress(progress);
          setTrainingStatus(status);
        }
      );

      setTrainedModels(ModelTrainer.getModels());
      setSelectedModel(model);
      setTrainingStatus('Entraînement terminé avec succès !');
    } catch (error) {
      console.error('Erreur lors de l\'entraînement:', error);
      setTrainingStatus('Erreur lors de l\'entraînement');
    } finally {
      setIsTraining(false);
    }
  };

  const handleDeleteModel = (modelId: string) => {
    if (confirm('Êtes-vous sûr de vouloir supprimer ce modèle ?')) {
      ModelTrainer.deleteModel(modelId);
      setTrainedModels(ModelTrainer.getModels());
      if (selectedModel?.id === modelId) {
        setSelectedModel(null);
      }
    }
  };

  const handleTestModel = () => {
    if (!selectedModel || !testText.trim()) return;

    const result = ModelTrainer.predict(selectedModel, testText);
    setPrediction(result);
  };

  const handleLoadModel = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    ModelTrainer.loadModel(file)
      .then(model => {
        setTrainedModels(ModelTrainer.getModels());
        setSelectedModel(model);
      })
      .catch(error => {
        console.error('Erreur lors du chargement:', error);
        alert('Erreur lors du chargement du modèle');
      });
  };

  const getModelTypeIcon = (type: string) => {
    switch (type) {
      case 'naive_bayes': return <Brain className="h-5 w-5" />;
      case 'logistic_regression': return <Target className="h-5 w-5" />;
      case 'svm': return <Zap className="h-5 w-5" />;
      case 'neural_network': return <Cpu className="h-5 w-5" />;
      default: return <Settings className="h-5 w-5" />;
    }
  };

  const getModelTypeColor = (type: string) => {
    switch (type) {
      case 'naive_bayes': return 'text-blue-400 bg-blue-500/20 border-blue-500/30';
      case 'logistic_regression': return 'text-green-400 bg-green-500/20 border-green-500/30';
      case 'svm': return 'text-purple-400 bg-purple-500/20 border-purple-500/30';
      case 'neural_network': return 'text-orange-400 bg-orange-500/20 border-orange-500/30';
      default: return 'text-gray-400 bg-gray-500/20 border-gray-500/30';
    }
  };

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-4xl font-bold text-white mb-4">Entraînement de Modèles ML</h2>
        <p className="text-white/70 text-xl">Créez et entraînez vos propres modèles de classification de sentiment</p>
      </div>

      {/* Configuration d'entraînement */}
      <div className="bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-white/10 shadow-2xl">
        <h3 className="text-2xl font-bold text-white mb-6 flex items-center space-x-3">
          <Settings className="h-7 w-7 text-cyan-400" />
          <span>Configuration d'entraînement</span>
        </h3>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Type de modèle */}
          <div>
            <label className="block text-white font-medium mb-3">Type de modèle</label>
            <select
              value={config.modelType}
              onChange={(e) => setConfig({...config, modelType: e.target.value as any})}
              className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-cyan-400"
              disabled={isTraining}
            >
              <option value="naive_bayes">Naive Bayes</option>
              <option value="logistic_regression">Régression Logistique</option>
              <option value="svm">SVM (Support Vector Machine)</option>
              <option value="neural_network">Réseau de Neurones</option>
            </select>
          </div>

          {/* Taille du test */}
          <div>
            <label className="block text-white font-medium mb-3">Taille du test (%)</label>
            <input
              type="number"
              min="10"
              max="50"
              value={config.testSize * 100}
              onChange={(e) => setConfig({...config, testSize: parseInt(e.target.value) / 100})}
              className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-cyan-400"
              disabled={isTraining}
            />
          </div>

          {/* Max features */}
          <div>
            <label className="block text-white font-medium mb-3">Features max</label>
            <input
              type="number"
              min="100"
              max="10000"
              step="100"
              value={config.maxFeatures}
              onChange={(e) => setConfig({...config, maxFeatures: parseInt(e.target.value)})}
              className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-cyan-400"
              disabled={isTraining}
            />
          </div>

          {/* N-grams */}
          <div>
            <label className="block text-white font-medium mb-3">N-grams (min-max)</label>
            <div className="flex space-x-2">
              <input
                type="number"
                min="1"
                max="3"
                value={config.ngrams[0]}
                onChange={(e) => setConfig({...config, ngrams: [parseInt(e.target.value), config.ngrams[1]]})}
                className="flex-1 p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-cyan-400"
                disabled={isTraining}
              />
              <input
                type="number"
                min="1"
                max="3"
                value={config.ngrams[1]}
                onChange={(e) => setConfig({...config, ngrams: [config.ngrams[0], parseInt(e.target.value)]})}
                className="flex-1 p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-cyan-400"
                disabled={isTraining}
              />
            </div>
          </div>

          {/* Epochs (pour neural network) */}
          {(config.modelType === 'neural_network' || config.modelType === 'logistic_regression') && (
            <div>
              <label className="block text-white font-medium mb-3">Epochs</label>
              <input
                type="number"
                min="10"
                max="1000"
                step="10"
                value={config.epochs || 100}
                onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
                className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-cyan-400"
                disabled={isTraining}
              />
            </div>
          )}

          {/* Learning rate */}
          {(config.modelType === 'neural_network' || config.modelType === 'logistic_regression') && (
            <div>
              <label className="block text-white font-medium mb-3">Taux d'apprentissage</label>
              <input
                type="number"
                min="0.001"
                max="0.1"
                step="0.001"
                value={config.learningRate || 0.01}
                onChange={(e) => setConfig({...config, learningRate: parseFloat(e.target.value)})}
                className="w-full p-3 bg-white/10 border border-white/20 rounded-xl text-white focus:outline-none focus:border-cyan-400"
                disabled={isTraining}
              />
            </div>
          )}
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
              <span>{Math.floor(reviews.length * (1 - config.testSize))} pour l'entraînement</span>
            </div>
            <div className="flex items-center space-x-2">
              <Eye className="h-5 w-5" />
              <span>{Math.floor(reviews.length * config.testSize)} pour le test</span>
            </div>
          </div>
        </div>

        {/* Bouton d'entraînement */}
        <div className="mt-6">
          <button
            onClick={handleTrainModel}
            disabled={isTraining || reviews.length === 0}
            className="w-full px-8 py-4 bg-gradient-to-r from-green-500 to-teal-600 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-3"
          >
            {isTraining ? (
              <>
                <RefreshCw className="h-6 w-6 animate-spin" />
                <span>Entraînement en cours...</span>
              </>
            ) : (
              <>
                <Play className="h-6 w-6" />
                <span>Commencer l'entraînement</span>
              </>
            )}
          </button>

          {/* Barre de progression */}
          {isTraining && (
            <div className="mt-4">
              <div className="flex justify-between text-white/70 text-sm mb-2">
                <span>{trainingStatus}</span>
                <span>{trainingProgress.toFixed(0)}%</span>
              </div>
              <div className="w-full bg-white/20 rounded-full h-3">
                <div 
                  className="bg-gradient-to-r from-green-400 to-teal-500 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${trainingProgress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Liste des modèles entraînés */}
      <div className="bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-white/10 shadow-2xl">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-2xl font-bold text-white flex items-center space-x-3">
            <Brain className="h-7 w-7 text-purple-400" />
            <span>Modèles entraînés ({trainedModels.length})</span>
          </h3>
          
          <div className="flex space-x-3">
            <label className="px-4 py-2 bg-blue-500/20 text-blue-400 rounded-xl hover:bg-blue-500/30 transition-colors cursor-pointer flex items-center space-x-2 border border-blue-500/30">
              <Upload className="h-5 w-5" />
              <span>Charger modèle</span>
              <input
                type="file"
                accept=".json"
                onChange={handleLoadModel}
                className="hidden"
              />
            </label>
          </div>
        </div>

        {trainedModels.length === 0 ? (
          <div className="text-center py-12 text-white/60">
            <Brain className="h-16 w-16 mx-auto mb-4 opacity-50" />
            <p className="text-lg">Aucun modèle entraîné</p>
            <p className="text-sm">Commencez par entraîner votre premier modèle</p>
          </div>
        ) : (
          <div className="grid gap-4">
            {trainedModels.map((model) => (
              <div
                key={model.id}
                className={`p-6 rounded-xl border transition-all hover:scale-[1.02] cursor-pointer ${
                  selectedModel?.id === model.id
                    ? 'bg-cyan-500/20 border-cyan-500/50'
                    : 'bg-white/5 border-white/10 hover:border-white/20'
                }`}
                onClick={() => setSelectedModel(model)}
              >
                <div className="flex justify-between items-start mb-4">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${getModelTypeColor(model.type)}`}>
                      {getModelTypeIcon(model.type)}
                    </div>
                    <div>
                      <h4 className="text-white font-bold text-lg">{model.name}</h4>
                      <p className="text-white/60 text-sm">
                        Entraîné le {new Date(model.createdAt).toLocaleDateString()} • 
                        {model.trainedOn} exemples
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex space-x-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        ModelTrainer.downloadModel(model);
                      }}
                      className="p-2 text-white/60 hover:text-white transition-colors"
                    >
                      <Download className="h-5 w-5" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteModel(model.id);
                      }}
                      className="p-2 text-red-400 hover:text-red-300 transition-colors"
                    >
                      <Trash2 className="h-5 w-5" />
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-400">
                      {(model.metrics.accuracy * 100).toFixed(1)}%
                    </div>
                    <div className="text-white/60 text-sm">Précision</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-400">
                      {(model.metrics.f1Score * 100).toFixed(1)}%
                    </div>
                    <div className="text-white/60 text-sm">F1-Score</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-400">
                      {model.vocabulary.length}
                    </div>
                    <div className="text-white/60 text-sm">Features</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-400">
                      {formatDuration(model.metrics.trainingTime)}
                    </div>
                    <div className="text-white/60 text-sm">Temps</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Test du modèle sélectionné */}
      {selectedModel && (
        <div className="bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-white/10 shadow-2xl">
          <h3 className="text-2xl font-bold text-white mb-6 flex items-center space-x-3">
            <Target className="h-7 w-7 text-cyan-400" />
            <span>Tester le modèle : {selectedModel.name}</span>
          </h3>

          <div className="space-y-6">
            <div>
              <label className="block text-white font-medium mb-3">Texte à analyser</label>
              <textarea
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                placeholder="Entrez un avis à classifier..."
                className="w-full h-32 p-4 bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/50 focus:outline-none focus:border-cyan-400 resize-none"
              />
            </div>

            <button
              onClick={handleTestModel}
              disabled={!testText.trim()}
              className="px-8 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold transition-all hover:scale-105 disabled:opacity-50"
            >
              Analyser avec le modèle
            </button>

            {prediction && (
              <div className="p-6 bg-white/5 rounded-xl">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-2xl font-bold text-white mb-2">
                      Prédiction : {prediction.prediction === 1 ? 'Positif' : 'Négatif'}
                    </div>
                    <div className="text-white/60">
                      Confiance : {(prediction.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className={`text-6xl ${prediction.prediction === 1 ? 'text-green-400' : 'text-red-400'}`}>
                    {prediction.prediction === 1 ? '😊' : '😞'}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Métriques détaillées */}
      {selectedModel && (
        <div className="bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-white/10 shadow-2xl">
          <h3 className="text-2xl font-bold text-white mb-6 flex items-center space-x-3">
            <BarChart3 className="h-7 w-7 text-purple-400" />
            <span>Métriques détaillées</span>
          </h3>

          <div className="grid md:grid-cols-2 gap-8">
            {/* Métriques principales */}
            <div>
              <h4 className="text-white font-bold text-lg mb-4">Performance</h4>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-white/70">Précision (Accuracy)</span>
                  <span className="text-green-400 font-bold">
                    {(selectedModel.metrics.accuracy * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/70">Précision (Precision)</span>
                  <span className="text-blue-400 font-bold">
                    {(selectedModel.metrics.precision * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/70">Rappel (Recall)</span>
                  <span className="text-purple-400 font-bold">
                    {(selectedModel.metrics.recall * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/70">F1-Score</span>
                  <span className="text-orange-400 font-bold">
                    {(selectedModel.metrics.f1Score * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Matrice de confusion */}
            <div>
              <h4 className="text-white font-bold text-lg mb-4">Matrice de confusion</h4>
              <div className="grid grid-cols-2 gap-2 text-center">
                <div className="p-4 bg-green-500/20 rounded-lg">
                  <div className="text-2xl font-bold text-green-400">
                    {selectedModel.metrics.confusionMatrix[0][0]}
                  </div>
                  <div className="text-white/60 text-sm">Vrais Négatifs</div>
                </div>
                <div className="p-4 bg-red-500/20 rounded-lg">
                  <div className="text-2xl font-bold text-red-400">
                    {selectedModel.metrics.confusionMatrix[0][1]}
                  </div>
                  <div className="text-white/60 text-sm">Faux Positifs</div>
                </div>
                <div className="p-4 bg-red-500/20 rounded-lg">
                  <div className="text-2xl font-bold text-red-400">
                    {selectedModel.metrics.confusionMatrix[1][0]}
                  </div>
                  <div className="text-white/60 text-sm">Faux Négatifs</div>
                </div>
                <div className="p-4 bg-green-500/20 rounded-lg">
                  <div className="text-2xl font-bold text-green-400">
                    {selectedModel.metrics.confusionMatrix[1][1]}
                  </div>
                  <div className="text-white/60 text-sm">Vrais Positifs</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}; 