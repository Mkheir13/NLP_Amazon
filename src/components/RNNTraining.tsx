import React, { useState, useEffect } from 'react';
import { Brain, Play, Eye, Save, Download, BarChart3, Zap, BookOpen, CheckCircle, AlertCircle, Info, Code2 } from 'lucide-react';
import CodeViewer from './CodeViewer';

interface RNNTrainingProps {
  isVisible?: boolean;
}

interface RNNInfo {
  status: string;
  model_type?: string;
  framework?: string;
  vocab_size?: number;
  architecture?: {
    embedding_dim: number;
    hidden_dim: number;
    output_dim: number;
  };
  parameters?: number;
  device?: string;
  implementation_requirements?: string;
}

interface TrainingResults {
  final_train_acc: number;
  final_val_acc: number;
  vocab_size: number;
  architecture: string;
  implementation_validation?: {
    [key: string]: string;
  };
}

interface Prediction {
  text: string;
  sentiment: string;
  confidence: number;
  probabilities: {
    negative: number;
    positive: number;
  };
  model_type: string;
}

const RNNTraining: React.FC<RNNTrainingProps> = ({ isVisible = true }) => {
  const [rnnInfo, setRnnInfo] = useState<RNNInfo>({ status: 'not_trained' });
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResults, setTrainingResults] = useState<TrainingResults | null>(null);
  const [testText, setTestText] = useState('This product is absolutely amazing and works perfectly!');
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [showCode, setShowCode] = useState(false);

  // Configuration d'entraînement simplifiée
  const [config, setConfig] = useState({
    epochs: 20,
    batch_size: 32,
    learning_rate: 0.001
  });

  useEffect(() => {
    if (isVisible) {
      fetchRNNInfo();
    }
  }, [isVisible]);

  const fetchRNNInfo = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/rnn/info');
      const data = await response.json();
      if (data.success) {
        setRnnInfo(data.info);
      }
    } catch (error) {
      console.error('Erreur récupération info RNN:', error);
    }
  };

  const trainRNN = async () => {
    setIsTraining(true);
    setTrainingResults(null);
    
    try {
      // Dataset d'exemple pour l'entraînement supervisé
      const trainingData = {
        texts: [
          // Avis positifs
          "This product is excellent and I love it",
          "Great quality and fantastic value for money",
          "Amazing item works perfectly as described",
          "Outstanding quality highly recommend this product",
          "Brilliant design and excellent build quality",
          "Superb craftsmanship and innovative features",
          "Fantastic product exceeds all expectations",
          "Wonderful purchase very satisfied with quality",
          "Impressive features and outstanding performance",
          "Remarkable innovation and exceptional value",
          "Perfect for my needs works flawlessly",
          "High quality materials and excellent design",
          
          // Avis négatifs
          "Terrible product completely broken on arrival",
          "Very poor quality waste of money",
          "Awful experience and horrible customer service",
          "Disappointing quality not worth the price",
          "Substandard materials and cheap construction",
          "Inferior quality and mediocre performance",
          "Poor quality broke after one day",
          "Defective item arrived damaged",
          "Unsatisfactory performance not recommended",
          "Mediocre quality average at best",
          "Cheap materials and poor construction",
          "Not worth the money disappointed"
        ],
        labels: [
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 12 Positifs
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   // 12 Négatifs
        ],
        config: config
      };

      console.log('🚀 Démarrage entraînement RNN from scratch...');
      
      const response = await fetch('http://localhost:5000/api/rnn/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainingData)
      });

      const data = await response.json();
      
      if (data.success) {
        setTrainingResults(data.results);
        await fetchRNNInfo(); // Rafraîchir les infos
        console.log('✅ RNN entraîné avec succès !');
      } else {
        throw new Error(data.error || 'Erreur inconnue');
      }
    } catch (error) {
      console.error('Erreur entraînement RNN:', error);
      alert('Erreur lors de l\'entraînement: ' + error);
    } finally {
      setIsTraining(false);
    }
  };

  const predictSentiment = async () => {
    if (!testText.trim()) return;
    
    setIsPredicting(true);
    setPrediction(null);
    
    try {
      const response = await fetch('http://localhost:5000/api/rnn/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: testText })
      });

      const data = await response.json();
      
      if (data.success) {
        setPrediction(data.prediction);
      } else {
        throw new Error(data.error || 'Erreur prédiction');
      }
    } catch (error) {
      console.error('Erreur prédiction:', error);
      alert('Erreur lors de la prédiction: ' + error);
    } finally {
      setIsPredicting(false);
    }
  };

  if (!isVisible) return null;

  return (
    <div className="space-y-6">
      {/* Header avec implémentation from scratch */}
      <div className="bg-gradient-to-r from-purple-900 to-indigo-900 rounded-xl p-6 border border-purple-500/30">
        <div className="flex items-center space-x-3 mb-4">
          <Brain className="h-8 w-8 text-purple-400" />
          <div>
            <h2 className="text-2xl font-bold text-white">RNN From Scratch</h2>
            <p className="text-purple-200">Implémentation complète from scratch avec PyTorch</p>
          </div>
        </div>
        
        <div className="bg-purple-800/30 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-3">
            <BookOpen className="h-5 w-5 text-purple-300" />
            <h3 className="text-lg font-semibold text-white">✅ Implémentation From Scratch Complète</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-4 w-4 text-green-400" />
              <span className="text-purple-100">RNN implémenté from scratch (pas nn.RNN)</span>
            </div>
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-4 w-4 text-green-400" />
              <span className="text-purple-100">CrossEntropyLoss (pas MSELoss)</span>
            </div>
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-4 w-4 text-green-400" />
              <span className="text-purple-100">Métriques de précision détaillées</span>
            </div>
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-4 w-4 text-green-400" />
              <span className="text-purple-100">Visualisation Matplotlib</span>
            </div>
          </div>
        </div>
      </div>

      {/* Informations du modèle */}
      <div className="bg-slate-800/90 backdrop-blur-xl rounded-xl p-6 border border-slate-600/30">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold text-white flex items-center space-x-2">
            <Info className="h-6 w-6 text-blue-400" />
            <span>État du Modèle RNN</span>
          </h3>
          <button
            onClick={fetchRNNInfo}
            className="px-3 py-1 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
          >
            Actualiser
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-slate-400">Statut:</span>
              <span className={`px-2 py-1 rounded text-sm font-medium ${
                rnnInfo.status === 'trained' 
                  ? 'bg-green-500/20 text-green-400' 
                  : 'bg-yellow-500/20 text-yellow-400'
              }`}>
                {rnnInfo.status === 'trained' ? 'Entraîné' : 'Non entraîné'}
              </span>
            </div>
            
            {rnnInfo.framework && (
              <div className="flex justify-between">
                <span className="text-slate-400">Framework:</span>
                <span className="text-white font-medium">{rnnInfo.framework}</span>
              </div>
            )}
            
            {rnnInfo.device && (
              <div className="flex justify-between">
                <span className="text-slate-400">Device:</span>
                <span className="text-white font-mono text-sm">{rnnInfo.device}</span>
              </div>
            )}
          </div>

          <div className="space-y-3">
            {rnnInfo.vocab_size && (
              <div className="flex justify-between">
                <span className="text-slate-400">Vocabulaire:</span>
                <span className="text-white font-medium">{rnnInfo.vocab_size.toLocaleString()} mots</span>
              </div>
            )}
            
            {rnnInfo.parameters && (
              <div className="flex justify-between">
                <span className="text-slate-400">Paramètres:</span>
                <span className="text-white font-medium">{rnnInfo.parameters.toLocaleString()}</span>
              </div>
            )}
            
            {rnnInfo.architecture && (
              <div className="flex justify-between">
                <span className="text-slate-400">Architecture:</span>
                <span className="text-white font-mono text-sm">
                  {rnnInfo.architecture.embedding_dim}D → {rnnInfo.architecture.hidden_dim}D → {rnnInfo.architecture.output_dim}D
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Configuration d'entraînement */}
      <div className="bg-slate-800/90 backdrop-blur-xl rounded-xl p-6 border border-slate-600/30">
        <h3 className="text-xl font-semibold text-white mb-4 flex items-center space-x-2">
          <Zap className="h-6 w-6 text-yellow-400" />
          <span>Configuration d'Entraînement</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Époques
            </label>
            <input
              type="number"
              value={config.epochs}
              onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
              min="1"
              max="100"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Batch Size
            </label>
            <input
              type="number"
              value={config.batch_size}
              onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
              min="1"
              max="128"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Learning Rate
            </label>
            <input
              type="number"
              step="0.0001"
              value={config.learning_rate}
              onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
              min="0.0001"
              max="0.01"
            />
          </div>
        </div>

        <button
          onClick={trainRNN}
          disabled={isTraining}
          className="w-full flex items-center justify-center space-x-2 bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-6 py-3 rounded-xl hover:from-purple-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
        >
          {isTraining ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span>Entraînement en cours...</span>
            </>
          ) : (
            <>
              <Play className="h-5 w-5" />
              <span>🎓 Entraîner RNN from scratch (PyTorch)</span>
            </>
          )}
        </button>
      </div>

      {/* Résultats d'entraînement */}
      {trainingResults && (
        <div className="bg-slate-800/90 backdrop-blur-xl rounded-xl p-6 border border-slate-600/30">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center space-x-2">
            <BarChart3 className="h-6 w-6 text-green-400" />
            <span>Résultats d'Entraînement</span>
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-medium text-white mb-3">Métriques Finales</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-slate-400">Précision Train:</span>
                  <span className="text-green-400 font-bold">{trainingResults.final_train_acc?.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Précision Validation:</span>
                  <span className="text-blue-400 font-bold">{trainingResults.final_val_acc?.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Vocabulaire:</span>
                  <span className="text-white">{trainingResults.vocab_size} mots</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Architecture:</span>
                  <span className="text-purple-400 text-sm">{trainingResults.architecture}</span>
                </div>
              </div>
            </div>

            <div className="bg-slate-700/50 rounded-lg p-4">
              <h4 className="text-lg font-medium text-white mb-3">Validation Implémentation</h4>
              {trainingResults.implementation_validation && (
                <div className="space-y-2 text-sm">
                  {Object.entries(trainingResults.implementation_validation).map(([key, value]) => (
                    <div key={key} className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-400 flex-shrink-0" />
                      <span className="text-green-300">{value}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Test de prédiction */}
      {rnnInfo.status === 'trained' && (
        <div className="bg-slate-800/90 backdrop-blur-xl rounded-xl p-6 border border-slate-600/30">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center space-x-2">
            <Eye className="h-6 w-6 text-cyan-400" />
            <span>Test de Prédiction RNN</span>
          </h3>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Texte à analyser:
              </label>
              <textarea
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-cyan-500 resize-none"
                rows={3}
                placeholder="Entrez votre texte ici..."
              />
            </div>

            <button
              onClick={predictSentiment}
              disabled={isPredicting || !testText.trim()}
              className="flex items-center space-x-2 bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isPredicting ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  <span>Analyse...</span>
                </>
              ) : (
                <>
                  <Eye className="h-4 w-4" />
                  <span>Analyser avec RNN</span>
                </>
              )}
            </button>

            {prediction && (
              <div className="bg-slate-700/50 rounded-lg p-4">
                <h4 className="text-lg font-medium text-white mb-3">Résultat de l'Analyse</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Sentiment:</span>
                    <span className={`px-2 py-1 rounded text-sm font-medium ${
                      prediction.sentiment === 'positive' 
                        ? 'bg-green-500/20 text-green-400' 
                        : 'bg-red-500/20 text-red-400'
                    }`}>
                      {prediction.sentiment === 'positive' ? '😊 Positif' : '😞 Négatif'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Confiance:</span>
                    <span className="text-white font-medium">{(prediction.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Modèle:</span>
                    <span className="text-purple-400 text-sm">{prediction.model_type}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Actions et Code */}
      <div className="bg-slate-800/90 backdrop-blur-xl rounded-xl p-6 border border-slate-600/30">
        <h3 className="text-xl font-semibold text-white mb-4">Actions et Code</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <button
            onClick={() => setShowCode(true)}
            className="flex items-center justify-center space-x-2 bg-gradient-to-r from-slate-600 to-slate-700 text-white px-4 py-3 rounded-lg hover:from-slate-700 hover:to-slate-800 transition-all duration-200"
          >
            <Code2 className="h-5 w-5" />
            <span>Voir le Code RNN</span>
          </button>

          <button
            onClick={() => window.open('http://localhost:5000/api/rnn/info', '_blank')}
            className="flex items-center justify-center space-x-2 bg-gradient-to-r from-blue-600 to-blue-700 text-white px-4 py-3 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all duration-200"
          >
            <Info className="h-5 w-5" />
            <span>API Info</span>
          </button>
        </div>
      </div>

      {/* Code Viewer Modal */}
      {showCode && (
        <CodeViewer
          isOpen={showCode}
          onClose={() => setShowCode(false)}
          section="rnn"
        />
      )}
    </div>
  );
};

export default RNNTraining; 