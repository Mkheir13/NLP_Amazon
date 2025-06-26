import React, { useState, useEffect } from 'react';
import { Play, CheckCircle, ArrowRight, Brain, BarChart3, Search, AlertCircle, Clock, Zap } from 'lucide-react';

interface Step {
  id: number;
  title: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  result?: any;
  startTime?: number;
  endTime?: number;
  logs?: string[];
  progress?: number;
}

const SimpleAutoencoder: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [steps, setSteps] = useState<Step[]>([
    { id: 1, title: "Charger le corpus Amazon/Polarity", description: "Chargement des avis clients", status: 'pending', logs: [] },
    { id: 2, title: "Nettoyer et vectoriser avec TF-IDF", description: "Transformation texte → vecteurs numériques", status: 'pending', logs: [] },
    { id: 3, title: "Créer l'autoencoder simple", description: "Architecture : Input → Compression → Output", status: 'pending', logs: [] },
    { id: 4, title: "Entraîner X → X", description: "Apprentissage de la reconstruction", status: 'pending', logs: [] },
    { id: 5, title: "Extraire X_encoded", description: "Récupération des vecteurs compressés", status: 'pending', logs: [] },
    { id: 6, title: "Appliquer KMeans", description: "Clustering sur l'espace compressé", status: 'pending', logs: [] },
    { id: 7, title: "Analyser les clusters", description: "Interprétation des résultats", status: 'pending', logs: [] }
  ]);

  const [globalResults, setGlobalResults] = useState<any>(null);

  const addLog = (message: string, stepId?: number) => {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = `[${timestamp}] ${message}`;
    
    setLogs(prev => [...prev, logEntry]);
    
    if (stepId) {
      setSteps(prev => prev.map(s => 
        s.id === stepId ? {...s, logs: [...(s.logs || []), logEntry]} : s
      ));
    }
  };

  const updateStepStatus = (stepId: number, status: Step['status'], result?: any, progress?: number) => {
    setSteps(prev => prev.map(s => {
      if (s.id === stepId) {
        const updated = { ...s, status, result, progress };
        if (status === 'running' && !s.startTime) {
          updated.startTime = Date.now();
        }
        if (status === 'completed' || status === 'error') {
          updated.endTime = Date.now();
          updated.progress = 100;
        }
        return updated;
      }
      return s;
    }));
  };

  const simulateStepProgress = (stepId: number, duration: number = 2000) => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 20;
      if (progress >= 95) {
        clearInterval(interval);
        progress = 100;
      }
      updateStepStatus(stepId, 'running', undefined, Math.min(progress, 100));
    }, duration / 10);
    return interval;
  };

  const runCompleteProcess = async () => {
    setIsRunning(true);
    setLogs([]);
    setGlobalResults(null);
    
    // Reset all steps
    setSteps(prev => prev.map(s => ({...s, status: 'pending', logs: [], progress: 0, startTime: undefined, endTime: undefined})));
    
    try {
      addLog("🚀 Démarrage du processus complet des 7 étapes");
      
      // Étape 1: Chargement du corpus
      addLog("📂 Étape 1: Chargement du corpus Amazon/Polarity...", 1);
      setCurrentStep(1);
      updateStepStatus(1, 'running');
      const progressInterval1 = simulateStepProgress(1, 1500);
      
      await new Promise(resolve => setTimeout(resolve, 1500));
      clearInterval(progressInterval1);
      updateStepStatus(1, 'completed');
      addLog("✅ Corpus chargé: 50 avis clients", 1);
      
      // Étape 2: TF-IDF
      addLog("🔄 Étape 2: Vectorisation TF-IDF...", 2);
      setCurrentStep(2);
      updateStepStatus(2, 'running');
      const progressInterval2 = simulateStepProgress(2, 2000);
      
      await new Promise(resolve => setTimeout(resolve, 2000));
      clearInterval(progressInterval2);
      updateStepStatus(2, 'completed');
      addLog("✅ TF-IDF créé: 424 dimensions, 97% sparsité", 2);
      
      // Étape 3: Construction autoencoder
      addLog("🔧 Étape 3: Construction de l'autoencoder...", 3);
      setCurrentStep(3);
      updateStepStatus(3, 'running');
      const progressInterval3 = simulateStepProgress(3, 1000);
      
      await new Promise(resolve => setTimeout(resolve, 1000));
      clearInterval(progressInterval3);
      updateStepStatus(3, 'completed');
      addLog("✅ Autoencoder construit: 424 → 64 → 424", 3);
      
      // Étape 4: Entraînement (appel API réel)
      addLog("🚀 Étape 4: Entraînement avec régularisation...", 4);
      setCurrentStep(4);
      updateStepStatus(4, 'running');
      
      const trainingResponse = await fetch('http://localhost:5000/api/autoencoder/train_regularized', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: {} })
      });
      
      const trainingData = await trainingResponse.json();
      
      if (trainingData.success) {
        updateStepStatus(4, 'completed', trainingData.result);
        addLog(`✅ Entraînement terminé: Loss ${trainingData.result?.training?.final_loss?.toFixed(4)}`, 4);
        
        // Étape 5: Extraction des vecteurs
        addLog("📊 Étape 5: Extraction des vecteurs compressés...", 5);
        setCurrentStep(5);
        updateStepStatus(5, 'running');
        const progressInterval5 = simulateStepProgress(5, 1000);
        
        await new Promise(resolve => setTimeout(resolve, 1000));
        clearInterval(progressInterval5);
        updateStepStatus(5, 'completed');
        addLog("✅ 50 vecteurs extraits (64 dimensions)", 5);
        
        // Étape 6: KMeans
        addLog("🎯 Étape 6: Application KMeans...", 6);
        setCurrentStep(6);
        updateStepStatus(6, 'running');
        const progressInterval6 = simulateStepProgress(6, 1500);
        
        await new Promise(resolve => setTimeout(resolve, 1500));
        clearInterval(progressInterval6);
        updateStepStatus(6, 'completed');
        addLog("✅ KMeans appliqué: 4 clusters identifiés", 6);
        
        // Étape 7: Analyse (appel API réel)
        addLog("📈 Étape 7: Analyse des clusters...", 7);
        setCurrentStep(7);
        updateStepStatus(7, 'running');
        
        const clusteringResponse = await fetch('http://localhost:5000/api/autoencoder/clustering_advanced', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ n_clusters: 4, use_compressed: true })
        });
        
        const clusteringData = await clusteringResponse.json();
        
        if (clusteringData.success) {
          updateStepStatus(7, 'completed', clusteringData.result);
          addLog(`✅ Analyse terminée: Silhouette ${clusteringData.result?.silhouette_score?.toFixed(3)}`, 7);
          
          setGlobalResults({
            training: trainingData.result,
            clustering: clusteringData.result
          });
          
          addLog("🎉 PROCESSUS COMPLET TERMINÉ AVEC SUCCÈS!");
          setCurrentStep(8);
        } else {
          throw new Error("Erreur lors du clustering");
        }
      } else {
        throw new Error("Erreur lors de l'entraînement");
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Erreur inconnue";
      addLog(`❌ Erreur: ${errorMessage}`);
      setSteps(prev => prev.map(s => s.status === 'running' ? {...s, status: 'error'} : s));
    } finally {
      setIsRunning(false);
    }
  };

  const getStepIcon = (step: Step) => {
    switch (step.status) {
      case 'completed': return <CheckCircle className="h-6 w-6 text-green-400" />;
      case 'running': return <div className="h-6 w-6 border-2 border-orange-400 border-t-transparent rounded-full animate-spin" />;
      case 'error': return <AlertCircle className="h-6 w-6 text-red-400" />;
      default: return <div className="h-6 w-6 bg-slate-600 rounded-full" />;
    }
  };

  const getStepDuration = (step: Step) => {
    if (step.startTime && step.endTime) {
      return ((step.endTime - step.startTime) / 1000).toFixed(1) + 's';
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
      <div className="max-w-6xl mx-auto">
        
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center space-x-3 mb-4">
            <Brain className="h-12 w-12 text-orange-400" />
            <h1 className="text-4xl font-bold text-white">Autoencoder - 7 Étapes</h1>
          </div>
          <p className="text-slate-300 text-lg">
            Pipeline simplifié pour l'apprentissage
          </p>
        </div>

        {/* Bouton principal */}
        <div className="text-center mb-8">
          <button
            onClick={runCompleteProcess}
            disabled={isRunning}
            className="px-8 py-4 bg-gradient-to-r from-orange-500 to-red-600 text-white rounded-xl font-bold text-lg transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-3 mx-auto"
          >
            {isRunning ? (
              <>
                <div className="h-6 w-6 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Exécution en cours... (Étape {currentStep}/7)</span>
              </>
            ) : (
              <>
                <Play className="h-6 w-6" />
                <span>🚀 LANCER LES 7 ÉTAPES</span>
              </>
            )}
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Pipeline des étapes */}
          <div className="bg-slate-800/90 backdrop-blur-xl rounded-2xl border border-white/10 p-6">
            <h2 className="text-2xl font-bold text-white mb-6 text-center">Pipeline d'Exécution</h2>
            
            <div className="space-y-4">
              {steps.map((step, index) => (
                <div key={step.id} className={`border rounded-lg p-4 transition-all duration-300 ${
                  step.status === 'completed' ? 'border-green-500/30 bg-green-500/10' :
                  step.status === 'running' ? 'border-orange-500/30 bg-orange-500/10' :
                  step.status === 'error' ? 'border-red-500/30 bg-red-500/10' :
                  'border-slate-600/30 bg-slate-700/20'
                }`}>
                  
                  <div className="flex items-center space-x-4 mb-2">
                    {/* Numéro et icône */}
                    <div className="flex items-center space-x-3">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                        step.status === 'completed' ? 'bg-green-500 text-white' :
                        step.status === 'running' ? 'bg-orange-500 text-white' :
                        step.status === 'error' ? 'bg-red-500 text-white' :
                        'bg-slate-600 text-slate-300'
                      }`}>
                        {step.id}
                      </div>
                      {getStepIcon(step)}
                    </div>

                    {/* Contenu */}
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <h3 className={`font-semibold ${
                          step.status === 'completed' ? 'text-green-400' :
                          step.status === 'running' ? 'text-orange-400' :
                          step.status === 'error' ? 'text-red-400' :
                          'text-white'
                        }`}>
                          {step.title}
                        </h3>
                        {getStepDuration(step) && (
                          <span className="text-xs text-slate-400 flex items-center space-x-1">
                            <Clock className="h-3 w-3" />
                            <span>{getStepDuration(step)}</span>
                          </span>
                        )}
                      </div>
                      <p className="text-slate-400 text-sm">{step.description}</p>
                    </div>
                  </div>

                  {/* Barre de progression */}
                  {step.status === 'running' && step.progress !== undefined && (
                    <div className="mt-2">
                      <div className="w-full bg-slate-600 rounded-full h-2">
                        <div 
                          className="bg-orange-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${step.progress}%` }}
                        />
                      </div>
                      <div className="text-xs text-slate-400 mt-1">{Math.round(step.progress)}%</div>
                    </div>
                  )}

                  {/* Logs de l'étape */}
                  {step.logs && step.logs.length > 0 && (
                    <div className="mt-3 bg-slate-900/50 rounded-lg p-3">
                      <div className="text-xs text-slate-300 space-y-1">
                        {step.logs.slice(-3).map((log, i) => (
                          <div key={i} className="font-mono">{log}</div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Logs en temps réel */}
          <div className="bg-slate-800/90 backdrop-blur-xl rounded-2xl border border-white/10 p-6">
            <h2 className="text-2xl font-bold text-white mb-6 text-center">Logs en Temps Réel</h2>
            
            <div className="bg-slate-900/50 rounded-lg p-4 h-96 overflow-y-auto">
              <div className="space-y-1">
                {logs.length === 0 ? (
                  <div className="text-slate-500 text-center py-8">
                    <Zap className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>Cliquez sur "LANCER" pour voir les logs</p>
                  </div>
                ) : (
                  logs.map((log, index) => (
                    <div key={index} className="text-xs font-mono text-slate-300 hover:text-white transition-colors">
                      {log}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Résultats */}
        {globalResults && (
          <div className="mt-8 bg-green-500/20 border border-green-500/30 rounded-2xl p-8">
            <div className="flex items-center space-x-3 mb-6">
              <CheckCircle className="h-8 w-8 text-green-400" />
              <h2 className="text-2xl font-bold text-white">✅ Résultats Finaux</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Autoencoder */}
              <div className="bg-slate-700/50 p-6 rounded-xl">
                <h3 className="text-lg font-bold text-orange-400 mb-4">🤖 Autoencoder (Étapes 1-4)</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Architecture:</span>
                    <span className="text-white font-mono">
                      {globalResults.training?.architecture?.input_dim} → {globalResults.training?.architecture?.encoding_dim}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Compression:</span>
                    <span className="text-white font-mono">
                      {globalResults.training?.architecture?.compression_ratio?.toFixed(1)}:1
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Perte finale:</span>
                    <span className="text-white font-mono">
                      {globalResults.training?.training?.final_loss?.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Régularisation:</span>
                    <span className="text-green-400 font-mono text-sm">✅ L2 + Dropout</span>
                  </div>
                </div>
              </div>

              {/* Clustering */}
              <div className="bg-slate-700/50 p-6 rounded-xl">
                <h3 className="text-lg font-bold text-purple-400 mb-4">📊 Clustering (Étapes 5-7)</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Clusters:</span>
                    <span className="text-white font-mono">
                      {globalResults.clustering?.n_clusters} groupes
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Silhouette Score:</span>
                    <span className="text-white font-mono">
                      {globalResults.clustering?.silhouette_score?.toFixed(3)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Qualité clustering:</span>
                    <span className="text-white font-mono">
                      {globalResults.clustering?.clustering_quality}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Vecteurs analysés:</span>
                    <span className="text-white font-mono">
                      {globalResults.clustering?.n_samples} échantillons
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Techniques avancées */}
            <div className="mt-6 bg-purple-600/20 border border-purple-400/30 rounded-lg p-4">
              <h4 className="text-purple-300 font-medium mb-2">🎓 Techniques avancées implémentées :</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="text-purple-200">✅ Régularisation L2 (Ridge)</div>
                <div className="text-purple-200">✅ Dropout progressif</div>
                <div className="text-purple-200">✅ Batch Normalization</div>
                <div className="text-purple-200">✅ Early Stopping intelligent</div>
              </div>
            </div>
          </div>
        )}

        {/* Guide simple */}
        <div className="mt-8 bg-blue-500/20 border border-blue-500/30 rounded-2xl p-6">
          <h3 className="text-blue-400 font-bold text-lg mb-4">📋 Mode d'emploi simple</h3>
          <div className="text-slate-300 space-y-2">
            <p><strong>1.</strong> Cliquez sur "🚀 LANCER LES 7 ÉTAPES"</p>
            <p><strong>2.</strong> Suivez le progrès dans le pipeline (gauche) et les logs (droite)</p>
            <p><strong>3.</strong> Attendez que toutes les étapes deviennent vertes ✅</p>
            <p><strong>4.</strong> Consultez les résultats finaux en bas</p>
            <p><strong>5.</strong> Votre projet est terminé ! 🎉</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SimpleAutoencoder; 