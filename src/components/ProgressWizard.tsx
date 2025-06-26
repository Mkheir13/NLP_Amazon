import React, { useState } from 'react';
import { CheckCircle, Circle, ArrowRight, Database, Brain, Target, BarChart3, Search, Eye, Network, Sparkles, Clock, AlertCircle, Code2, Unlock } from 'lucide-react';
import CodeViewer from './CodeViewer';

interface Step {
  id: string;
  title: string;
  description: string;
  view: string;
  icon: React.ReactNode;
  status: 'completed' | 'current' | 'locked' | 'available';
  dependencies?: string[];
  estimatedTime?: string;
}

interface ProgressWizardProps {
  currentView: string;
  onNavigate: (view: string) => void;
  completedSteps: string[];
  userProgress: {
    hasDataset: boolean;
    hasAnalyzed: boolean;
    hasTrained: boolean;
    hasVisualized: boolean;
  };
}

const ProgressWizard: React.FC<ProgressWizardProps> = ({
  currentView,
  onNavigate,
  completedSteps,
  userProgress
}) => {
  const [showCode, setShowCode] = useState<string | null>(null);
  const [forceUnlockAll, setForceUnlockAll] = useState(false);
  
  const steps: Step[] = [
    {
      id: 'dataset',
      title: 'Choisir des avis',
      description: 'Explorez et sélectionnez vos données Amazon',
      view: 'explore',
      icon: <Database className="h-5 w-5" />,
      status: 'available',
      estimatedTime: '2 min'
    },
    {
      id: 'preprocess',
      title: 'Prétraiter les textes',
      description: 'TF-IDF, nettoyage et vectorisation',
      view: 'embeddings_hub',
      icon: <Network className="h-5 w-5" />,
      status: forceUnlockAll || userProgress.hasDataset ? 'available' : 'locked',
      dependencies: ['dataset'],
      estimatedTime: '3 min'
    },
    {
      id: 'analyze',
      title: 'Détecter les sentiments',
      description: 'Analyse émotionnelle avec IA',
      view: 'analyze',
      icon: <Brain className="h-5 w-5" />,
      status: forceUnlockAll || userProgress.hasDataset ? 'available' : 'locked',
      dependencies: ['dataset'],
      estimatedTime: '1 min'
    },
    {
      id: 'train',
      title: 'Entraîner un modèle',
      description: 'Autoencoder ou classification',
      view: 'training',
      icon: <Target className="h-5 w-5" />,
      status: forceUnlockAll || userProgress.hasAnalyzed || completedSteps.includes('analyze') ? 'available' : 'locked',
      dependencies: ['dataset', 'analyze'],
      estimatedTime: '5 min'
    },
    {
      id: 'visualize',
      title: 'Visualiser ou clusteriser',
      description: 'Graphiques et insights',
      view: 'embeddings',
      icon: <Eye className="h-5 w-5" />,
      status: forceUnlockAll || userProgress.hasTrained || completedSteps.includes('train') ? 'available' : 'locked',
      dependencies: ['preprocess', 'train'],
      estimatedTime: '2 min'
    }
  ];

  // Mise à jour du statut des étapes
  const updatedSteps = steps.map(step => {
    if (completedSteps.includes(step.id)) {
      return { ...step, status: 'completed' as const };
    }
    if (step.view === currentView) {
      return { ...step, status: 'current' as const };
    }
    return step;
  });

  const getStepIcon = (step: Step) => {
    switch (step.status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-400" />;
      case 'current':
        return <div className="h-5 w-5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />;
      case 'locked':
        return <Circle className="h-5 w-5 text-gray-500" />;
      default:
        return step.icon;
    }
  };

  const getStepStyle = (step: Step) => {
    switch (step.status) {
      case 'completed':
        return 'bg-green-500/20 border-green-500/30 text-green-400';
      case 'current':
        return 'bg-blue-500/20 border-blue-500/30 text-blue-400 ring-2 ring-blue-500/50';
      case 'locked':
        return 'bg-gray-500/10 border-gray-500/20 text-gray-500 cursor-not-allowed';
      default:
        return 'bg-slate-700/50 border-slate-600/30 text-white hover:bg-slate-600/50 cursor-pointer';
    }
  };

  const handleStepClick = (step: Step) => {
    if (step.status === 'locked') return;
    onNavigate(step.view);
  };

  const getNextStep = () => {
    const currentIndex = updatedSteps.findIndex(step => step.status === 'current');
    if (currentIndex === -1) return updatedSteps.find(step => step.status === 'available');
    return updatedSteps[currentIndex + 1];
  };

  const nextStep = getNextStep();
  const completedCount = updatedSteps.filter(step => step.status === 'completed').length;
  const progressPercentage = (completedCount / updatedSteps.length) * 100;

  return (
    <div className="bg-slate-800/90 backdrop-blur-xl rounded-2xl border border-white/10 p-6 mb-8">
      
      {/* Header avec progression */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-4">
          <div className="relative">
            <Sparkles className="h-8 w-8 text-purple-400" />
            <div className="absolute inset-0 bg-purple-400 rounded-full blur-lg opacity-30"></div>
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white">Parcours Guidé NLP</h2>
            <p className="text-slate-400">Suivez les étapes pour une analyse complète</p>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-right">
            <div className="text-2xl font-bold text-white">{completedCount}/{updatedSteps.length}</div>
            <div className="text-sm text-slate-400">étapes terminées</div>
          </div>
          <div className="flex flex-col space-y-2">
            <button
              onClick={() => setForceUnlockAll(!forceUnlockAll)}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                forceUnlockAll 
                  ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' 
                  : 'bg-slate-700/50 text-slate-400 border border-slate-600/30 hover:text-white'
              }`}
            >
              <Unlock className="h-3 w-3 mr-1 inline" />
              {forceUnlockAll ? 'Verrouiller' : 'Tout débloquer'}
            </button>
            <button
              onClick={() => setShowCode(showCode ? null : currentView)}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                showCode 
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' 
                  : 'bg-slate-700/50 text-slate-400 border border-slate-600/30 hover:text-white'
              }`}
            >
              <Code2 className="h-3 w-3 mr-1 inline" />
              {showCode ? 'Masquer code' : 'Voir code'}
            </button>
          </div>
        </div>
      </div>

      {/* Barre de progression */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-slate-400">Progression globale</span>
          <span className="text-sm text-white font-medium">{Math.round(progressPercentage)}%</span>
        </div>
        <div className="w-full bg-slate-700 rounded-full h-3">
          <div 
            className="bg-gradient-to-r from-purple-500 to-blue-500 h-3 rounded-full transition-all duration-500"
            style={{ width: `${progressPercentage}%` }}
          />
        </div>
      </div>

      {/* Étapes */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
        {updatedSteps.map((step, index) => (
          <div key={step.id} className="relative">
            <button
              onClick={() => handleStepClick(step)}
              disabled={step.status === 'locked'}
              className={`w-full p-4 rounded-xl border transition-all duration-300 ${getStepStyle(step)}`}
            >
              <div className="flex flex-col items-center space-y-2">
                <div className="flex items-center justify-center">
                  {getStepIcon(step)}
                </div>
                <div className="text-center">
                  <h3 className="font-semibold text-sm">{step.title}</h3>
                  <p className="text-xs opacity-75 mt-1">{step.description}</p>
                  {step.estimatedTime && (
                    <div className="flex items-center justify-center space-x-1 mt-2">
                      <Clock className="h-3 w-3" />
                      <span className="text-xs">{step.estimatedTime}</span>
                    </div>
                  )}
                </div>
              </div>
            </button>
            
            {/* Flèche de connexion */}
            {index < updatedSteps.length - 1 && (
              <div className="hidden md:block absolute top-1/2 -right-2 transform -translate-y-1/2 z-10">
                <ArrowRight className="h-4 w-4 text-slate-500" />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Suggestions et prochaine étape */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        {/* Prochaine étape suggérée */}
        {nextStep && (
          <div className="bg-blue-500/20 border border-blue-500/30 rounded-xl p-4">
            <div className="flex items-center space-x-3 mb-2">
              <div className="h-2 w-2 bg-blue-400 rounded-full animate-pulse"></div>
              <h4 className="font-semibold text-blue-400">Prochaine étape suggérée</h4>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-white font-medium">{nextStep.title}</p>
                <p className="text-blue-200 text-sm">{nextStep.description}</p>
              </div>
              <button
                onClick={() => handleStepClick(nextStep)}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center space-x-2"
              >
                <span>Continuer</span>
                <ArrowRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        )}

        {/* Conseils contextuels */}
        <div className="bg-purple-500/20 border border-purple-500/30 rounded-xl p-4">
          <div className="flex items-center space-x-3 mb-2">
            <Sparkles className="h-4 w-4 text-purple-400" />
            <h4 className="font-semibold text-purple-400">Conseil</h4>
          </div>
          <p className="text-white text-sm">
            {currentView === 'explore' && "Explorez les avis pour comprendre vos données avant l'analyse."}
            {currentView === 'embeddings_hub' && "Commencez par entraîner TF-IDF pour vectoriser vos textes."}
            {currentView === 'analyze' && "Testez l'analyse sur différents types d'avis pour voir la précision."}
            {currentView === 'training' && "L'autoencoder simple est parfait pour débuter avec la compression."}
            {currentView === 'embeddings' && "Utilisez la visualisation 2D pour identifier les groupes d'avis similaires."}
            {!['explore', 'embeddings_hub', 'analyze', 'training', 'embeddings'].includes(currentView) && 
             "Suivez le parcours guidé pour une expérience optimale."}
          </p>
        </div>
      </div>

      {/* Indicateurs de dépendances */}
      {updatedSteps.some(step => step.status === 'locked') && !forceUnlockAll && (
        <div className="mt-4 p-3 bg-yellow-500/20 border border-yellow-500/30 rounded-lg">
          <div className="flex items-center space-x-2 text-yellow-400">
            <AlertCircle className="h-4 w-4" />
            <span className="text-sm font-medium">
              Certaines étapes sont verrouillées. Complétez les étapes précédentes pour débloquer la suite.
            </span>
          </div>
        </div>
      )}

      {/* Mode déblocage complet */}
      {forceUnlockAll && (
        <div className="mt-4 p-3 bg-orange-500/20 border border-orange-500/30 rounded-lg">
          <div className="flex items-center space-x-2 text-orange-400">
            <Unlock className="h-4 w-4" />
            <span className="text-sm font-medium">
              Mode libre activé : Toutes les étapes sont accessibles. Vous pouvez naviguer librement dans le parcours.
            </span>
          </div>
        </div>
      )}

      {/* Code viewer */}
      {showCode && (
        <div className="mt-6">
          <CodeViewer stepId={showCode} isVisible={true} />
        </div>
      )}
    </div>
  );
};

export default ProgressWizard; 