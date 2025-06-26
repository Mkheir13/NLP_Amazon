import React from 'react';
import { ArrowRight, CheckCircle, Sparkles, Zap } from 'lucide-react';

interface NextStepButtonProps {
  currentView: string;
  onNavigate: (view: string, data?: any) => void;
  isEnabled?: boolean;
  completedAction?: boolean;
  contextData?: any;
}

const NextStepButton: React.FC<NextStepButtonProps> = ({
  currentView,
  onNavigate,
  isEnabled = true,
  completedAction = false,
  contextData
}) => {

  const getNextStepInfo = () => {
    switch (currentView) {
      case 'explore':
        return {
          nextView: 'embeddings_hub',
          title: 'Prétraiter les textes',
          description: 'Vectoriser avec TF-IDF',
          icon: <Zap className="h-5 w-5" />,
          color: 'from-purple-500 to-indigo-600',
          enabled: true
        };
      case 'embeddings_hub':
        return {
          nextView: 'analyze',
          title: 'Analyser les sentiments',
          description: 'Détecter les émotions',
          icon: <Sparkles className="h-5 w-5" />,
          color: 'from-pink-500 to-purple-600',
          enabled: completedAction
        };
      case 'analyze':
        return {
          nextView: 'training',
          title: 'Entraîner un modèle',
          description: 'Autoencoder ou classification',
          icon: <ArrowRight className="h-5 w-5" />,
          color: 'from-orange-500 to-red-600',
          enabled: completedAction
        };
      case 'training':
        return {
          nextView: 'embeddings',
          title: 'Visualiser les résultats',
          description: 'Graphiques et clustering',
          icon: <CheckCircle className="h-5 w-5" />,
          color: 'from-green-500 to-teal-600',
          enabled: completedAction
        };
      default:
        return null;
    }
  };

  const nextStepInfo = getNextStepInfo();

  if (!nextStepInfo) return null;

  const handleClick = () => {
    if (!nextStepInfo.enabled || !isEnabled) return;
    
    // Préparer les données contextuelles à transférer
    let dataToTransfer = contextData;
    
    // Logique spécifique selon la page actuelle
    switch (currentView) {
      case 'explore':
        // Transférer les avis sélectionnés vers le préprocessing
        dataToTransfer = {
          selectedReviews: contextData?.selectedReviews || [],
          datasetInfo: contextData?.datasetInfo || {}
        };
        break;
      case 'embeddings_hub':
        // Transférer les embeddings vers l'analyse
        dataToTransfer = {
          embeddings: contextData?.embeddings || [],
          tfidfModel: contextData?.tfidfModel || null
        };
        break;
      case 'analyze':
        // Transférer les résultats d'analyse vers l'entraînement
        dataToTransfer = {
          analysisResults: contextData?.analysisResults || [],
          sentimentDistribution: contextData?.sentimentDistribution || {}
        };
        break;
      case 'training':
        // Transférer le modèle entraîné vers la visualisation
        dataToTransfer = {
          trainedModel: contextData?.trainedModel || null,
          trainingMetrics: contextData?.trainingMetrics || {}
        };
        break;
    }
    
    onNavigate(nextStepInfo.nextView, dataToTransfer);
  };

  return (
    <div className="fixed bottom-6 right-6 z-50">
      <div className="flex flex-col items-end space-y-3">
        
        {/* Tooltip informatif */}
        {!nextStepInfo.enabled && (
          <div className="bg-yellow-500/20 border border-yellow-500/30 rounded-lg px-4 py-2 max-w-xs">
            <div className="text-yellow-400 text-sm font-medium">
              Complétez une action sur cette page pour débloquer l'étape suivante
            </div>
          </div>
        )}

        {/* Bouton principal */}
        <button
          onClick={handleClick}
          disabled={!nextStepInfo.enabled || !isEnabled}
          className={`
            group relative px-6 py-4 rounded-2xl font-bold text-white transition-all duration-300 transform
            ${nextStepInfo.enabled && isEnabled 
              ? `bg-gradient-to-r ${nextStepInfo.color} hover:scale-105 hover:shadow-2xl shadow-lg` 
              : 'bg-gray-600 cursor-not-allowed opacity-50'
            }
          `}
        >
          <div className="flex items-center space-x-3">
            <div className="flex flex-col items-start">
              <div className="text-sm opacity-90">Étape suivante</div>
              <div className="text-lg font-bold">{nextStepInfo.title}</div>
              <div className="text-xs opacity-75">{nextStepInfo.description}</div>
            </div>
            <div className={`
              transform transition-transform duration-300
              ${nextStepInfo.enabled && isEnabled ? 'group-hover:translate-x-2' : ''}
            `}>
              {nextStepInfo.icon}
            </div>
          </div>

          {/* Effet de brillance */}
          {nextStepInfo.enabled && isEnabled && (
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transform -skew-x-12 group-hover:animate-pulse rounded-2xl"></div>
          )}
        </button>

        {/* Indicateur de progression */}
        {completedAction && (
          <div className="flex items-center space-x-2 bg-green-500/20 border border-green-500/30 rounded-lg px-3 py-2">
            <CheckCircle className="h-4 w-4 text-green-400" />
            <span className="text-green-400 text-sm font-medium">Action terminée !</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default NextStepButton; 