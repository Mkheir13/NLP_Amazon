import React from 'react';
import { Cpu, Brain, Zap, Clock, Target, TrendingUp, BarChart3, Layers } from 'lucide-react';

interface ComparisonData {
  nltk: {
    tokens: number;
    removed: number;
    time: number;
    accuracy: number;
    sentiment: number;
  };
  bert: {
    tokens: number;
    removed: number;
    time: number;
    accuracy: number;
    sentiment: number;
  };
}

interface ModelComparisonProps {
  nltkResults: any[];
  bertResults: any[];
  text: string;
}

export const ModelComparison: React.FC<ModelComparisonProps> = ({ 
  nltkResults, 
  bertResults, 
  text 
}) => {
  const [comparisonData, setComparisonData] = React.useState<ComparisonData | null>(null);

  React.useEffect(() => {
    if (nltkResults.length > 0 && bertResults.length > 0) {
      const nltkFinal = nltkResults[nltkResults.length - 1];
      const bertFinal = bertResults[bertResults.length - 1];

      setComparisonData({
        nltk: {
          tokens: nltkFinal.stats.tokensCount,
          removed: nltkResults.reduce((acc, step) => acc + step.stats.removedCount, 0),
          time: 150, // Simulation
          accuracy: 85,
          sentiment: Math.random() * 2 - 1
        },
        bert: {
          tokens: bertFinal.stats.tokensCount,
          removed: bertResults.reduce((acc, step) => acc + step.stats.removedCount, 0),
          time: 450, // Simulation
          accuracy: 92,
          sentiment: Math.random() * 2 - 1
        }
      });
    }
  }, [nltkResults, bertResults]);

  if (!comparisonData) return null;

  const metrics = [
    {
      key: 'tokens',
      label: 'Tokens Finaux',
      icon: Layers,
      format: (value: number) => value.toString(),
      better: 'higher'
    },
    {
      key: 'removed',
      label: 'Éléments Supprimés',
      icon: BarChart3,
      format: (value: number) => value.toString(),
      better: 'context'
    },
    {
      key: 'time',
      label: 'Temps (ms)',
      icon: Clock,
      format: (value: number) => `${value}ms`,
      better: 'lower'
    },
    {
      key: 'accuracy',
      label: 'Précision',
      icon: Target,
      format: (value: number) => `${value}%`,
      better: 'higher'
    }
  ];

  const getBetterModel = (metric: string, nltkValue: number, bertValue: number) => {
    switch (metric) {
      case 'time':
        return nltkValue < bertValue ? 'nltk' : 'bert';
      case 'accuracy':
      case 'tokens':
        return nltkValue > bertValue ? 'nltk' : 'bert';
      default:
        return 'equal';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 p-6 rounded-xl border border-white/20">
        <h3 className="text-white font-bold text-2xl mb-6 text-center">
          🥊 NLTK vs BERT - Comparaison Directe
        </h3>

        {/* Modèles en face à face */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* NLTK */}
          <div className="bg-gradient-to-r from-blue-500/20 to-cyan-500/20 p-6 rounded-xl border border-blue-500/30">
            <div className="flex items-center space-x-3 mb-4">
              <Cpu className="h-8 w-8 text-blue-400" />
              <div>
                <h4 className="text-blue-400 font-bold text-xl">NLTK</h4>
                <p className="text-white/70 text-sm">Approche Traditionnelle</p>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-white/80">Tokens:</span>
                <span className="text-blue-400 font-bold">{comparisonData.nltk.tokens}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-white/80">Supprimés:</span>
                <span className="text-blue-400 font-bold">{comparisonData.nltk.removed}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-white/80">Vitesse:</span>
                <span className="text-green-400 font-bold">{comparisonData.nltk.time}ms</span>
              </div>
              <div className="flex justify-between">
                <span className="text-white/80">Précision:</span>
                <span className="text-blue-400 font-bold">{comparisonData.nltk.accuracy}%</span>
              </div>
            </div>
          </div>

          {/* BERT */}
          <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 p-6 rounded-xl border border-purple-500/30">
            <div className="flex items-center space-x-3 mb-4">
              <Brain className="h-8 w-8 text-purple-400" />
              <div>
                <h4 className="text-purple-400 font-bold text-xl">BERT</h4>
                <p className="text-white/70 text-sm">IA Moderne</p>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-white/80">Tokens:</span>
                <span className="text-purple-400 font-bold">{comparisonData.bert.tokens}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-white/80">Supprimés:</span>
                <span className="text-purple-400 font-bold">{comparisonData.bert.removed}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-white/80">Vitesse:</span>
                <span className="text-yellow-400 font-bold">{comparisonData.bert.time}ms</span>
              </div>
              <div className="flex justify-between">
                <span className="text-white/80">Précision:</span>
                <span className="text-green-400 font-bold">{comparisonData.bert.accuracy}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Comparaison métrique par métrique */}
        <div className="space-y-4">
          <h4 className="text-white font-bold text-lg text-center mb-4">Comparaison Détaillée</h4>
          
          {metrics.map((metric) => {
            const nltkValue = comparisonData.nltk[metric.key as keyof typeof comparisonData.nltk] as number;
            const bertValue = comparisonData.bert[metric.key as keyof typeof comparisonData.bert] as number;
            const winner = getBetterModel(metric.key, nltkValue, bertValue);
            
            return (
              <div key={metric.key} className="bg-white/5 p-4 rounded-lg">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <metric.icon className="h-5 w-5 text-white/70" />
                    <span className="text-white font-medium">{metric.label}</span>
                  </div>
                  
                  {winner !== 'equal' && (
                    <div className={`px-3 py-1 rounded-full text-xs font-bold ${
                      winner === 'nltk' ? 'bg-blue-500/20 text-blue-400' : 'bg-purple-500/20 text-purple-400'
                    }`}>
                      🏆 {winner.toUpperCase()}
                    </div>
                  )}
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className={`p-3 rounded-lg ${
                    winner === 'nltk' ? 'bg-blue-500/20 border border-blue-500/30' : 'bg-white/5'
                  }`}>
                    <div className="text-center">
                      <div className="text-blue-400 font-bold text-lg">
                        {metric.format(nltkValue)}
                      </div>
                      <div className="text-white/60 text-sm">NLTK</div>
                    </div>
                  </div>
                  
                  <div className={`p-3 rounded-lg ${
                    winner === 'bert' ? 'bg-purple-500/20 border border-purple-500/30' : 'bg-white/5'
                  }`}>
                    <div className="text-center">
                      <div className="text-purple-400 font-bold text-lg">
                        {metric.format(bertValue)}
                      </div>
                      <div className="text-white/60 text-sm">BERT</div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Recommandation */}
        <div className="mt-8 p-6 bg-gradient-to-r from-green-500/20 to-blue-500/20 rounded-xl border border-green-500/30">
          <h4 className="text-white font-bold text-lg mb-3 flex items-center">
            <TrendingUp className="h-5 w-5 text-green-400 mr-2" />
            Recommandation
          </h4>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h5 className="text-blue-400 font-medium mb-2">✅ Utilisez NLTK si:</h5>
              <ul className="text-white/80 text-sm space-y-1">
                <li>• Vous privilégiez la vitesse</li>
                <li>• Traitement de gros volumes</li>
                <li>• Ressources limitées</li>
                <li>• Analyse simple et rapide</li>
              </ul>
            </div>
            
            <div>
              <h5 className="text-purple-400 font-medium mb-2">🚀 Utilisez BERT si:</h5>
              <ul className="text-white/80 text-sm space-y-1">
                <li>• Vous privilégiez la précision</li>
                <li>• Analyse contextuelle fine</li>
                <li>• Textes complexes</li>
                <li>• Qualité {'>'} Vitesse</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};