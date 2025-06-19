import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, Area, AreaChart } from 'recharts';
import { TrendingUp, BarChart3, PieChart as PieChartIcon, Activity } from 'lucide-react';

interface ChartData {
  step: string;
  tokens: number;
  removed: number;
  efficiency: number;
  sentiment?: number;
}

interface TokenFrequency {
  token: string;
  frequency: number;
  sentiment: number;
}

interface InteractiveChartsProps {
  pipelineResults: any[];
  finalTokens: string[];
  model: 'nltk' | 'bert';
}

export const InteractiveCharts: React.FC<InteractiveChartsProps> = ({ 
  pipelineResults, 
  finalTokens, 
  model 
}) => {
  const [activeChart, setActiveChart] = React.useState<'pipeline' | 'frequency' | 'sentiment' | 'efficiency'>('pipeline');

  // Préparation des données pour les graphiques
  const pipelineData: ChartData[] = pipelineResults.map((result, index) => ({
    step: `Étape ${index + 1}`,
    tokens: result.stats.tokensCount || result.stats.outputLength,
    removed: result.stats.removedCount,
    efficiency: result.stats.inputLength > 0 ? 
      Math.round(((result.stats.tokensCount || result.stats.outputLength) / result.stats.inputLength) * 100) : 100,
    sentiment: Math.random() * 2 - 1 // Simulation pour la démo
  }));

  // Calcul de la fréquence des tokens
  const tokenFrequency: TokenFrequency[] = React.useMemo(() => {
    const frequency: { [key: string]: number } = {};
    const sentimentScores: { [key: string]: number } = {
      'amazing': 0.8, 'excellent': 0.7, 'great': 0.6, 'good': 0.5, 'love': 0.7,
      'terrible': -0.8, 'awful': -0.7, 'bad': -0.6, 'hate': -0.7, 'poor': -0.5,
      'product': 0.1, 'quality': 0.3, 'price': 0.0, 'shipping': 0.2, 'fast': 0.4
    };

    finalTokens.forEach(token => {
      const cleanToken = token.toLowerCase().replace('##', '');
      frequency[cleanToken] = (frequency[cleanToken] || 0) + 1;
    });

    return Object.entries(frequency)
      .map(([token, freq]) => ({
        token,
        frequency: freq,
        sentiment: sentimentScores[token] || 0
      }))
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 15);
  }, [finalTokens]);

  const modelColors = {
    nltk: {
      primary: '#3B82F6',
      secondary: '#60A5FA',
      accent: '#93C5FD'
    },
    bert: {
      primary: '#8B5CF6',
      secondary: '#A78BFA',
      accent: '#C4B5FD'
    }
  };

  const colors = modelColors[model];

  const chartTabs = [
    { id: 'pipeline', label: 'Pipeline', icon: Activity },
    { id: 'frequency', label: 'Fréquence', icon: BarChart3 },
    { id: 'sentiment', label: 'Sentiment', icon: TrendingUp },
    { id: 'efficiency', label: 'Efficacité', icon: PieChartIcon }
  ];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-800 p-4 rounded-lg border border-white/20 shadow-xl">
          <p className="text-white font-medium">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value}
              {entry.name === 'efficiency' && '%'}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 p-6 rounded-xl border border-white/20">
        <h3 className="text-white font-bold text-xl mb-6 flex items-center">
          <BarChart3 className="h-6 w-6 text-cyan-400 mr-3" />
          Visualisations Interactives - {model.toUpperCase()}
        </h3>

        {/* Onglets */}
        <div className="flex flex-wrap gap-2 mb-6">
          {chartTabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveChart(tab.id as any)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                activeChart === tab.id
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                  : 'bg-white/10 text-white/70 hover:text-white hover:bg-white/20'
              }`}
            >
              <tab.icon className="h-4 w-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Graphique Pipeline */}
        {activeChart === 'pipeline' && (
          <div className="h-80">
            <h4 className="text-white font-medium mb-4">Évolution du Pipeline</h4>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={pipelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="step" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip content={<CustomTooltip />} />
                <Area 
                  type="monotone" 
                  dataKey="tokens" 
                  stackId="1"
                  stroke={colors.primary} 
                  fill={colors.primary}
                  fillOpacity={0.6}
                  name="Tokens"
                />
                <Area 
                  type="monotone" 
                  dataKey="removed" 
                  stackId="2"
                  stroke="#EF4444" 
                  fill="#EF4444"
                  fillOpacity={0.4}
                  name="Supprimés"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Graphique Fréquence */}
        {activeChart === 'frequency' && (
          <div className="h-80">
            <h4 className="text-white font-medium mb-4">Fréquence des Tokens</h4>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={tokenFrequency} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis type="number" stroke="#9CA3AF" />
                <YAxis dataKey="token" type="category" stroke="#9CA3AF" width={80} />
                <Tooltip content={<CustomTooltip />} />
                <Bar 
                  dataKey="frequency" 
                  fill={colors.primary}
                  name="Fréquence"
                  radius={[0, 4, 4, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Graphique Sentiment */}
        {activeChart === 'sentiment' && (
          <div className="h-80">
            <h4 className="text-white font-medium mb-4">Évolution du Sentiment</h4>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={pipelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="step" stroke="#9CA3AF" />
                <YAxis domain={[-1, 1]} stroke="#9CA3AF" />
                <Tooltip content={<CustomTooltip />} />
                <Line 
                  type="monotone" 
                  dataKey="sentiment" 
                  stroke={colors.primary}
                  strokeWidth={3}
                  dot={{ fill: colors.primary, strokeWidth: 2, r: 6 }}
                  name="Score Sentiment"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Graphique Efficacité */}
        {activeChart === 'efficiency' && (
          <div className="h-80">
            <h4 className="text-white font-medium mb-4">Efficacité par Étape</h4>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pipelineData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ step, efficiency }) => `${step}: ${efficiency}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="efficiency"
                >
                  {pipelineData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={`hsl(${200 + index * 40}, 70%, ${50 + index * 10}%)`} 
                    />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Statistiques rapides */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-r from-blue-500/20 to-cyan-500/20 p-4 rounded-lg border border-blue-500/30">
          <div className="text-2xl font-bold text-blue-400">
            {pipelineData[pipelineData.length - 1]?.tokens || 0}
          </div>
          <div className="text-white/70 text-sm">Tokens Finaux</div>
        </div>
        
        <div className="bg-gradient-to-r from-green-500/20 to-emerald-500/20 p-4 rounded-lg border border-green-500/30">
          <div className="text-2xl font-bold text-green-400">
            {tokenFrequency.length}
          </div>
          <div className="text-white/70 text-sm">Tokens Uniques</div>
        </div>
        
        <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 p-4 rounded-lg border border-purple-500/30">
          <div className="text-2xl font-bold text-purple-400">
            {pipelineData.reduce((acc, curr) => acc + curr.removed, 0)}
          </div>
          <div className="text-white/70 text-sm">Total Supprimés</div>
        </div>
        
        <div className="bg-gradient-to-r from-yellow-500/20 to-orange-500/20 p-4 rounded-lg border border-yellow-500/30">
          <div className="text-2xl font-bold text-yellow-400">
            {Math.round(pipelineData.reduce((acc, curr) => acc + curr.efficiency, 0) / pipelineData.length)}%
          </div>
          <div className="text-white/70 text-sm">Efficacité Moyenne</div>
        </div>
      </div>
    </div>
  );
};