import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Cloud, Sparkles, Zap, Target } from 'lucide-react';

interface WordCloudProps {
  tokens: string[];
  sentiments?: { [key: string]: number };
  model: 'nltk' | 'bert';
}

interface WordData {
  text: string;
  size: number;
  sentiment: number;
  frequency: number;
  x?: number;
  y?: number;
  emotion: string;
}

export const WordCloud: React.FC<WordCloudProps> = ({ tokens, sentiments = {}, model }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [isAnimating, setIsAnimating] = React.useState(false);
  const [selectedWord, setSelectedWord] = React.useState<WordData | null>(null);
  const [animationPhase, setAnimationPhase] = React.useState<'loading' | 'positioning' | 'complete'>('loading');

  // Dictionnaire de sentiments étendu pour le nuage de mots
  const sentimentLexicon: { [key: string]: number } = {
    'amazing': 0.9, 'excellent': 0.8, 'great': 0.6, 'good': 0.5, 'love': 0.7, 'perfect': 0.9,
    'awesome': 0.7, 'fantastic': 0.8, 'wonderful': 0.8, 'brilliant': 0.8, 'outstanding': 0.9,
    'terrible': -0.9, 'awful': -0.8, 'bad': -0.6, 'hate': -0.8, 'horrible': -0.9,
    'disappointing': -0.6, 'poor': -0.5, 'useless': -0.7, 'worst': -0.9, 'pathetic': -0.8,
    'okay': 0.1, 'fine': 0.3, 'decent': 0.4, 'average': 0.0, 'normal': 0.0,
    'fast': 0.4, 'slow': -0.4, 'expensive': -0.3, 'cheap': -0.4, 'quality': 0.4,
    'broken': -0.7, 'working': 0.4, 'easy': 0.5, 'difficult': -0.4, 'smooth': 0.5
  };

  // Émotions associées
  const emotionMap: { [key: string]: string } = {
    'amazing': 'joy', 'love': 'joy', 'happy': 'joy', 'excited': 'joy', 'perfect': 'joy',
    'hate': 'anger', 'angry': 'anger', 'terrible': 'anger', 'awful': 'anger', 'worst': 'anger',
    'scared': 'fear', 'worried': 'fear', 'anxious': 'fear', 'nervous': 'fear',
    'sad': 'sadness', 'disappointed': 'sadness', 'poor': 'sadness', 'bad': 'sadness',
    'surprised': 'surprise', 'amazing': 'surprise', 'incredible': 'surprise',
    'disgusting': 'disgust', 'horrible': 'disgust', 'pathetic': 'disgust'
  };

  useEffect(() => {
    if (!tokens.length || !svgRef.current) return;

    setIsAnimating(true);
    setAnimationPhase('loading');

    // Calculer la fréquence des mots avec filtrage intelligent
    const frequency: { [key: string]: number } = {};
    const processedTokens = tokens.filter(token => {
      const cleanToken = token.toLowerCase().replace('##', '');
      return cleanToken.length > 2 && !['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'way', 'who', 'oil', 'sit', 'set'].includes(cleanToken);
    });

    processedTokens.forEach(token => {
      const cleanToken = token.toLowerCase().replace('##', '');
      frequency[cleanToken] = (frequency[cleanToken] || 0) + 1;
    });

    // Préparer les données avec émotions et sentiments
    const words: WordData[] = Object.entries(frequency)
      .map(([text, freq]) => {
        const sentiment = sentimentLexicon[text] || 0;
        const emotion = emotionMap[text] || 'neutral';
        return {
          text,
          size: Math.max(16, Math.min(64, freq * 12 + 20)),
          sentiment,
          frequency: freq,
          emotion
        };
      })
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 40);

    // Configuration du SVG
    const svg = d3.select(svgRef.current);
    const width = 900;
    const height = 500;
    
    svg.selectAll("*").remove();
    svg.attr("width", width).attr("height", height);

    // Couleurs avancées basées sur le sentiment et le modèle
    const getColor = (sentiment: number, emotion: string) => {
      const emotionColors = {
        joy: '#10B981',      // Vert émeraude
        anger: '#EF4444',    // Rouge
        fear: '#8B5CF6',     // Violet
        sadness: '#3B82F6',  // Bleu
        surprise: '#F59E0B', // Orange
        disgust: '#84CC16',  // Vert lime
        neutral: model === 'bert' ? '#8B5CF6' : '#2563EB'
      };

      if (emotion !== 'neutral') {
        return emotionColors[emotion as keyof typeof emotionColors];
      }

      if (sentiment > 0.4) return '#059669';  // Vert foncé
      if (sentiment < -0.4) return '#DC2626'; // Rouge foncé
      return emotionColors.neutral;
    };

    // Créer un conteneur principal
    const container = svg.append("g")
      .attr("transform", `translate(${width/2},${height/2})`);

    // Simulation de force pour le positionnement
    const simulation = d3.forceSimulation(words as any)
      .force("center", d3.forceCenter(0, 0))
      .force("collision", d3.forceCollide().radius((d: any) => d.size / 2 + 8))
      .force("x", d3.forceX(0).strength(0.05))
      .force("y", d3.forceY(0).strength(0.05))
      .force("charge", d3.forceManyBody().strength(-50));

    // Créer les groupes pour chaque mot
    const wordGroups = container.selectAll("g.word-group")
      .data(words)
      .enter()
      .append("g")
      .attr("class", "word-group")
      .style("cursor", "pointer")
      .style("opacity", 0);

    // Ajouter des cercles de fond avec gradient
    const defs = svg.append("defs");
    
    words.forEach((word, i) => {
      const gradientId = `gradient-${i}`;
      const gradient = defs.append("radialGradient")
        .attr("id", gradientId)
        .attr("cx", "50%")
        .attr("cy", "50%")
        .attr("r", "50%");
      
      gradient.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", getColor(word.sentiment, word.emotion))
        .attr("stop-opacity", 0.3);
      
      gradient.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", getColor(word.sentiment, word.emotion))
        .attr("stop-opacity", 0.1);
    });

    wordGroups.append("circle")
      .attr("r", (d: WordData) => d.size / 2 + 6)
      .attr("fill", (d: WordData, i: number) => `url(#gradient-${i})`)
      .attr("stroke", (d: WordData) => getColor(d.sentiment, d.emotion))
      .attr("stroke-width", 2)
      .attr("stroke-opacity", 0.6);

    // Ajouter le texte avec style avancé
    const texts = wordGroups.append("text")
      .text((d: WordData) => d.text)
      .attr("font-size", (d: WordData) => `${d.size}px`)
      .attr("font-weight", "bold")
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("fill", (d: WordData) => getColor(d.sentiment, d.emotion))
      .attr("filter", "drop-shadow(2px 2px 4px rgba(0,0,0,0.3))")
      .style("font-family", "'Inter', sans-serif");

    // Animation d'apparition séquentielle
    setTimeout(() => {
      setAnimationPhase('positioning');
      
      wordGroups
        .transition()
        .duration(1500)
        .delay((d, i) => i * 100)
        .style("opacity", 1)
        .on("end", (d, i) => {
          if (i === words.length - 1) {
            setAnimationPhase('complete');
            setIsAnimating(false);
          }
        });
    }, 500);

    // Mise à jour des positions pendant la simulation
    simulation.on("tick", () => {
      wordGroups.attr("transform", (d: any) => `translate(${d.x},${d.y})`);
    });

    // Interactions avancées
    wordGroups
      .on("mouseover", function(event, d: WordData) {
        setSelectedWord(d);
        
        d3.select(this)
          .transition()
          .duration(200)
          .style("transform", "scale(1.2)");
        
        d3.select(this).select("text")
          .transition()
          .duration(200)
          .attr("font-size", `${d.size * 1.3}px`);
        
        d3.select(this).select("circle")
          .transition()
          .duration(200)
          .attr("stroke-width", 4)
          .attr("stroke-opacity", 1);
      })
      .on("mouseout", function(event, d: WordData) {
        setSelectedWord(null);
        
        d3.select(this)
          .transition()
          .duration(200)
          .style("transform", "scale(1)");
        
        d3.select(this).select("text")
          .transition()
          .duration(200)
          .attr("font-size", `${d.size}px`);
        
        d3.select(this).select("circle")
          .transition()
          .duration(200)
          .attr("stroke-width", 2)
          .attr("stroke-opacity", 0.6);
      })
      .on("click", function(event, d: WordData) {
        // Animation de clic
        d3.select(this)
          .transition()
          .duration(100)
          .style("transform", "scale(0.9)")
          .transition()
          .duration(100)
          .style("transform", "scale(1.1)")
          .transition()
          .duration(100)
          .style("transform", "scale(1)");
      });

    // Nettoyage
    return () => {
      simulation.stop();
    };
  }, [tokens, sentiments, model]);

  const modelColors = {
    nltk: 'from-blue-600 to-cyan-500',
    bert: 'from-purple-600 to-pink-500'
  };

  const getPhaseIcon = () => {
    switch (animationPhase) {
      case 'loading': return <Zap className="h-5 w-5 animate-pulse" />;
      case 'positioning': return <Target className="h-5 w-5 animate-spin" />;
      case 'complete': return <Sparkles className="h-5 w-5" />;
    }
  };

  const getPhaseText = () => {
    switch (animationPhase) {
      case 'loading': return 'Analyse des mots...';
      case 'positioning': return 'Positionnement intelligent...';
      case 'complete': return 'Nuage de mots généré !';
    }
  };

  return (
    <div className="space-y-6">
      <div className="relative overflow-hidden">
        <div className={`absolute inset-0 bg-gradient-to-br ${modelColors[model]} opacity-10 rounded-2xl`}></div>
        <div className="relative bg-slate-800/90 backdrop-blur-xl p-8 rounded-2xl border border-white/10 shadow-2xl">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-4">
              <div className={`p-3 rounded-xl bg-gradient-to-br ${modelColors[model]} shadow-lg`}>
                <Cloud className="h-6 w-6 text-white" />
              </div>
              <div>
                <h3 className="text-white font-bold text-xl">Nuage de Mots Intelligent</h3>
                <p className="text-white/60">Modèle {model.toUpperCase()} • Analyse émotionnelle</p>
              </div>
            </div>
            
            {isAnimating && (
              <div className="flex items-center space-x-2 text-cyan-400">
                {getPhaseIcon()}
                <span className="text-sm">{getPhaseText()}</span>
              </div>
            )}
          </div>
          
          <div className="bg-slate-900/50 rounded-xl p-6 overflow-hidden relative">
            <svg
              ref={svgRef}
              className="w-full"
              style={{ maxWidth: '100%', height: '500px' }}
            />
            
            {/* Overlay d'information sur le mot sélectionné */}
            {selectedWord && (
              <div className="absolute top-4 right-4 bg-slate-800/95 backdrop-blur-sm p-4 rounded-xl border border-white/20 shadow-xl">
                <div className="text-white font-bold text-lg mb-2">{selectedWord.text}</div>
                <div className="space-y-1 text-sm">
                  <div className="text-white/80">Fréquence: <span className="text-cyan-400 font-bold">{selectedWord.frequency}</span></div>
                  <div className="text-white/80">Sentiment: <span className={`font-bold ${selectedWord.sentiment > 0 ? 'text-green-400' : selectedWord.sentiment < 0 ? 'text-red-400' : 'text-slate-400'}`}>
                    {selectedWord.sentiment.toFixed(2)}
                  </span></div>
                  <div className="text-white/80">Émotion: <span className="text-purple-400 font-bold">{selectedWord.emotion}</span></div>
                </div>
              </div>
            )}
          </div>
          
          {/* Légende améliorée */}
          <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex items-center space-x-3 p-3 bg-green-500/20 rounded-xl border border-green-500/30">
              <div className="w-4 h-4 bg-green-500 rounded-full"></div>
              <div>
                <div className="text-green-400 font-medium text-sm">Positif</div>
                <div className="text-white/60 text-xs">Sentiment &gt; 0.4</div>
              </div>
            </div>
            
            <div className="flex items-center space-x-3 p-3 bg-red-500/20 rounded-xl border border-red-500/30">
              <div className="w-4 h-4 bg-red-500 rounded-full"></div>
              <div>
                <div className="text-red-400 font-medium text-sm">Négatif</div>
                <div className="text-white/60 text-xs">Sentiment &lt; -0.4</div>
              </div>
            </div>
            
            <div className="flex items-center space-x-3 p-3 bg-slate-500/20 rounded-xl border border-slate-500/30">
              <div className={`w-4 h-4 rounded-full ${model === 'bert' ? 'bg-purple-500' : 'bg-blue-500'}`}></div>
              <div>
                <div className="text-slate-400 font-medium text-sm">Neutre</div>
                <div className="text-white/60 text-xs">-0.4 à 0.4</div>
              </div>
            </div>
            
            <div className="flex items-center space-x-3 p-3 bg-orange-500/20 rounded-xl border border-orange-500/30">
              <div className="w-4 h-4 bg-gradient-to-r from-orange-400 to-yellow-400 rounded-full"></div>
              <div>
                <div className="text-orange-400 font-medium text-sm">Émotions</div>
                <div className="text-white/60 text-xs">Couleurs variées</div>
              </div>
            </div>
          </div>
          
          <div className="mt-4 text-center text-white/60 text-sm">
            <span className="font-medium">Taille</span> = Fréquence • <span className="font-medium">Couleur</span> = Sentiment & Émotion • <span className="font-medium">Survolez</span> pour les détails
          </div>
        </div>
      </div>
    </div>
  );
};