export interface Review {
  id: number;
  text: string;
  label: number; // 0 = negative, 1 = positive
  sentiment: 'positive' | 'negative';
  title: string;
  rating: number;
  category: string;
  date: string;
  intensity: number;
}

export class DatasetLoader {
  private static readonly DATASET_URL = 'https://datasets-server.huggingface.co/rows?dataset=amazon_polarity&config=amazon_polarity&split=train';
  private static readonly SAMPLE_SIZE = 1000; // Limiter à 1000 avis pour les performances

  static async loadAmazonPolarityDataset(sampleSize: number = this.SAMPLE_SIZE, randomSample: boolean = false): Promise<Review[]> {
    try {
      console.log(`🚀 Chargement du dataset Amazon Polarity COMPLET: ${sampleSize.toLocaleString()} avis...`);
      
      // Essayer de charger depuis le backend avec le dataset complet
      try {
        const response = await fetch('http://localhost:5000/api/dataset/amazon', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            max_samples: sampleSize,
            random_sample: randomSample,
            split: 'all',
            use_full_dataset: true  // Demander explicitement le dataset complet
          }),
        });

        if (response.ok) {
          const data = await response.json();
          console.log(`✅ Dataset COMPLET chargé depuis le backend: ${data.samples?.length || 0} avis`);
          
          if (data.samples && Array.isArray(data.samples)) {
            return data.samples.map((item: any, index: number) => ({
              id: index + 1,
              text: item.text,
              label: item.label === 2 ? 1 : 0, // Convertir 2->1 (positif), 1->0 (négatif)
              sentiment: item.label === 2 ? 'positive' : 'negative',
              title: item.title || `Review #${index + 1}`,
              rating: item.label === 2 ? Math.floor(Math.random() * 2) + 4 : Math.floor(Math.random() * 2) + 1,
              category: this.getRandomCategory(),
              date: this.getRandomDate(),
              intensity: this.calculateIntensity(item.text)
            }));
          }
        } else {
          console.warn('⚠️ Backend non disponible ou dataset complet non téléchargé');
        }
      } catch (backendError) {
        console.warn('⚠️ Erreur backend, fallback vers données simulées:', backendError);
      }
      
      // Fallback vers les données simulées étendues
      console.log('🔄 Utilisation de données simulées étendues (fallback)...');
      return this.generateFallbackData(sampleSize, randomSample);

    } catch (error) {
      console.error('❌ Erreur lors du chargement du dataset:', error);
      return this.generateFallbackData(sampleSize, randomSample);
    }
  }

  private static getRandomCategory(): string {
    const categories = ['Electronics', 'Books', 'Clothing', 'Home & Kitchen', 'Sports', 'Beauty', 'Toys', 'Automotive'];
    return categories[Math.floor(Math.random() * categories.length)];
  }

  private static getRandomDate(): string {
    const start = new Date(2020, 0, 1);
    const end = new Date();
    const randomTime = start.getTime() + Math.random() * (end.getTime() - start.getTime());
    return new Date(randomTime).toISOString().split('T')[0];
  }

  private static calculateIntensity(text: string): number {
    // Analyse simple de l'intensité basée sur des mots-clés
    const strongPositive = ['amazing', 'incredible', 'outstanding', 'excellent', 'fantastic', 'perfect', 'love', 'awesome'];
    const strongNegative = ['terrible', 'horrible', 'awful', 'disgusting', 'pathetic', 'worst', 'hate', 'useless'];
    const moderate = ['good', 'nice', 'okay', 'fine', 'bad', 'poor', 'disappointing'];

    const words = text.toLowerCase().split(/\s+/);
    let intensity = 0;
    let count = 0;

    words.forEach(word => {
      if (strongPositive.includes(word)) {
        intensity += 0.8;
        count++;
      } else if (strongNegative.includes(word)) {
        intensity -= 0.8;
        count++;
      } else if (moderate.includes(word)) {
        intensity += Math.random() > 0.5 ? 0.3 : -0.3;
        count++;
      }
    });

    return count > 0 ? intensity / count : Math.random() * 0.4 - 0.2;
  }

  private static generateFallbackData(sampleSize: number, randomSample: boolean = false): Review[] {
    console.log(`Génération de données de fallback: ${sampleSize} avis...`);
    
    const reviewTemplates = [
      // Positifs
      { text: "This product is absolutely incredible! The quality exceeded all my expectations and the delivery was lightning fast. I'm genuinely amazed by how well-designed and functional it is. Would definitely recommend to anyone!", sentiment: 'positive', rating: 5, intensity: 0.9 },
      { text: "Outstanding quality and exceptional value! I've been using this for months and it still works perfectly. The customer service was also fantastic when I had questions. Love it!", sentiment: 'positive', rating: 5, intensity: 0.85 },
      { text: "Perfect product! Easy to use, beautiful design, and exactly what I needed. The packaging was elegant and everything arrived in pristine condition. Couldn't be happier!", sentiment: 'positive', rating: 5, intensity: 0.88 },
      { text: "Good product overall. It works as described and the price is reasonable. Shipping was a bit slow but the item quality makes up for it. Satisfied with my purchase.", sentiment: 'positive', rating: 4, intensity: 0.5 },
      { text: "Nice quality and decent functionality. It's not perfect but it does what I need it to do. Would consider buying again if needed.", sentiment: 'positive', rating: 4, intensity: 0.45 },
      
      // Négatifs
      { text: "Absolutely terrible product! It broke after just two days of normal use. Complete waste of money and the customer service was unhelpful. Avoid at all costs!", sentiment: 'negative', rating: 1, intensity: -0.92 },
      { text: "Worst purchase I've made in years. The product is completely useless and doesn't work as described. Feels like a scam. Extremely disappointed and angry.", sentiment: 'negative', rating: 1, intensity: -0.88 },
      { text: "Horrible quality and awful experience. The item arrived damaged and when I tried to return it, the process was a nightmare. Never buying from here again!", sentiment: 'negative', rating: 1, intensity: -0.85 },
      { text: "Disappointing purchase. The quality is below what I expected for the price. It works but feels cheap and flimsy. Probably won't buy from this brand again.", sentiment: 'negative', rating: 2, intensity: -0.55 },
      { text: "Not great. Had some issues with functionality and the customer service wasn't very helpful. It's usable but I expected better quality.", sentiment: 'negative', rating: 2, intensity: -0.48 }
    ];

    const categories = ['Electronics', 'Books', 'Clothing', 'Home & Kitchen', 'Sports', 'Beauty', 'Toys', 'Automotive'];
    const fallbackData: Review[] = [];

    for (let i = 0; i < sampleSize; i++) {
      const template = reviewTemplates[Math.floor(Math.random() * reviewTemplates.length)];
      const category = categories[Math.floor(Math.random() * categories.length)];
      
      fallbackData.push({
        id: i + 1,
        text: template.text,
        label: template.sentiment === 'positive' ? 1 : 0,
        sentiment: template.sentiment as 'positive' | 'negative',
        title: `${category} Product Review #${i + 1}`,
        rating: template.rating,
        category: category,
        date: this.getRandomDate(),
        intensity: template.intensity
      });
    }

    // Mélanger si échantillonnage aléatoire demandé
    if (randomSample) {
      for (let i = fallbackData.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [fallbackData[i], fallbackData[j]] = [fallbackData[j], fallbackData[i]];
      }
    }

    return fallbackData;
  }

  // Méthode pour télécharger et sauvegarder le dataset localement
  static async downloadAndSaveDataset(sampleSize: number = 5000): Promise<void> {
    try {
      console.log('Téléchargement du dataset complet...');
      const reviews = await this.loadAmazonPolarityDataset(sampleSize);
      
      // Sauvegarder en JSON
      const jsonData = JSON.stringify(reviews, null, 2);
      
      // Pour le navigateur, on peut utiliser le download
      const blob = new Blob([jsonData], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `amazon_polarity_${sampleSize}_reviews.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      console.log(`Dataset sauvegardé: ${reviews.length} avis`);
    } catch (error) {
      console.error('Erreur lors de la sauvegarde:', error);
    }
  }
} 