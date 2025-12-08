# Feature configuration with sentences and activations - 80 features across 4 layers
feature_configs = [
    # Layer 0 - Basic linguistic features (20 features)
    {"ID": (0, 0), "Description": "detect nouns", "Top_activations": [
        {"sentence": "The cat sat on the mat", "activations": [0.9, 0.1, 0.2, 0.1, 0.1]},
        {"sentence": "Dogs love playing in parks", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Books contain valuable knowledge", "activations": [0.7, 0.1, 0.3, 0.1, 0.1]}
    ]},
    {"ID": (0, 1), "Description": "detect verbs", "Top_activations": [
        {"sentence": "She runs every morning", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "They swim in the ocean", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "I write code daily", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (0, 2), "Description": "detect adjectives", "Top_activations": [
        {"sentence": "The beautiful sunset amazed us", "activations": [0.1, 0.1, 0.9, 0.1, 0.1]},
        {"sentence": "A small kitten meowed loudly", "activations": [0.2, 0.1, 0.8, 0.1, 0.1]},
        {"sentence": "The old building needs repair", "activations": [0.1, 0.2, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (0, 3), "Description": "detect prepositions", "Top_activations": [
        {"sentence": "The book is on the table", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "She walked through the forest", "activations": [0.1, 0.2, 0.1, 0.8, 0.1]},
        {"sentence": "Birds fly above the clouds", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]},
    {"ID": (0, 4), "Description": "detect articles", "Top_activations": [
        {"sentence": "The dog chased a ball", "activations": [0.1, 0.1, 0.1, 0.1, 0.9]},
        {"sentence": "An apple fell from tree", "activations": [0.1, 0.1, 0.2, 0.1, 0.8]},
        {"sentence": "A student reads the book", "activations": [0.1, 0.2, 0.1, 0.1, 0.7]}
    ]},
    {"ID": (0, 5), "Description": "detect sports", "Top_activations": [
        {"sentence": "I like to play football", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "Basketball is my favorite sport", "activations": [0.2, 0.1, 0.8, 0.1, 0.1]},
        {"sentence": "Tennis match was exciting", "activations": [0.1, 0.2, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (0, 6), "Description": "detect colors", "Top_activations": [
        {"sentence": "The red car is fast", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]},
        {"sentence": "Blue ocean waves crash", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Green grass grows everywhere", "activations": [0.7, 0.1, 0.2, 0.1, 0.1]}
    ]},
    {"ID": (0, 7), "Description": "detect numbers", "Top_activations": [
        {"sentence": "I have five apples today", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "Two cats sleep peacefully", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "Ten birds fly together", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (0, 8), "Description": "detect time words", "Top_activations": [
        {"sentence": "Yesterday was very cold", "activations": [0.1, 0.1, 0.9, 0.1, 0.1]},
        {"sentence": "Tomorrow will be sunny", "activations": [0.1, 0.2, 0.8, 0.1, 0.1]},
        {"sentence": "Today feels quite warm", "activations": [0.2, 0.1, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (0, 9), "Description": "detect weather", "Top_activations": [
        {"sentence": "Rain falls on the roof", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "Snow covers the ground", "activations": [0.1, 0.1, 0.2, 0.8, 0.1]},
        {"sentence": "Sunshine brightens the day", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]},
    {"ID": (0, 10), "Description": "detect animals", "Top_activations": [
        {"sentence": "Elephants are very large", "activations": [0.1, 0.1, 0.1, 0.1, 0.9]},
        {"sentence": "Lions roar in jungle", "activations": [0.1, 0.1, 0.1, 0.2, 0.8]},
        {"sentence": "Fish swim in water", "activations": [0.1, 0.2, 0.1, 0.1, 0.7]}
    ]},
    {"ID": (0, 11), "Description": "detect food", "Top_activations": [
        {"sentence": "Pizza tastes absolutely delicious", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]},
        {"sentence": "Chocolate makes people happy", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Vegetables are very healthy", "activations": [0.7, 0.1, 0.2, 0.1, 0.1]}
    ]},
    {"ID": (0, 12), "Description": "detect family", "Top_activations": [
        {"sentence": "Mother cooks dinner tonight", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "Father reads newspaper daily", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "Sister plays piano beautifully", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (0, 13), "Description": "detect transportation", "Top_activations": [
        {"sentence": "Cars drive on highways", "activations": [0.1, 0.1, 0.9, 0.1, 0.1]},
        {"sentence": "Trains travel long distances", "activations": [0.1, 0.2, 0.8, 0.1, 0.1]},
        {"sentence": "Planes fly above clouds", "activations": [0.2, 0.1, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (0, 14), "Description": "detect clothing", "Top_activations": [
        {"sentence": "Shirts come in sizes", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "Shoes protect our feet", "activations": [0.1, 0.1, 0.2, 0.8, 0.1]},
        {"sentence": "Hats shield from sun", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]},
    {"ID": (0, 15), "Description": "detect emotions", "Top_activations": [
        {"sentence": "Happiness fills my heart", "activations": [0.1, 0.1, 0.1, 0.1, 0.9]},
        {"sentence": "Sadness overwhelms sometimes today", "activations": [0.1, 0.1, 0.1, 0.2, 0.8]},
        {"sentence": "Anger clouds clear judgment", "activations": [0.1, 0.2, 0.1, 0.1, 0.7]}
    ]},
    {"ID": (0, 16), "Description": "detect technology", "Top_activations": [
        {"sentence": "Computers process data quickly", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]},
        {"sentence": "Phones connect people worldwide", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Internet provides vast information", "activations": [0.7, 0.1, 0.2, 0.1, 0.1]}
    ]},
    {"ID": (0, 17), "Description": "detect music", "Top_activations": [
        {"sentence": "Guitar creates beautiful melodies", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "Piano sounds very elegant", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "Drums keep steady rhythm", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (0, 18), "Description": "detect locations", "Top_activations": [
        {"sentence": "Paris is beautiful city", "activations": [0.1, 0.1, 0.9, 0.1, 0.1]},
        {"sentence": "Mountains reach high peaks", "activations": [0.1, 0.2, 0.8, 0.1, 0.1]},
        {"sentence": "Beaches have soft sand", "activations": [0.2, 0.1, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (0, 19), "Description": "detect professions", "Top_activations": [
        {"sentence": "Doctors help sick patients", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "Teachers educate young minds", "activations": [0.1, 0.1, 0.2, 0.8, 0.1]},
        {"sentence": "Engineers build strong bridges", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]},

    # Layer 1 - Semantic patterns (20 features)
    {"ID": (1, 0), "Description": "subject-verb agreement", "Top_activations": [
        {"sentence": "The cat runs quickly", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]},
        {"sentence": "Dogs bark at strangers", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Children play in yard", "activations": [0.7, 0.1, 0.2, 0.1, 0.1]}
    ]},
    {"ID": (1, 1), "Description": "past tense patterns", "Top_activations": [
        {"sentence": "Yesterday I walked home", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "She finished her homework", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "They visited the museum", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (1, 2), "Description": "question formation", "Top_activations": [
        {"sentence": "What time is it", "activations": [0.1, 0.1, 0.9, 0.1, 0.1]},
        {"sentence": "Where are you going", "activations": [0.1, 0.2, 0.8, 0.1, 0.1]},
        {"sentence": "How does this work", "activations": [0.2, 0.1, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (1, 3), "Description": "identify locations", "Top_activations": [
        {"sentence": "Paris is beautiful", "activations": [0.8, 0.1, 0.2, 0.1, 0.1]},
        {"sentence": "New York never sleeps", "activations": [0.7, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Tokyo has great food", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (1, 4), "Description": "comparison structures", "Top_activations": [
        {"sentence": "Bigger than expected today", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "Faster than light speed", "activations": [0.1, 0.1, 0.2, 0.8, 0.1]},
        {"sentence": "Better than before now", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]},
    {"ID": (1, 5), "Description": "possession patterns", "Top_activations": [
        {"sentence": "John's car is red", "activations": [0.1, 0.1, 0.1, 0.1, 0.9]},
        {"sentence": "Mary's book is thick", "activations": [0.1, 0.1, 0.1, 0.2, 0.8]},
        {"sentence": "Dog's tail wags happily", "activations": [0.1, 0.2, 0.1, 0.1, 0.7]}
    ]},
    {"ID": (1, 6), "Description": "conditional statements", "Top_activations": [
        {"sentence": "If it rains tomorrow", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]},
        {"sentence": "When the sun sets", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Unless you hurry up", "activations": [0.7, 0.1, 0.2, 0.1, 0.1]}
    ]},
    {"ID": (1, 7), "Description": "negation patterns", "Top_activations": [
        {"sentence": "I do not understand", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "She cannot come today", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "They will not participate", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (1, 8), "Description": "modal verbs", "Top_activations": [
        {"sentence": "You should try harder", "activations": [0.1, 0.1, 0.9, 0.1, 0.1]},
        {"sentence": "We might go tomorrow", "activations": [0.1, 0.2, 0.8, 0.1, 0.1]},
        {"sentence": "I could help you", "activations": [0.2, 0.1, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (1, 9), "Description": "relative clauses", "Top_activations": [
        {"sentence": "The person who called", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "Book that I read", "activations": [0.1, 0.1, 0.2, 0.8, 0.1]},
        {"sentence": "Place where we met", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]},
    {"ID": (1, 10), "Description": "compound sentences", "Top_activations": [
        {"sentence": "I study and she works", "activations": [0.1, 0.1, 0.1, 0.1, 0.9]},
        {"sentence": "He runs but feels tired", "activations": [0.1, 0.1, 0.1, 0.2, 0.8]},
        {"sentence": "Rain falls or snow comes", "activations": [0.1, 0.2, 0.1, 0.1, 0.7]}
    ]},
    {"ID": (1, 11), "Description": "passive voice", "Top_activations": [
        {"sentence": "The house was built", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]},
        {"sentence": "Dinner is being prepared", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Books were distributed yesterday", "activations": [0.7, 0.1, 0.2, 0.1, 0.1]}
    ]},
    {"ID": (1, 12), "Description": "reflexive pronouns", "Top_activations": [
        {"sentence": "She taught herself piano", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "I hurt myself today", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "They blamed themselves entirely", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (1, 13), "Description": "gerund phrases", "Top_activations": [
        {"sentence": "Swimming is good exercise", "activations": [0.1, 0.1, 0.9, 0.1, 0.1]},
        {"sentence": "Reading improves vocabulary significantly", "activations": [0.1, 0.2, 0.8, 0.1, 0.1]},
        {"sentence": "Writing requires much practice", "activations": [0.2, 0.1, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (1, 14), "Description": "infinitive phrases", "Top_activations": [
        {"sentence": "I want to learn", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "She decided to stay", "activations": [0.1, 0.1, 0.2, 0.8, 0.1]},
        {"sentence": "They plan to travel", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]},
    {"ID": (1, 15), "Description": "reported speech", "Top_activations": [
        {"sentence": "He said he would", "activations": [0.1, 0.1, 0.1, 0.1, 0.9]},
        {"sentence": "She told me about", "activations": [0.1, 0.1, 0.1, 0.2, 0.8]},
        {"sentence": "They mentioned that yesterday", "activations": [0.1, 0.2, 0.1, 0.1, 0.7]}
    ]},
    {"ID": (1, 16), "Description": "parallel structure", "Top_activations": [
        {"sentence": "Reading, writing, and thinking", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]},
        {"sentence": "Fast, efficient, and reliable", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Study hard, work smart", "activations": [0.7, 0.1, 0.2, 0.1, 0.1]}
    ]},
    {"ID": (1, 17), "Description": "emphatic structures", "Top_activations": [
        {"sentence": "It is John who", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "What I need is", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "The thing that matters", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (1, 18), "Description": "temporal sequences", "Top_activations": [
        {"sentence": "First we eat then", "activations": [0.1, 0.1, 0.9, 0.1, 0.1]},
        {"sentence": "After dinner we watch", "activations": [0.1, 0.2, 0.8, 0.1, 0.1]},
        {"sentence": "Before leaving we check", "activations": [0.2, 0.1, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (1, 19), "Description": "causal relationships", "Top_activations": [
        {"sentence": "Because it was raining", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "Since you asked nicely", "activations": [0.1, 0.1, 0.2, 0.8, 0.1]},
        {"sentence": "Therefore we must leave", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]},

    # Layer 2 - Complex semantic understanding (20 features)
    {"ID": (2, 0), "Description": "sentiment analysis", "Top_activations": [
        {"sentence": "This movie is amazing", "activations": [0.1, 0.1, 0.1, 0.1, 0.9]},
        {"sentence": "Terrible weather ruins plans", "activations": [0.1, 0.1, 0.1, 0.2, 0.8]},
        {"sentence": "Neutral stance on topic", "activations": [0.1, 0.2, 0.1, 0.1, 0.7]}
    ]},
    {"ID": (2, 1), "Description": "irony detection", "Top_activations": [
        {"sentence": "Great, another meeting today", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]},
        {"sentence": "Perfect timing for rain", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Just what I needed", "activations": [0.7, 0.1, 0.2, 0.1, 0.1]}
    ]},
    {"ID": (2, 2), "Description": "metaphor understanding", "Top_activations": [
        {"sentence": "Time is money indeed", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "Life is a journey", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "Knowledge is power always", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (2, 3), "Description": "implication detection", "Top_activations": [
        {"sentence": "The door is open", "activations": [0.1, 0.1, 0.9, 0.1, 0.1]},
        {"sentence": "She glanced at watch", "activations": [0.1, 0.2, 0.8, 0.1, 0.1]},
        {"sentence": "He cleared his throat", "activations": [0.2, 0.1, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (2, 4), "Description": "tone recognition", "Top_activations": [
        {"sentence": "Could you please help", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "Do it right now", "activations": [0.1, 0.1, 0.2, 0.8, 0.1]},
        {"sentence": "Maybe we should consider", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]},
    {"ID": (2, 5), "Description": "contradiction detection", "Top_activations": [
        {"sentence": "I never always forget", "activations": [0.1, 0.1, 0.1, 0.1, 0.9]},
        {"sentence": "Definitely maybe tomorrow perhaps", "activations": [0.1, 0.1, 0.1, 0.2, 0.8]},
        {"sentence": "Honestly lying about truth", "activations": [0.1, 0.2, 0.1, 0.1, 0.7]}
    ]},
    {"ID": (2, 6), "Description": "discourse markers", "Top_activations": [
        {"sentence": "However, I think differently", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]},
        {"sentence": "Furthermore, we should consider", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Nevertheless, the plan continues", "activations": [0.7, 0.1, 0.2, 0.1, 0.1]}
    ]},
    {"ID": (2, 7), "Description": "topic coherence", "Top_activations": [
        {"sentence": "Sports cars racing championship", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "Cooking recipes dinner preparation", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "Weather forecast rain predictions", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (2, 8), "Description": "recognize emotions", "Top_activations": [
        {"sentence": "I am very happy today", "activations": [0.1, 0.1, 0.1, 0.1, 0.8]},
        {"sentence": "Feeling sad about this", "activations": [0.1, 0.1, 0.1, 0.2, 0.9]},
        {"sentence": "Excited for the party", "activations": [0.1, 0.1, 0.2, 0.1, 0.7]}
    ]},
    {"ID": (2, 9), "Description": "register detection", "Top_activations": [
        {"sentence": "Pursuant to regulations aforementioned", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "Hey dude, what's up", "activations": [0.1, 0.1, 0.2, 0.8, 0.1]},
        {"sentence": "Good morning, how are", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]},
    {"ID": (2, 10), "Description": "argument structure", "Top_activations": [
        {"sentence": "Evidence suggests that conclusion", "activations": [0.1, 0.1, 0.1, 0.1, 0.9]},
        {"sentence": "Therefore we can conclude", "activations": [0.1, 0.1, 0.1, 0.2, 0.8]},
        {"sentence": "On the other hand", "activations": [0.1, 0.2, 0.1, 0.1, 0.7]}
    ]},
    {"ID": (2, 11), "Description": "narrative perspective", "Top_activations": [
        {"sentence": "I remember when I", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]},
        {"sentence": "She thought to herself", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "The narrator explains that", "activations": [0.7, 0.1, 0.2, 0.1, 0.1]}
    ]},
    {"ID": (2, 12), "Description": "cultural references", "Top_activations": [
        {"sentence": "Like David versus Goliath", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "Opening Pandora's box here", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "Achilles heel of plan", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (2, 13), "Description": "temporal reasoning", "Top_activations": [
        {"sentence": "Before the war ended", "activations": [0.1, 0.1, 0.9, 0.1, 0.1]},
        {"sentence": "During the Renaissance period", "activations": [0.1, 0.2, 0.8, 0.1, 0.1]},
        {"sentence": "After graduation ceremony completed", "activations": [0.2, 0.1, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (2, 14), "Description": "spatial reasoning", "Top_activations": [
        {"sentence": "North of the border", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "Inside the building structure", "activations": [0.1, 0.1, 0.2, 0.8, 0.1]},
        {"sentence": "Beyond the horizon line", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]},
    {"ID": (2, 15), "Description": "modal logic", "Top_activations": [
        {"sentence": "It must be true", "activations": [0.1, 0.1, 0.1, 0.1, 0.9]},
        {"sentence": "Possibly the best option", "activations": [0.1, 0.1, 0.1, 0.2, 0.8]},
        {"sentence": "Certainly a good choice", "activations": [0.1, 0.2, 0.1, 0.1, 0.7]}
    ]},
    {"ID": (2, 16), "Description": "expertise detection", "Top_activations": [
        {"sentence": "According to research studies", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]},
        {"sentence": "In my professional opinion", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Based on extensive experience", "activations": [0.7, 0.1, 0.2, 0.1, 0.1]}
    ]},
    {"ID": (2, 17), "Description": "pragmatic inference", "Top_activations": [
        {"sentence": "Can you pass salt", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "Do you have time", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "Would you mind closing", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (2, 18), "Description": "world knowledge", "Top_activations": [
        {"sentence": "Water boils at temperature", "activations": [0.1, 0.1, 0.9, 0.1, 0.1]},
        {"sentence": "Gravity pulls objects down", "activations": [0.1, 0.2, 0.8, 0.1, 0.1]},
        {"sentence": "Sun rises in east", "activations": [0.2, 0.1, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (2, 19), "Description": "context dependency", "Top_activations": [
        {"sentence": "It depends on situation", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "Given the current circumstances", "activations": [0.1, 0.1, 0.2, 0.8, 0.1]},
        {"sentence": "Under these specific conditions", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]},

    # Layer 3 - High-level reasoning (20 features)
    {"ID": (3, 0), "Description": "logical consistency", "Top_activations": [
        {"sentence": "All birds fly therefore", "activations": [0.1, 0.1, 0.1, 0.1, 0.9]},
        {"sentence": "If premise then conclusion", "activations": [0.1, 0.1, 0.1, 0.2, 0.8]},
        {"sentence": "Either option A or", "activations": [0.1, 0.2, 0.1, 0.1, 0.7]}
    ]},
    {"ID": (3, 1), "Description": "causal reasoning", "Top_activations": [
        {"sentence": "Rain causes wet streets", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]},
        {"sentence": "Study leads to knowledge", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Exercise improves health significantly", "activations": [0.7, 0.1, 0.2, 0.1, 0.1]}
    ]},
    {"ID": (3, 2), "Description": "analogical reasoning", "Top_activations": [
        {"sentence": "Brain is like computer", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "Heart works as pump", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "Memory functions like library", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (3, 3), "Description": "counterfactual thinking", "Top_activations": [
        {"sentence": "If I had studied", "activations": [0.1, 0.1, 0.9, 0.1, 0.1]},
        {"sentence": "Had we left earlier", "activations": [0.1, 0.2, 0.8, 0.1, 0.1]},
        {"sentence": "What if things were", "activations": [0.2, 0.1, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (3, 4), "Description": "intention recognition", "Top_activations": [
        {"sentence": "She packed her bags", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "He bought flowers today", "activations": [0.1, 0.1, 0.2, 0.8, 0.1]},
        {"sentence": "They studied all night", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]},
    {"ID": (3, 5), "Description": "theory of mind", "Top_activations": [
        {"sentence": "She thinks he knows", "activations": [0.1, 0.1, 0.1, 0.1, 0.9]},
        {"sentence": "I believe you understand", "activations": [0.1, 0.1, 0.1, 0.2, 0.8]},
        {"sentence": "They assume we agree", "activations": [0.1, 0.2, 0.1, 0.1, 0.7]}
    ]},
    {"ID": (3, 6), "Description": "moral reasoning", "Top_activations": [
        {"sentence": "It's wrong to lie", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]},
        {"sentence": "Justice requires fairness always", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Help others in need", "activations": [0.7, 0.1, 0.2, 0.1, 0.1]}
    ]},
    {"ID": (3, 7), "Description": "strategic thinking", "Top_activations": [
        {"sentence": "Plan three moves ahead", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "Consider all possible outcomes", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "Anticipate their next move", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (3, 8), "Description": "abstract concepts", "Top_activations": [
        {"sentence": "Freedom requires responsibility always", "activations": [0.1, 0.1, 0.9, 0.1, 0.1]},
        {"sentence": "Justice embodies fairness principle", "activations": [0.1, 0.2, 0.8, 0.1, 0.1]},
        {"sentence": "Love transcends all boundaries", "activations": [0.2, 0.1, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (3, 9), "Description": "social dynamics", "Top_activations": [
        {"sentence": "Group pressure influences decisions", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "Leadership requires trust building", "activations": [0.1, 0.1, 0.2, 0.8, 0.1]},
        {"sentence": "Cooperation benefits everyone involved", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]},
    {"ID": (3, 10), "Description": "meta-cognition", "Top_activations": [
        {"sentence": "I know that I", "activations": [0.1, 0.1, 0.1, 0.1, 0.9]},
        {"sentence": "Thinking about thinking process", "activations": [0.1, 0.1, 0.1, 0.2, 0.8]},
        {"sentence": "Aware of my awareness", "activations": [0.1, 0.2, 0.1, 0.1, 0.7]}
    ]},
    {"ID": (3, 11), "Description": "conceptual blending", "Top_activations": [
        {"sentence": "Computer virus spreads quickly", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]},
        {"sentence": "Information highway connects minds", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Digital footprint leaves traces", "activations": [0.7, 0.1, 0.2, 0.1, 0.1]}
    ]},
    {"ID": (3, 12), "Description": "emergent properties", "Top_activations": [
        {"sentence": "Whole greater than parts", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "Systems exhibit complex behavior", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "Collective intelligence emerges naturally", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (3, 13), "Description": "recursive thinking", "Top_activations": [
        {"sentence": "Problem within problem within", "activations": [0.1, 0.1, 0.9, 0.1, 0.1]},
        {"sentence": "Story about story telling", "activations": [0.1, 0.2, 0.8, 0.1, 0.1]},
        {"sentence": "Dream within a dream", "activations": [0.2, 0.1, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (3, 14), "Description": "paradigm recognition", "Top_activations": [
        {"sentence": "Fundamental shift in thinking", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "Revolutionary approach changes everything", "activations": [0.1, 0.1, 0.2, 0.8, 0.1]},
        {"sentence": "New framework emerges gradually", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]},
    {"ID": (3, 15), "Description": "cross-domain transfer", "Top_activations": [
        {"sentence": "Music theory applies mathematics", "activations": [0.1, 0.1, 0.1, 0.1, 0.9]},
        {"sentence": "Sports psychology improves performance", "activations": [0.1, 0.1, 0.1, 0.2, 0.8]},
        {"sentence": "Biology inspires engineering design", "activations": [0.1, 0.2, 0.1, 0.1, 0.7]}
    ]},
    {"ID": (3, 16), "Description": "systematic thinking", "Top_activations": [
        {"sentence": "Everything connects to everything", "activations": [0.9, 0.1, 0.1, 0.1, 0.1]},
        {"sentence": "Feedback loops create stability", "activations": [0.8, 0.2, 0.1, 0.1, 0.1]},
        {"sentence": "Complex interactions produce patterns", "activations": [0.7, 0.1, 0.2, 0.1, 0.1]}
    ]},
    {"ID": (3, 17), "Description": "philosophical inquiry", "Top_activations": [
        {"sentence": "What is the nature", "activations": [0.1, 0.9, 0.1, 0.1, 0.1]},
        {"sentence": "Why do things exist", "activations": [0.1, 0.8, 0.2, 0.1, 0.1]},
        {"sentence": "How do we know", "activations": [0.2, 0.7, 0.1, 0.1, 0.1]}
    ]},
    {"ID": (3, 18), "Description": "creative synthesis", "Top_activations": [
        {"sentence": "Combining unlikely elements creates", "activations": [0.1, 0.1, 0.9, 0.1, 0.1]},
        {"sentence": "Innovation emerges from fusion", "activations": [0.1, 0.2, 0.8, 0.1, 0.1]},
        {"sentence": "Novel solutions require integration", "activations": [0.2, 0.1, 0.7, 0.1, 0.1]}
    ]},
    {"ID": (3, 19), "Description": "wisdom integration", "Top_activations": [
        {"sentence": "Experience teaches deeper truths", "activations": [0.1, 0.1, 0.1, 0.9, 0.1]},
        {"sentence": "Knowledge becomes understanding through", "activations": [0.1, 0.1, 0.2, 0.8, 0.1]},
        {"sentence": "Wisdom emerges from reflection", "activations": [0.2, 0.1, 0.1, 0.7, 0.1]}
    ]}
]
