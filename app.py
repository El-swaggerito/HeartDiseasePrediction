from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from deap import base, creator, tools
import joblib

app = Flask(__name__)

# Load dataset and preprocess data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]
data = pd.read_csv(url, names=columns, na_values='?')
data = data.dropna()

X = data.drop(columns='target').values
y = data['target'].values

# Feature selection using PSO
def evaluate(individual, X, y):
    mask = np.array(individual, dtype=bool)
    if not np.any(mask):
        return 0,  
    clf = DecisionTreeClassifier()
    scores = cross_val_score(clf, X[:, mask], y, cv=5)
    return scores.mean(),

def update_particle(particle, best):
    for i in range(len(particle)):
        particle.speed[i] = np.random.uniform(-1, 1) * (particle.best[i] - particle[i]) + \
                            np.random.uniform(-1, 1) * (best[i] - particle[i])
        particle[i] = np.clip(particle[i] + particle.speed[i], 0, 1)
        particle[i] = int(particle[i] > 0.5)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=None, smin=None, smax=None, best=None)

def init_particle(pcls, size):
    part = pcls([np.random.randint(2) for _ in range(size)])
    part.speed = [np.random.uniform(-1, 1) for _ in range(size)]
    return part

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 2)
toolbox.register("particle", init_particle, creator.Particle, size=len(X[0]))
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("evaluate", evaluate, X=X, y=y)
toolbox.register("update", update_particle)

def PSO_feature_selection(X, y, n_particles=30, n_iterations=100):
    pop = toolbox.population(n=n_particles)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    best = None
    for g in range(n_iterations):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values

        for part in pop:
            toolbox.update(part, best)

    return best

best_particle = PSO_feature_selection(X, y)
selected_features = np.array(best_particle, dtype=bool)

# Train and save the model
X_selected = X[:, selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

joblib.dump((clf, selected_features), 'heart_disease_model.pkl')

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[feature]) for feature in [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
            'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]]
        model, selected_features = joblib.load('heart_disease_model.pkl')
        user_input = np.array(features).reshape(1, -1)
        user_input_selected = user_input[:, selected_features]

        # Get the probabilities of the prediction
        proba = model.predict_proba(user_input_selected)
        threshold = 0.5  # Adjust this threshold as needed
        prediction = (proba[:, 1] >= threshold).astype(int)

        # Print probabilities for debugging
        print(f"Probabilities: {proba}")

        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"
        return jsonify({'result': result})
    except ValueError:
        return jsonify({'error': 'Invalid input. Please enter valid numbers.'})



if __name__ == '__main__':
    app.run(debug=True)


