import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define column names
col_names = ['AGE', 'CHEST PAIN', 'LUNG_CANCER']

data = pd.read_csv("D:/Study/University/10/AI/Final Project/Lung Cancer dataset.csv", usecols=col_names)

# Replace numerical values with 'YES' and 'NO'
data.replace({1: 0, 2: 1}, inplace=True)
data.replace({'YES': 1, 'NO': 0}, inplace=True)
data.replace({'M': 1, 'F': 0}, inplace=True)

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        # split until stopping conditions are met
        if (
            num_samples >= self.min_samples_split
            and (self.max_depth is None or curr_depth <= self.max_depth)
        ):
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)

            # check if information gain is positive
            if "info_gain" in best_split and best_split["info_gain"] > 0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                # return decision node
                return Node(
                    best_split["feature_index"],
                    best_split["threshold"],
                    left_subtree,
                    right_subtree,
                    best_split.get("info_gain", 0),
                )

        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        if len(parent) == 0 or len(l_child) == 0 or len(r_child) == 0:
            return 0  # Default value when information gain cannot be computed
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

class RandomForestClassifier:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=2, bootstrap_ratio=0.8):
        ''' constructor '''
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.bootstrap_ratio = bootstrap_ratio
        self.trees = []

    def fit(self, X, Y):
        ''' function to train the Random Forest '''
        for _ in range(self.n_trees):
            # Create a bootstrap sample
            bootstrap_indices = np.random.choice(len(X), size=int(self.bootstrap_ratio * len(X)), replace=True)
            X_bootstrap, Y_bootstrap = X[bootstrap_indices], Y[bootstrap_indices]
            
            # Train a decision tree on the bootstrap sample
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            tree.fit(X_bootstrap, Y_bootstrap)
            
            # Add the trained tree to the ensemble
            self.trees.append(tree)

    def predict(self, X):
        ''' function to make predictions using the ensemble '''
        predictions = [self.aggregate_predictions(x) for x in X]
        return predictions

    def aggregate_predictions(self, x):
        ''' function to aggregate predictions from all trees '''
        tree_predictions = [tree.make_prediction(x, tree.root) for tree in self.trees]
        # You can customize the aggregation method (e.g., majority voting for classification)
        # Here, we use the most common prediction
        return max(set(tree_predictions), key=tree_predictions.count)

class GeneticAlgorithmRandomForest:
    def __init__(self, population_size=10, generations=5):
        self.population_size = population_size
        self.generations = generations
        self.best_hyperparameters = None

    def initialize_population(self):
        return [{'n_trees': np.random.choice([10, 50, 100, 200]),
                 'max_depth': np.random.choice([3, 5, 10, 20])} for _ in range(self.population_size)]

    def fitness(self, hyperparameters, X_train, Y_train, X_val, Y_val):
        # Train a RandomForest with the given hyperparameters and evaluate accuracy on the validation set
        n_trees = hyperparameters['n_trees']
        max_depth = hyperparameters['max_depth']

        random_forest = RandomForestClassifier(n_trees=n_trees, max_depth=max_depth)
        random_forest.fit(X_train, Y_train)
        Y_pred = random_forest.predict(X_val)
        
        accuracy = accuracy_score(Y_val, Y_pred) if accuracy_score(Y_val, Y_pred) > 0 else 0
        return accuracy

    def selection(self, population, fitness_scores):
        selected_indices = np.argsort(fitness_scores)[-len(population) // 2:]
        return [population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        crossover_point = np.random.choice(list(parent1.keys()))
        child = {
            'n_trees': parent1['n_trees'] if crossover_point == 'n_trees' else parent2['n_trees'],
            'max_depth': parent1['max_depth'] if crossover_point == 'max_depth' else parent2['max_depth']
        }
        return child

    def mutation(self, child):
        mutation_point = np.random.choice(list(child.keys()))
        if mutation_point == 'n_trees':
            while True:
                new_value = child['n_trees'] + np.random.choice([-5, -2, 2, 5])
                if new_value > 0:
                    child['n_trees'] = new_value
                    break
        elif mutation_point == 'max_depth':
            while True:
                new_value = child['max_depth'] + np.random.choice([-2, -1, 1, 2])
                if new_value > 0:
                    child['max_depth'] = new_value
                    break
        return child

    def genetic_algorithm(self, X_train, Y_train, X_val, Y_val):
        population = self.initialize_population()

        # Initialize populations
        good_population = []
        normal_population = population
        bad_population = []

        for generation in range(self.generations):
            fitness_scores = [self.fitness(h, X_train, Y_train, X_val, Y_val) for h in population]

            # Categorize individuals based on their fitness scores
            good_indices = np.argsort(fitness_scores)[-len(population) // 5:]
            bad_indices = np.argsort(fitness_scores)[:len(population) // 5]
            normal_indices = np.setdiff1d(np.arange(len(population)), np.concatenate([good_indices, bad_indices]))

            good_population += [population[i] for i in good_indices]
            bad_population += [population[i] for i in bad_indices]
            normal_population = [population[i] for i in normal_indices]

            # Apply genetic algorithm to each population
            good_population = self.genetic_algorithm_population(good_population, X_train, Y_train, X_val, Y_val)
            normal_population = self.genetic_algorithm_population(normal_population, X_train, Y_train, X_val, Y_val)
            bad_population = self.genetic_algorithm_population(bad_population, X_train, Y_train, X_val, Y_val)

            # Combine populations
            population = good_population + normal_population + bad_population

        self.best_hyperparameters = max(population, key=lambda h: self.fitness(h, X_train, Y_train, X_val, Y_val))

    def genetic_algorithm_population(self, population, X_train, Y_train, X_val, Y_val):
        # This is a modified genetic algorithm for a specific population
        offspring = []

        # Randomly select a portion of the population to undergo crossover and mutation
        num_parents_to_modify = int(0.8 * len(population)) 
        parents_to_modify = np.random.choice(population, size=num_parents_to_modify, replace=False)
        for parent in population:
            if parent in parents_to_modify and len(parents_to_modify) > 1:
                parent1, parent2 = np.random.choice(parents_to_modify, size=min(2, len(parents_to_modify)), replace=False)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                offspring.append(child)
            else:
                offspring.append(parent)

        return offspring

    def get_best_hyperparameters(self):
        return self.best_hyperparameters

# Example usage:
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

# Create an instance of GeneticAlgorithmRandomForest
genetic_algorithm_rf = GeneticAlgorithmRandomForest(population_size=10, generations=5)

# Run the genetic algorithm to find the best hyperparameters
genetic_algorithm_rf.genetic_algorithm(X_train, Y_train, X_test, Y_test)

# Get the best hyperparameters
best_hyperparameters = genetic_algorithm_rf.get_best_hyperparameters()
print("Best Hyperparameters:", best_hyperparameters)

# Train a RandomForest with the best hyperparameters
best_random_forest = RandomForestClassifier(n_trees=best_hyperparameters['n_trees'],
                                            max_depth=best_hyperparameters['max_depth'])
best_random_forest.fit(X_train, Y_train)

# Make predictions
Y_pred_rf = best_random_forest.predict(X_test)

# Evaluate accuracy
print("Random Forest Accuracy with Best Hyperparameters:", accuracy_score(Y_test, Y_pred_rf))

true_positive = sum((Y_test == 1) & (Y_pred_rf == 1))
true_negative = sum((Y_test == 0) & (Y_pred_rf == 0))
false_positive = sum((Y_test == 0) & (Y_pred_rf == 1))
false_negative = sum((Y_test == 1) & (Y_pred_rf == 0))

# Calculate metrics
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

accuracy = (true_positive + true_negative) / len(Y_test) if len(Y_pred_rf) > 0 else 0

# Print metrics
print("Confusion Matrix:")
print("True Positive:", true_positive)
print("True Negative:", true_negative)
print("False Positive:", false_positive)
print("False Negative:", false_negative)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("Accuracy:", accuracy)

from sklearn.metrics import accuracy_score, confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_rf)
print("Confusion Matrix:")
print(cm)