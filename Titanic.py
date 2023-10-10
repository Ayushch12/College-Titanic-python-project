import pandas as pd
import matplotlib.pyplot as plt

# Étape 1 : Récupération des données

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv" # Récupérez les données à partir de l'URL fournie.
def request_data(url):
    data = pd.read_csv(url) # Utilisez des pandas pour lire les données directement depuis l'URL
    return data

titanic_data = request_data(url)

# Étape 2 : Création du modèle de données

def extract_model(data):
    passenger_data = {
        "sex": data["Sex"],
        "class": data["Pclass"],
        "age": data["Age"],
        "survived": data["Survived"],
        "price": data["Fare"],
        "embarked": data["Embarked"]
    }
    return passenger_data

model_data = extract_model(titanic_data)
print(model_data)

# Étape 3 : Nettoyage et formatage des données
def transform_data(data):
    data = data.dropna()
    data['age'] = data['Age'].astype(int)  # Utilisez "Age" au lieu de "age"
    data = data[data['age'] > 0]
    return data

cleaned_data = transform_data(titanic_data)
print(cleaned_data)

# Étape 4 : Enregistrement des données
def load_data(data, output_file):
    data.to_csv(output_file, index=False)

load_data(cleaned_data, "titanic_cleaned.csv")

# Étape 5 : Analyse des données

# Question 1 : Combien de femmes de moins de 18 ans ont survécu ?
survived_females_under_18 = cleaned_data[(cleaned_data['Sex'] == 'female') & (cleaned_data['Age'] < 18) & (cleaned_data['Survived'] == 1)]
count_survived_females_under_18 = len(survived_females_under_18)
print(f"Nombre de femmes de moins de 18 ans ayant survécu : {count_survived_females_under_18}")

# Question 2 : Répartition par classe parmi ces femmes
class_distribution = survived_females_under_18['Pclass'].value_counts()
print("Répartition par classe parmi les femmes de moins de 18 ans ayant survécu :")
print(class_distribution)

# Question 3 : Influence du port d'embarquement sur la survie
port_survival_distribution = cleaned_data.groupby('Embarked')['Survived'].value_counts()
print("Répartition des morts et des survivants en fonction du port de départ :")
print(port_survival_distribution)

# Question 4 : Répartition par sexe et par âge des passagers
plt.figure(figsize=(10, 6))
plt.hist(cleaned_data[cleaned_data['Sex'] == 'male']['Age'], bins=20, alpha=0.5, label='Hommes')
plt.hist(cleaned_data[cleaned_data['Sex'] == 'female']['Age'], bins=20, alpha=0.5, label='Femmes')
plt.xlabel('Âge')
plt.ylabel('Nombre de passagers')
plt.legend()
plt.title('Répartition par sexe et par âge des passagers')
plt.show()

# Étape 6 : Conclusion et documentation (est dans le fichier pdf Ecole.pdf)
# Rédigez une conclusion basée sur les résultats obtenus et documentez le pipeline DataOps.
