import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from PIL import Image

#st.set_page_config(page_title="TennisBetPy", page_icon=None, initial_sidebar_state="auto", menu_items=None)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

pages = ["Introduction",
"Exploration du jeu de données",
"Nettoyage des données",
"Visualisation du dataset",
"Data Processing",
"Modèles de Machine Learning",
"Conclusion"]
page = st.sidebar.radio("Menu", options = pages)

df = pd.read_csv("C:/Users/Mdrouge/Downloads/atp_data.csv")
st.session_state['df'] = df

if page == pages[0]:
    df = st.session_state['df']
    st.title("Projet Beat the bookmaker - TennisBetPy :tennis:")

    image = Image.open('bg_tennis.jpg')
    st.image(image)#, caption='')
    st.header(pages[0])
    st.markdown("""---""")


    st.markdown("""
        Nous vous présentons dans ce rapport l’étude qui nous a été proposée de réaliser pendant notre formation. Il s’agit du projet “Beat the Bookmaker”. Le sujet nous laissait le choix entre deux univers sportifs : le football ou le tennis. Nordine et moi-même ayant une préférence pour le tennis, et devant prendre une décision pendant Rolland Garros, notre choix s’est donc facilement porté sur le Tennis.
    """)
    st.markdown("""
        Présenté simplement, l’objectif de notre étude est de “battre” les bookmakers.""")
    st.markdown("""
        On peut voir plusieurs axes de recherche pour obtenir de meilleurs résultats que les bookmakers : <ul style="list-style-type:none">
            <li> :one: prédire avec un meilleur taux de prédiction que le bookmaker, le joueur qui remporte la rencontre. Il va s’agir pour cela de déterminer un modèle de machine learning capable de surpasser les performances des bookmakers concernant le gagnant de chaque match</li>
            <li> :two: déterminer la côte optimale de chaque match pour avoir la plus grande rentabilité en terme de gain</li>
        </ul>
    """, unsafe_allow_html=True)
    st.markdown("""
        Par manque de temps et afin de rendre un travail de qualité, nous avons fait le choix de concentrer notre étude sur le premier axe : <b style="color:#D66700">créer et optimiser un modèle de machine learning capable de prédire l’issue d’un match.</b> Nous n’étudierons pas la partie “côte” pour rentabiliser notre modèle par rapport à ceux des bookmakers.
        Commençons donc par l’exploration de notre jeux de données.
    """, unsafe_allow_html=True)
   
if page == pages[1]:
    st.header(pages[1])
    st.markdown("""---""")

    st.markdown("""
        Pour notre projet nous nous appuyons sur le dataset disponible sur Kaggle : https://www.kaggle.com/code/edouardthomas/beat-the-bookmakers-with-machine-learning-tennis/data?select=atp_data.csv.
        Ce dataset est tiré des données du site tennis.co.uk , listant tous les matchs ayant eu lieu entre les années 2000 et 2018.
    """)
    st.markdown("""
        Afin d’exploiter celui-ci, importons pandas ainsi que les autres bibliothèques nécessaires pour bien démarrer notre exploration de données : pandas, numpy, matplotlib ainsi que seaborn.
        Affichons les premières lignes de notre dataset avec pandas.read_csv  :
    """)

    st.code("""
        %matplotlib inline\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns
    """)
    st.code("df = pd.read_csv('atp_data.csv')")

    st.write(df)

    st.subheader("Composition du dataset")

    st.markdown("""
        Il est indispensable avant tout d’étudier notre dataset ; chaque ligne représente un match entre deux joueurs faisant partie du classement ATP.
    """)

    st.code("""
    df.info()
    """)
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("Définition des colonnes")
    st.markdown("""
        La description des colonnes est disponible sur Kaggle : https://www.kaggle.com/code/edouardthomas/beat-the-bookmakers-with-machine-learning-tennis/data?select=atp_data.csv
    """)



if page == pages[2]:
    st.header("Nettoyage des données")
    st.markdown("""---""")
    st.subheader("Gestion des valeurs manquantes , NA et doublons")
    st.markdown("Nous voulons nous assurer dans un premier temps que notre datasets est 'propre'. Regardons quelle est la proportion de valeurs manquantes par colonne :")



    st.code("df.isna().sum().sort_values(ascending=False)")
    st.write(df.isna().sum().sort_values(ascending=False))

    st.markdown("""
        Nous pouvons supprimer certaines colonnes qui ne sont pas pertinentes pour le modèle de machine learning que nous voulons définir.
        Les colonnes PSW, PSL, B365W et B365L correspondent aux côtes établies par les bookmakers. Nous n’en avons pas besoin.
        De même, nous enlevons les colonnes proba_elo, elo_winner et elo_loser qui sont des colonnes calculées. La colonne proba_elo est une probabilité définie par le créateur du dataset, assez difficile à interpréter que nous ne détaillerons pas dans ce document.
        Si vous souhaitez avoir plus d’information sur la formule utilisée, je vous invite à visiter la page : https://github.com/edouardthom/ATPBetting/blob/master/Python/elo_features.py Les colonnes Wsets et Lsets doivent être retirées également car elles correspondent à des informations d’après-match.
    """, unsafe_allow_html=True)


    st.code("""
    df = df.drop(['PSW', 'PSL', 'B365W', 'B365L', 'proba_elo', 'elo_winner', 'elo_loser', 'Wsets', 'Lsets'], axis=1)
    """)
    st.markdown("""Il n’y a donc plus de valeur manquante dans notre jeux de données. Vérifions qu’il n’existe pas de lignes en doublons :""")
    st.code("""
    print(df.duplicated().sum())
    """)
    st.write("0")
    st.markdown("""Notre jeux de données ne comporte désormais plus de valeurs manquantes, ni de doublons.""")

if page == pages[3]:
    st.header("Nettoyage des données")
    st.markdown("""---""")
    st.subheader("Visualisation du dataset")

    st.markdown("""Affichons le nombre de matchs joués par les 10 joueurs ayant gagné le plus de matchs :""")
    st.code("""
    #Top 10 des joueurs ayant gagnés le plus de matchs\ntop_winner = df["Winner"].value_counts().head(10)\nsns.barplot(x = top_winner.values, y=top_winner.index).set(title='Top 10 des joueurs ayant gagnés le plus de matchs');
    """)
 
    top_winner = df["Winner"].value_counts().head(10)
    
    fig, ax = plt.subplots()
    ax = sns.barplot(x = top_winner.values, y=top_winner.index).set(title='Top 10 des joueurs ayant gagnés le plus de matchs');

    st.pyplot(fig)


    st.markdown("""On retrouve les noms des joueurs bien connus du circuit.""")

    st.markdown("""Attention, ce top 10 ne représente pas le top 10 du classement 2018. Nous pourrions vouloir afficher le top 10 des meilleurs joueurs du classement ; cela n’aurait pas de sens dans la mesure où le classement n’est pas figé. Il évolue à chaque match et est présent dans la colonne WRank.""")
    st.markdown("""En effet, on peut voir par exemple que Federer a été classé 65e dans les années 2000 :""")
    st.caption('Plus précisement, il est passé de 65ème fin de saison 1999 à 29ème fin de saison 2000.')

    st.code("""
    df[df['Winner'] == 'Federer R.'][['WRank','Date']])
    """)
    df[df['Winner'] == 'Federer R.'][['WRank','Date']]

    st.code("""
    #Top 10 des joueurs ayant PERDU le plus de match:\ntop_looser = df["Loser"].value_counts().head(10)\nsns.barplot(x = top_looser.values, y=top_looser.index).set(title='Nombre de matchs perdu par joueur du Bottom 10');
    """)
    top_looser = df["Loser"].value_counts().head(10)
    fig, ax = plt.subplots()
    ax = sns.barplot(x = top_looser.values, y=top_looser.index).set(title='Nombre de matchs perdu par joueur du Bottom 10')
    st.pyplot(fig)
    st.markdown("""Le graphique ci-dessus présente, au contraire, le top 10 des joueurs cumulant le plus de défaites sur le circuit à cette période (2000 - 2018). Nous retrouvons ici aussi des joueurs connus. Au global, 1402 joueurs sont représentés dans le jeux de données :""")
    
    st.code("""
    dfC = pd.concat([df['Winner'], df['Loser']], axis = 0).drop_duplicates()\ndfC
    """)
    
    dfC = pd.concat([df['Winner'], df['Loser']], axis = 0).drop_duplicates()
    dfC

    st.code("""
    sns.displot(x="WRank", data=df, kde=True).set(title='Concentration du nombre de match gagnés par classement');
    plt.xlim(0, 500)
    """)
    
    st.image("https://drive.google.com/uc?id=1hNDL8po_8VPuM9rx_0oay3FZE3KyWT6e")
    
    st.markdown("""
        Les 200 premiers joueurs du classement se partagent quasiment tous les matchs gagnés.\nUn autre graphique de type countplot nous renseigne sur le nombre de matchs par type de surface. Pour les non-initiés :<ul style="list-style-type:none">
            <li><div style="color:blue">Hard : surface dure</div></li>
            <li><div style="color:orange">Clay : terre battue</div></li>
            <li><div style="color:green">Grass : gazon</div></li>
            <li><div style="color:purple">Carpet : moquette (plus utilisé depuis 2009)</div></li>
        </ul>
    """, unsafe_allow_html=True)

    #ax = sns.countplot(x="Surface", data=df).set(title='Nombre de match par type de surface');
    #fig, ax = plt.subplots()
    #st.pyplot(fig)

    st.image('data:image/png;base64,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UVJNo9Fg/fr1ZncBfvDBBw0OPRkMBrz55pvw9fXFsGHDUF5ejnnz5knb8cgjjyAwMBDTp0+/5YeexsTHx2PdunXQarVISUlBaGioVHN3d8f777+PtLQ0+Pj4YNSoUdL1tjs5ttsqhRD84jOili45ORn5+fl/GL6SW21tLQYPHoydO3eiR48e0vQb1z7mzp1r0X6odeIZDRHd0ieffII///nPZiFDdKd4MwARNSgwMBBCCKSkpFi7FWrlOHRGRESy4tAZERHJikFDRESyYtAQEZGseDPALVRU1MBk4uUrIqKmUCoVcHLq0GCNQXMLJpNg0BARNQMOnRERkawYNEREJCsGDRERyYpBQ0REsmLQEBGRrBg0REQkKwYNERHJiv+P5i486NAO7exsrd1Gi1Crr0NVZe3tZySi+xaD5i60s7NF9MLN1m6jRdiyZhKqwKAholvj0BkREcmKQUNERLJi0BARkawYNEREJCsGDRERyYpBQ0REsmLQEBGRrBg0REQkKwYNERHJikFDRESyYtAQEZGsGDRERCQrBg0REcmKQUNERLJi0BARkawYNEREJCsGDRERyYpBQ0REsmLQEBGRrBg0REQkKwYNERHJikFDRESyYtAQEZGsGDRERCQrBg0REcmKQUNERLJi0BARkawYNEREJCuLBE1FRQVmzJiB4OBgRERE4IUXXkB5eTkA4MiRI4iMjERwcDCmT5+OsrIy6XVy1IiIyLIsEjQKhQLPPPMM9u7di4yMDDz88MNISkqCyWTCggULEBcXh71790Kn0yEpKQkAZKkREZHlWSRoHB0d4evrK/3u7e2NwsJCHD9+HHZ2dtDpdACAiRMnYs+ePQAgS42IiCxPZekVmkwmfPLJJwgMDERRURHc3d2lmrOzM0wmEy5fvixLzdHRscl9urjY3+OW3j9cXR+0dgtE1IJZPGiWL1+OBx54ADExMfjqq68svfomKyurhskkGqzxjdXcpUtV1m6BiKxMqVTc8gO6RYMmMTER+fn5SE1NhVKphEajQWFhoVQvLy+HUqmEo6OjLDUiIrI8i93e/NZbb+H48eNISUmBWq0GAHh5eaG2thY5OTkAgK1btyIkJES2GhERWZ5CCNHw+FAzys3NRXh4OHr06IF27doBAB566CGkpKTg8OHDiI+Ph16vR9euXbF27Vp06tQJAGSpNdXths6iF26+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')
    
if page == pages[4]:
    st.header("Data processing")
    st.markdown("""---""")

    st.markdown("Notre dataset actuel :")
    df=pd.read_csv("C:/Users/Mdrouge/Downloads/df_dataprocessing2.csv")
    st.code("""df""")
    #df = df.drop("Unnamed: 0")
    df
    st.markdown("""
        Optimisons notre dataset avant d’implémenter des algorithmes de machine learning. Maintenant que nous avons fait ressortir quelques graphiques pertinents, nous pouvons supprimer les colonnes Tournament, Series, Court, Surface, qui sont résumées dans la colonne ATP.
    """)
    st.code("""df = df.drop(['Location', 'Tournament','Court','Surface','Series','Best of'], axis=1)""")
    df = df.drop(['Location', 'Tournament','Court','Surface','Series','Best of'], axis=1)
    st.caption("NB : La matrice de corrélation suivante (qui a été réalisée en amont) montre aisément le lien entre la colonne ATP et les colonnes Location, Tournament, Court, Surface, Series, Best of.")
    st.image("https://drive.google.com/uc?id=1guoEJ-xRHPsHnE5VAJMJoS2oIGx5JGHo")

    st.markdown("""L'année semble être la variable la plus corrélée au reste de nos variables explicatives.""")

    st.markdown("""On s'assure qu'il n'y a pas de nouveau de doublons, après la suppressions de colonnes.""")

    st.code("print(df.duplicated().sum())")
    print(df.duplicated().sum())
    st.markdown("""La colonne 'Comment' renseigne de l’issue du match :""")
    st.code("df['Comment'].unique()")
    df['Comment'].unique()

    st.markdown("""Pour notre modèle de machine learning, nous ne garderons que les matchs ayant aboutis (Completed).""")
    st.code("""#On ne garde que les matchs ayant aboutis\ndf = df[df['Comment']=="Completed"]\n#On peut donc enlever cette colonne Comment\ndf = df.drop(columns=['Comment'], axis=1)\ndf""")
    #On ne garde que les matchs ayant aboutis
    df = df[df['Comment']=="Completed"]
    #On peut donc enlever cette colonne Comment
    df = df.drop(columns=['Comment'], axis=1)
    df

    st.markdown("""La colonne Date doit être modifiée pour que notre modèle avenir fonctionne convenablement : de celle-ci, créons les colonnes Day, Month et Year.""")
    st.markdown("""Ensuite, remplaçons les valeurs de la variable catégorielle Round par des valeurs numériques.""")
    st.code("""# On divise la colonne date en jour, mois, annee\ndf['Date'] = pd.to_datetime(df['Date'])\ndf['Day'] = df['Date'].dt.day\ndf['Month'] = df['Date'].dt.month\ndf['Year'] = df['Date'].dt.year\ndf = df.drop('Date', axis=1)\ndf""")
    # On divise la colonne date en jour, mois, annee
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df = df.drop('Date', axis=1)
    df
    
    st.code("""df = df.replace({'Round Robin':0,"1st Round" : 1,"2nd Round":2,'3rd Round':3,'4th Round':4,'Quarterfinals':5,'Semifinals':6,'The Final':7})""")
    df = df.replace({'Round Robin':0,"1st Round" : 1,"2nd Round":2,'3rd Round':3,'4th Round':4,'Quarterfinals':5,'Semifinals':6,'The Final':7})
    df
    #st.code("""df.info()""")
    #df.info()
    

    st.markdown("""Les colonnes Winner et Loser sont les deux colonnes restantes qui ne sont pas au format numérique. Pour rappel, l’objectif de notre modèle est de prédire le gagnant d’un match. Pour ce faire, nous allons dédoubler chaque ligne de notre dataset. On crée un dataframe df2 qui va contenir le double de ligne du dataframe initial. Chaque “paire” de ligne correspondra à un match. On crée une colonne Rank, qui va remplacer WRank et LRank. On “fusionne” WRank et LRank en faisant une ligne sur deux Rank=WRank , Rank=LRank.""")
    st.markdown("""Puis une ligne sur deux, on remplace le gagnant par le perdant et on renomme cette colonne ‘Winner’ par ‘Player’. On supprime ensuite la colonne ‘Loser’. Notre dataset ressemble désormais à la figure suivante :""")
    st.code("""df2 = pd.DataFrame(np.repeat(df.values, 2, axis=0))\ndf2.columns = df.columns\ndf2""")
    df2 = pd.DataFrame(np.repeat(df.values, 2, axis=0))
    df2.columns = df.columns
    df2

    #st.code("""# Avec l'opération précédante, toutes les colonnes sont devenu du type object. On les repassera en type numérique juste après\ndf2.info()""")
    #df2.info()
    #print("kikou")
    #st.code("""# On crée une colonne Rank, qui va remplacer WRank et LRank.\n# On 'fusionne' WRank et LRank en faisant une ligne sur deux Rank=WRank , Rank=LRank\nlistRank = []\nfor i in range(len(df2)):\n\tif i%2==0:\n\tlistRank.append(df2['WRank'].iloc[i])\nelse:\n\tlistRank.append(df2['LRank'].iloc[i])\ndf2['Rank']=listRank""")
    st.code("""# On crée une colonne Rank, qui va remplacer WRank et LRank.
# On 'fusionne' WRank et LRank en faisant une ligne sur deux Rank=WRank , Rank=LRank

listRank = []

for i in range(len(df2)):
    if i%2==0:
        listRank.append(df2['WRank'].iloc[i])
    else:
        listRank.append(df2['LRank'].iloc[i])

df2['Rank']=listRank""")


    st.code("""# une ligne sur 2,  on remplace le winner par le loser et on renomme cette colonne 'Player'
df2['Win'] = 0
for i in range(len(df2)):
    if i%2 ==0:
        df2['Winner'].iloc[i+1] = df2['Loser'].iloc[i]
        df2['Win'].iloc[i] = 1
    else:
        df2['Win'].iloc[i] = 0


df2.rename(columns={"Winner": "Player"}, inplace=True)
df2 = df2.drop('Loser', axis=1)""")
    
    
    df4 = pd.read_csv('C:/Users/Mdrouge/Downloads/atp_data_cleaned2.csv')
    df2 = df4.drop("Unnamed: 0", axis=1)
    df2

    st.markdown("""On se crée une table de correspondance : nom joueur/rank avant de supprimer les colonnes Player, WRank, LRank qui ne nous sont plus utiles.""")
    st.code("""df_player = df2[['Player','Rank']]
df = df2.drop(['Player','WRank','LRank'], axis=1)
df""")
    df_player = df2[['Player','Rank']]
    df = df2.drop(['Player','WRank','LRank'], axis=1)
    df


    st.markdown("""Notre dataset comporte 86030 lignes et 7 colonnes. Toutes les les colonnes sont au format 'object'. Transformons-les en variables numériques :""")
   
    st.code("""df = df.astype(int)
df.info()""")
 
    df = df.astype(int)
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.markdown("""Au final, voici à quoi ressemble le dataframe que nous exploiterons pour mettre en place nos modèles :""")
    st.code("df")
    df
  
if page == pages[5]:
    st.header(pages[5])
    st.markdown("""---""")
    st.markdown("""Définissons dans cette partie un modèle de machine learning capable de prédire le gagnant d’un match. Pour ce faire nous allons faire différents tests d’algorithmes bien connus. Nous en sélectionnerons un dont nous optimiserons ces hyperparamètres pour booster ses performances.""")
    st.markdown("On sépare df en deux dataFrames, 'features' et 'target', contenant respectivement les variables explicatives et la variable Win à prédire.")
    st.caption("N.B. : Compte tenu de nos données, il ne sera pas nécessaire de faire une normalisation avec l'utilisation d'un scaler.")
    st.code("""#variable cible :
target = df['Win']
features = df.drop('Win', axis=1)""")

    ##### il faut le remettre ici
    df4 = pd.read_csv('C:/Users/Mdrouge/Downloads/atp_data_cleaned2.csv')
    df2 = df4.drop("Unnamed: 0", axis=1)
    df_player = df2[['Player','Rank']]
    df = df2.drop(['Player','WRank','LRank'], axis=1)
    #####
    df
    target = df['Win']
    features = df.drop(['Win'], axis=1)

    st.markdown("""On crée à partir de features et target, un ensemble d'apprentissage contenant 80% des données (X_train, y_train) et un ensemble de test contenant les 20% de données restantes (X_test, y_test).""")
    st.markdown("Aussi, nous utilisons dans notre méthode train_test_split le paramètre shuffle à False. En effet, ce paramètre est indispensable pour la cohérence de notre modèle. Notre dataset ne peut être mélangé n'importe comment dans la mesure où chaque 'paire' de ligne représente un match.")
    st.code("""from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=789, shuffle=False)""")
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=789, shuffle=False)

    st.subheader("Modèle des K plus proches voisins")

    st.code("""from sklearn import neighbors

# Insérez votre code ici
score_minko = []
score_man = []
score_cheb = []

for k in range(1, 41):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric='minkowski')
    knn.fit(X_train, y_train)
    score_minko.append(knn.score(X_test, y_test))

for k in range(1, 41):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    knn.fit(X_train, y_train)
    score_man.append(knn.score(X_test, y_test))
    
for k in range(1, 41):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric='chebyshev')
    knn.fit(X_train, y_train)
    score_cheb.append(knn.score(X_test, y_test))""")


    st.code("""import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(range(1, 41), score_minko, color='blue', linestyle='dashed', lw=2, label='Minkowski')
plt.plot(range(1, 41), score_man, color='orange', linestyle='dashed', lw=2, label='Manhattan')
plt.plot(range(1, 41), score_cheb, color='red', linestyle='dashed', lw=2, label='Chebyshev')
plt.title('Score - valeur de K')  
plt.xlabel('Valeur de K')  
plt.ylabel('Accuracy') 
plt.legend();""")
    st.image("https://drive.google.com/uc?id=1JYYK2ZF-Y5CaTu9QQ5bSrJ0amEsP3PKM")
    st.markdown("""La précision du modèle converge vers une valeur de 0.61 pour k proche de k=28. La métrique 'Manhattan' est celle convergeant le plus rapidement.""")

    st.code("""#paramètres que l'on peut choisir : manhattan et k=28. Le modele converge vers une accuracy de 0.60
    knn = neighbors.KNeighborsClassifier(n_neighbors=28, metric='manhattan')
    knn.fit(X_train, y_train)
    knn.score(X_test, y_test)""")
    st.text("0.6082761827269557")
    st.subheader("Algorithme de Régression logistique")
    st.markdown("""Testons un algorithme de régression logistique :""")

    st.code("""from sklearn import linear_model
clf_reg = linear_model.LogisticRegression()
clf_reg.fit(X_train, y_train)""")
    st.text("LogisticRegression()")
    

    from sklearn import linear_model
    clf_reg = linear_model.LogisticRegression()
    clf_reg.fit(X_train, y_train)


    st.code("""clf_reg.score(X_test,y_test)""")
    clf_reg.score(X_test,y_test)
    st.text("0.5921771475066837")
    st.code("""from sklearn.metrics import confusion_matrix 
y_pred  = clf_reg.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
cm""")

    y_pred  = clf_reg.predict(X_test)
    from sklearn.metrics import confusion_matrix 
    cm = confusion_matrix(y_test,y_pred)
    cm


    st.subheader("Algorithme de Random forest")

   
    st.code("""
    #Random forest
from sklearn import ensemble

clf = ensemble.RandomForestClassifier(n_jobs=-1, random_state=321)

clf.fit(X_train, y_train)""")
     #Random forest
    from sklearn import ensemble

    clf = ensemble.RandomForestClassifier(n_jobs=-1, random_state=321)

    clf.fit(X_train, y_train)
    st.text("RandomForestClassifier(n_jobs=-1, random_state=321)")
    st.code("""y_pred = clf.predict(X_test)

pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames = ['Classe prédite']) """)

    y_pred = clf.predict(X_test)

    pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames = ['Classe prédite'])
    st.text("""Classe prédite	0	1
Classe réelle		
0	5591	3012
1	4020	4583""")
    st.code("""clf.score(X_test, y_test)""")
    clf.score(X_test, y_test)
    #0.5921771475066837
    st.text("0.5921771475066837")
    st.subheader("XGBoost")

    import xgboost as xgb
    # on cree un ensemble de validation , de test, et dentrainement

    X, X_valid, y, y_valid = train_test_split(features, target, test_size=0.2, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

    st.code("""
    import xgboost as xgb
# on cree un ensemble de validation , de test, et dentrainement

X, X_valid, y, y_valid = train_test_split(features, target, test_size=0.2, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)""")

    st.markdown("""Utilisons les DMatrix pour la suite de l'implémentation de notre algorithme XGBoost. DMatrix est une structure interne utilisée par XGBoost, qui est optimisée à la fois pour la performance de mémoire ainsi que pour la vitesse d'entrainement.""")
    
    train = xgb.DMatrix(data=X_train, label=y_train)
    test = xgb.DMatrix(data=X_test, label=y_test)
    valid = xgb.DMatrix(data=X_valid, label=y_valid)

    st.code("""train = xgb.DMatrix(data=X_train, label=y_train)
test = xgb.DMatrix(data=X_test, label=y_test)
valid = xgb.DMatrix(data=X_valid, label=y_valid)""")

    st.markdown("""Commençons avec des paramètres standard par défaut. Il s'agit d'un problème de classification : "objective" doit être donc égal à "binary:logistic .""")
    st.code("""params = {"booster":"gbtree","learning_rate":0.01,"objective":"binary:logistic" }
xgb1 = xgb.train(params=params,dtrain=train,num_boost_round = 50, early_stopping_rounds= 25, evals= [(train, 'train'), (test, 'eval')])
""")

    params = {"booster":"gbtree","learning_rate":0.01,"objective":"binary:logistic" }
    xgb1 = xgb.train(params=params,dtrain=train,num_boost_round = 50, early_stopping_rounds= 25, evals= [(train, 'train'), (test, 'eval')])

    st.text("""[0]	train-error:0.387452	eval-error:0.384982
Multiple eval metrics have been passed: 'eval-error' will be used for early stopping.

Will train until eval-error hasn't improved in 25 rounds.
[1]	train-error:0.3869	eval-error:0.384982
[2]	train-error:0.386784	eval-error:0.384866
[3]	train-error:0.386784	eval-error:0.383413
[4]	train-error:0.386653	eval-error:0.383878
[5]	train-error:0.386653	eval-error:0.383878
[6]	train-error:0.386696	eval-error:0.383413
[7]	train-error:0.38613	eval-error:0.383413
[8]	train-error:0.385912	eval-error:0.383297
[9]	train-error:0.386479	eval-error:0.383297
[10]	train-error:0.386217	eval-error:0.38382
[11]	train-error:0.385505	eval-error:0.38382
[12]	train-error:0.385461	eval-error:0.384633
[13]	train-error:0.385752	eval-error:0.384691
[14]	train-error:0.385214	eval-error:0.383936
[15]	train-error:0.38565	eval-error:0.38411
[16]	train-error:0.385767	eval-error:0.38411
[17]	train-error:0.385767	eval-error:0.38411
[18]	train-error:0.385578	eval-error:0.384982
[19]	train-error:0.385883	eval-error:0.384168
[20]	train-error:0.385767	eval-error:0.382832
[21]	train-error:0.385563	eval-error:0.382657
[22]	train-error:0.385287	eval-error:0.382599
[23]	train-error:0.385345	eval-error:0.382599
[24]	train-error:0.385389	eval-error:0.382367
[25]	train-error:0.38536	eval-error:0.383703
[26]	train-error:0.385476	eval-error:0.384517
[27]	train-error:0.385491	eval-error:0.383297
[28]	train-error:0.385389	eval-error:0.383936
[29]	train-error:0.385302	eval-error:0.383703
[30]	train-error:0.385331	eval-error:0.383587
[31]	train-error:0.385185	eval-error:0.384226
[32]	train-error:0.385069	eval-error:0.383122
[33]	train-error:0.385127	eval-error:0.383238
[34]	train-error:0.384938	eval-error:0.382715
[35]	train-error:0.384764	eval-error:0.382541
[36]	train-error:0.384938	eval-error:0.382715
[37]	train-error:0.385214	eval-error:0.384052
[38]	train-error:0.384953	eval-error:0.384052
[39]	train-error:0.384924	eval-error:0.384226
[40]	train-error:0.384822	eval-error:0.384226
[41]	train-error:0.384808	eval-error:0.384052
[42]	train-error:0.384924	eval-error:0.384168
[43]	train-error:0.384967	eval-error:0.383994
[44]	train-error:0.384866	eval-error:0.383994
[45]	train-error:0.384837	eval-error:0.384052
[46]	train-error:0.385127	eval-error:0.384924
[47]	train-error:0.385069	eval-error:0.384691
[48]	train-error:0.385098	eval-error:0.384866
[49]	train-error:0.385055	eval-error:0.384691
Stopping. Best iteration:
[24]	train-error:0.385389	eval-error:0.382367""")



    st.code("""print("best score (taux de bonne prédiction): ",(1 - xgb1.best_score))
print("best iteration : ",xgb1.best_iteration)
print("best  n tree limit : ",xgb1.best_ntree_limit)""")

    print("best score (taux de bonne prédiction): ",(1 - xgb1.best_score))
    print("best iteration : ",xgb1.best_iteration)
    print("best  n tree limit : ",xgb1.best_ntree_limit)
    st.text("""best score (taux de bonne prédiction):  0.617633
best iteration :  24
best  n tree limit :  25""")
    st.markdown("""Affichons l'importance des caractéristiques du modèles :""")

    st.code("""xgb.plot_importance(xgb1);""")

    st.image("https://drive.google.com/uc?export=view&id=1g2dMDLOUBopAmWXlMwCUNSpCwJloQ7OA")
    
    st.markdown(" Sans suprise le classement correspond à la feature influant le plus le score du modèle.")
    st.code("""predictions = xgb1.predict(test)
print(predictions) """)
    predictions = xgb1.predict(test)
    print(predictions)
    st.text("[0.4744337  0.4744337  0.61848074 ... 0.52120805 0.48682183 0.44901693]")
    st.markdown("""Ici, les prédictions affichée ne sont pas soit 0 soit 1 comme on pourrait s'y attendre pour un problème de classification, mais plutôt la probabilité entre 0 et 1 d'avoir 1, c'est-à-dire que le joueur gagne.""")
    
    st.markdown("""Réglons certains paramètres pour optimiser la performance. Après avoir fait différents essais, on choisit :
""")
    st.code("""param_2 = {
    "booster":"gbtree",
    "max_depth":2,
    "eta":0.3,
    "objective":"binary:logistic",
    "learning_rate":0.05
}

xgb2 = xgb.train(params=params,dtrain=train,num_boost_round = 150, early_stopping_rounds= 50, evals= [(train, 'train'), (test, 'eval')])
""")
    st.text("""[0]	train-error:0.387452	eval-error:0.384982
Multiple eval metrics have been passed: 'eval-error' will be used for early stopping.

Will train until eval-error hasn't improved in 50 rounds.
[1]	train-error:0.3869	eval-error:0.384982
[2]	train-error:0.386784	eval-error:0.384866
[3]	train-error:0.386784	eval-error:0.383413
[4]	train-error:0.386653	eval-error:0.383878
[5]	train-error:0.386653	eval-error:0.383878
[6]	train-error:0.386696	eval-error:0.383413
[7]	train-error:0.38613	eval-error:0.383413
[8]	train-error:0.385912	eval-error:0.383297
[9]	train-error:0.386479	eval-error:0.383297
[10]	train-error:0.386217	eval-error:0.38382
[11]	train-error:0.385505	eval-error:0.38382
[12]	train-error:0.385461	eval-error:0.384633
[13]	train-error:0.385752	eval-error:0.384691
[14]	train-error:0.385214	eval-error:0.383936
[15]	train-error:0.38565	eval-error:0.38411
[16]	train-error:0.385767	eval-error:0.38411
[17]	train-error:0.385767	eval-error:0.38411
[18]	train-error:0.385578	eval-error:0.384982
[19]	train-error:0.385883	eval-error:0.384168
[20]	train-error:0.385767	eval-error:0.382832
[21]	train-error:0.385563	eval-error:0.382657
[22]	train-error:0.385287	eval-error:0.382599
[23]	train-error:0.385345	eval-error:0.382599
[24]	train-error:0.385389	eval-error:0.382367
[25]	train-error:0.38536	eval-error:0.383703
[26]	train-error:0.385476	eval-error:0.384517
[27]	train-error:0.385491	eval-error:0.383297
[28]	train-error:0.385389	eval-error:0.383936
[29]	train-error:0.385302	eval-error:0.383703
[30]	train-error:0.385331	eval-error:0.383587
[31]	train-error:0.385185	eval-error:0.384226
[32]	train-error:0.385069	eval-error:0.383122
[33]	train-error:0.385127	eval-error:0.383238
[34]	train-error:0.384938	eval-error:0.382715
[35]	train-error:0.384764	eval-error:0.382541
[36]	train-error:0.384938	eval-error:0.382715
[37]	train-error:0.385214	eval-error:0.384052
[38]	train-error:0.384953	eval-error:0.384052
[39]	train-error:0.384924	eval-error:0.384226
[40]	train-error:0.384822	eval-error:0.384226
[41]	train-error:0.384808	eval-error:0.384052
[42]	train-error:0.384924	eval-error:0.384168
[43]	train-error:0.384967	eval-error:0.383994
[44]	train-error:0.384866	eval-error:0.383994
[45]	train-error:0.384837	eval-error:0.384052
[46]	train-error:0.385127	eval-error:0.384924
[47]	train-error:0.385069	eval-error:0.384691
[48]	train-error:0.385098	eval-error:0.384866
[49]	train-error:0.385055	eval-error:0.384691
[50]	train-error:0.384895	eval-error:0.384575
[51]	train-error:0.384212	eval-error:0.38475
[52]	train-error:0.384808	eval-error:0.383413
[53]	train-error:0.384372	eval-error:0.383878
[54]	train-error:0.384328	eval-error:0.383587
[55]	train-error:0.384328	eval-error:0.383297
[56]	train-error:0.384125	eval-error:0.383122
[57]	train-error:0.384372	eval-error:0.383297
[58]	train-error:0.384197	eval-error:0.382657
[59]	train-error:0.384197	eval-error:0.382308
[60]	train-error:0.384299	eval-error:0.382599
[61]	train-error:0.384226	eval-error:0.38225
[62]	train-error:0.384139	eval-error:0.382308
[63]	train-error:0.384226	eval-error:0.382308
[64]	train-error:0.384183	eval-error:0.38225
[65]	train-error:0.384299	eval-error:0.38225
[66]	train-error:0.384096	eval-error:0.38225
[67]	train-error:0.384212	eval-error:0.382367
[68]	train-error:0.384343	eval-error:0.382425
[69]	train-error:0.384255	eval-error:0.382367
[70]	train-error:0.384139	eval-error:0.382599
[71]	train-error:0.384154	eval-error:0.382483
[72]	train-error:0.384212	eval-error:0.382367
[73]	train-error:0.384314	eval-error:0.382308
[74]	train-error:0.38427	eval-error:0.38225
[75]	train-error:0.384096	eval-error:0.382483
[76]	train-error:0.384023	eval-error:0.382367
[77]	train-error:0.383907	eval-error:0.382541
[78]	train-error:0.383863	eval-error:0.382599
[79]	train-error:0.383849	eval-error:0.382367
[80]	train-error:0.383979	eval-error:0.382599
[81]	train-error:0.383834	eval-error:0.382541
[82]	train-error:0.383761	eval-error:0.382599
[83]	train-error:0.38382	eval-error:0.382483
[84]	train-error:0.383878	eval-error:0.382425
[85]	train-error:0.38382	eval-error:0.382483
[86]	train-error:0.383834	eval-error:0.382483
[87]	train-error:0.383761	eval-error:0.382599
[88]	train-error:0.383732	eval-error:0.382599
[89]	train-error:0.383791	eval-error:0.382541
[90]	train-error:0.383834	eval-error:0.382599
[91]	train-error:0.383907	eval-error:0.382483
[92]	train-error:0.383791	eval-error:0.382948
[93]	train-error:0.383703	eval-error:0.383064
[94]	train-error:0.383689	eval-error:0.382948
[95]	train-error:0.383587	eval-error:0.38289
[96]	train-error:0.383616	eval-error:0.383471
[97]	train-error:0.383616	eval-error:0.383297
[98]	train-error:0.383427	eval-error:0.383471
[99]	train-error:0.383427	eval-error:0.383355
[100]	train-error:0.383442	eval-error:0.383238
[101]	train-error:0.383442	eval-error:0.383355
[102]	train-error:0.383427	eval-error:0.383297
[103]	train-error:0.383384	eval-error:0.383413
[104]	train-error:0.383456	eval-error:0.383355
[105]	train-error:0.383471	eval-error:0.383238
[106]	train-error:0.383427	eval-error:0.383238
[107]	train-error:0.383413	eval-error:0.383238
[108]	train-error:0.383427	eval-error:0.383122
[109]	train-error:0.383427	eval-error:0.382773
[110]	train-error:0.383326	eval-error:0.38289
[111]	train-error:0.383282	eval-error:0.382715
Stopping. Best iteration:
[61]	train-error:0.384226	eval-error:0.38225
""")

    st.code("""print("best score (taux de bonne prédiction): ",(1 - xgb2.best_score))
print("best iteration : ",xgb2.best_iteration)
print("best n tree limit : ",xgb2.best_ntree_limit)""")

    st.text("""best score (taux de bonne prédiction):  0.61775
best iteration :  61
best  n tree limit :  62""")

    st.markdown("On obtient un best score de 0.61775 (on avait 0.617633 juste avant). Nous pouvons mettre en évidence la liste des prédictions réalisées avec le code suivant :")

    st.code("""predictions = xgb2.predict(test)
pred = [0 if i<0.5 else 1 for i in predictions ]
print(pred)""")

    st.text("[0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, ...]")


    st.markdown("Affichons les scores de notre modèle :")
    st.code("""from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
print('accuracy score : ', accuracy_score(y_valid, pred))
print('precision score : ', precision_score(y_valid, pred))
print('recall score : ', recall_score(y_valid, pred))
print('f1 score : ', f1_score(y_valid, pred))""")


    st.text("""accuracy score :  0.6172846681390213
precision score :  0.6367579289780428
recall score :  0.5460885737533419
f1 score :  0.5879481884738127""")

   



    st.markdown("""Toutes les métriques de performances ont très légérement augmentés. Nous pouvons choisir cet algorithme qui présente de meilleures performances que les précedents.""")
    st.markdown("""Cependant, dans notre projet, nous n'avons pas utilisé d'outil informatique existant pour chercher les meilleurs hyperparamètres pouvant booster encore les performances. Utilisons l'outil GridSearchCV de la bibliothèque sklearn avec notre algorithme de régression logistique :""")

    st.subheader("GridSearchCV")

    st.code("""from sklearn.model_selection import GridSearchCV""")
    from sklearn.model_selection import GridSearchCV
    st.code("""grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge""")
    grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
    st.code("""#Hyperparameter tuning

from sklearn.model_selection import StratifiedKFold
# define model/create instance
lr=linear_model.LogisticRegression()
#tuning weight for minority class then weight for majority class will be 1-weight of minority class
#Setting the range for class weights
weights = np.linspace(0.01,1,20)
#specifying all hyperparameters with possible values
param= {'C': [0.1, 0.5, 1,10], 'penalty': ['l1', 'l2'],"class_weight":[{0:x ,1:1.0 -x} for x in weights]}
# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = False)
#Gridsearch for hyperparam tuning
model= GridSearchCV(estimator= lr,param_grid=param,scoring="f1",cv=folds,return_train_score=True)
#train model to learn relationships between x and y
model.fit(X_train,y_train)""")
    #Hyperparameter tuning

    st.text("""GridSearchCV(cv=StratifiedKFold(n_splits=5, random_state=None, shuffle=False),
             estimator=LogisticRegression(),
             param_grid={'C': [0.1, 0.5, 1, 10],
                         'class_weight': [{0: 0.01, 1: 0.99},
                                          {0: 0.06210526315789474,
                                           1: 0.9378947368421052},
                                          {0: 0.11421052631578947,
                                           1: 0.8857894736842106},
                                          {0: 0.16631578947368422,
                                           1: 0.8336842105263158},
                                          {0: 0.21842105263157896,
                                           1: 0.781578947368421},
                                          {...
                                           1: 0.36473684210526314},
                                          {0: 0.6873684210526316,
                                           1: 0.31263157894736837},
                                          {0: 0.7394736842105263,
                                           1: 0.2605263157894737},
                                          {0: 0.791578947368421,
                                           1: 0.20842105263157895},
                                          {0: 0.8436842105263158,
                                           1: 0.15631578947368419},
                                          {0: 0.8957894736842106,
                                           1: 0.10421052631578942},
                                          {0: 0.9478947368421053,
                                           1: 0.052105263157894655},
                                          {0: 1.0, 1: 0.0}],
                         'penalty': ['l1', 'l2']},
             return_train_score=True, scoring='f1')""")

    st.code("""# print best hyperparameters
print("Best F1 score: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)""")

    st.text("""Best F1 score:  0.6705283815946953
Best hyperparameters:  {'C': 0.1, 'class_weight': {0: 0.37473684210526315, 1: 0.6252631578947369}, 'penalty': 'l2'}""")


    st.markdown("""Grâce au "tuning" d'hyperparamètres, nous arrivons à obtenir un f1-score de 0.67 pour notre modèle de régression, qui est au supérieur au f1-score de 0.59 du modèle XGBoost. En l'état donc, il serait plus approprié d'utiliser ce dernier algorithme pour prédire le gagnant d'un match. Cependant, il est à noter que le paramétrage de l'algorithme XGBoost n'est pas optimal. Il serait intéressant de procéder à l'utilisation d'un GridSearchCV également, ce qui est assez délicat.""")

if page == pages[6]:
    st.header(pages[6])
    st.markdown("""---""")

    st.markdown("""Ce premier projet a été très enrichissant et nous a permis de mettre en pratique un bon nombre de notions de Machine Learning que nous avons apprises à travers cette formation. C'est en faisant que l'on se confronte aux obstacles et problématiques jusqu'alors insoupçonnés. Beaucoups d'éléments sont optimisables, comme tester d'autres types d'algorithmes, prendre plus de temps pour trouver les meilleurs hyperparamètres. Nous pourrions chercher un dataset plus fourni en informations (comme le nombre d'ace par match, le nombre de coups droits gagnants, etc.).

Enfin, en plus d’être très stimulant intellectuellement, nos discussions prolongées, échanges, interrogations entre nous ainsi qu’avec notre coach ont été également des moments très sympathiques et bienveillants.
""")
        
    st.markdown("""
        <h3 style = "color:#D36B3E"><b> Difficile </b> (début de projet, optimisation des modèles) </h3>
        <h3 style = "color:blue"> Enrichissant </b>( confrontations des idées, mise en pratiques, notions abordées) </h3>
        <h3 style = "color:green"> Concret </b></h3>
    """, unsafe_allow_html=True)

    st.markdown("""
        <h2 style = "color:orange"> MERCI ! </h2>
        <h3 style = "color:grey">et merci à Dan pour son accompagnement !</h3>
    """, unsafe_allow_html=True)

 



