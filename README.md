# Calcul des word embeddings à partir d’une matrice de co-occurence transformée

En traitement automatique de langage, il est fortement encouragé de représenter les mots présents dans le vocabulaire de façon vectorielle via l’utilisation de prolongement de mots (Word Embeddings), afin d’apporter une dimension sémantique quantifiable dans les calculs réalisés durant l’entraînement des modèles d’apprentissage automatique.

Il existe une variété de méthodes pour obtenir ces prolongements de mots. Notre objectif, dans un premier temps, est de réaliser un état d’art des différents procédés employés ainsi que des métriques utilisées pour évaluer leur qualité, puis, de nous focaliser sur une méthode en particulier, nommée PSDVec, capable de transformer une matrice de cooccurrence en prolongements de mots via un algorithme d’apprentissage itératif, dont l’entraînement peut se poursuivre même
après avoir ajouté de nouvelles entrées à la matrice de cooccurrence.

Finalement, nous réaliserons une étude comparative entre PSDVec et d’autres méthodes de prolongement de mots, tout en testant des configurations différentes au niveau des dimensions, ainsi que de nouvelles transformations de la matrice de cooccurrence.
## I. Étude de l’existant 

### <u> 1. Principes théoriques et rappels </u>
#### 1.1 Prolongements de mots basés sur la fréquence

Le traitement automatique du langage consiste à modéliser le langage écrit ou parlé via des algorithmes informatiques. Afin que la modélisation soit compréhensible par l’ordinateur, le langage doit être représenté sous forme vectorielle.

L’une des méthodes les plus répandues est d’encoder chaque mot via l’approche one-hot, ou le mot en question serait remplacé par un vecteur de dimension n, égale au nombre de mots dans le vocabulaire utilisé. Le mot, placé en position i dans le vecteur, aurait une valeur égale à 1, tandis que les autres éléments prennent une valeur de 0. Une autre méthode répandue, lorsqu’on a un corpus constitué de documents, repose sur la constitution d’un vocabulaire via l’approche sac-de-mots (bag-of-words), qui se base sur la fréquence des mots pour constituer une matrice de cooccurence entre les mots et les documents. Une normalisation peut être apportée via TF-IDF (term frequency-inverse document frequency) afin de valoriser les mots plus rares dans l’ensemble des documents.

#### 1.2 Prolongements de mots basés sur la factorisation matricelle et les fenêtres de mots locales

Ces méthodes présentées dans la section 1.1 ont un point commun : Aucune considération n’est apportée à la dimension sémantique des mots, en effet, seule la fréquence globale des mots est prise en compte pour modéliser le vocabulaire sous forme de vecteurs. Ainsi, une partie significative de l’information est perdue, ce qui peut avoir des conséquences dramatiques sur l’entraînement des modèles d’apprentissage automatique.

Il existe une approche qui répond à cette problématique, et qui consiste à appliquer des méthodes de factorisation matricielles sur des matrices de co-occurrence calculées sur des fenêtres de mots
réduites et fixes, ce qui ouvre la possibilité d’obtenir des prolongements de mots tenant en compte leur côté sémantique, en partant du principe que les mots apparaissant souvent dans la
même fenêtre font partie du même champ lexical.

#### 1.3 Prolongements de mots basés sur les réseaux de neurones

Les méthodes de calcul de prolongements de mots plus récentes tentent aussi de répondre à cette problématique en faisant appel à la puissance et la flexibilité des modèles d’apprentissage automatique. Ces dernières modélisent le vocabulaire tout en incluant une dimension sémantique calculée via deux procédés phares : CBOW (Continuous Bag Of Words) et Skipgram [1].

Les deux méthodes emploient des réseaux de neurones pour apprendre des matrices d’encodage transformant les représentations one-hot des mots en vecteurs de taille réduite et fixe condensant autant que possible l’information syntaxique et le contexte sémantique du mot.

### <u> 2. État de l’art </u>
#### 2.1 Décomposition en Valeurs Singulières (SVD) d’une matrice de co-occurrence

La décomposition en valeurs singulières est une méthode de factorisation matricielle ayant pour but de représenter une matrice <b>A</b> quelconque sous forme d’un produit de trois matrices <b>USV  t</b>, les matrices <b>U</b> et <b>V</b> sont composées de vecteurs orthogonaux, et la matrice S est diagonale et est composée des valeurs singulières de A.
 
La décomposition de la matrice <b>A</b> peut être réalisée directement via extraction des valeurs propres si <b>A</b> est diagonalisable, ou, plus généralement, via le calcul des valeurs propres de la matrice <b>AA t</b>.

Dans le contexte des Word Embeddings, si l’on applique la décomposition en valeurs singulières à une matrice de co-occurence <b>A</b> de taille <b>n ∗ n</b> en un produit <b>USV t</b>, et l’on sélectionne les <b>k</b> valeurs singulières les plus larges, on obtient des prolongements de mots de taille <b>n ∗ k</b> via sélection des <b>k</b> premières lignes la matrice <b>U</b>, ou de taille <b>k ∗ n</b> via sélection des k premières colonnes de la matrice <b>V</b>.

#### 2.2 Word2Vec
Word2Vec [[1]](https://arxiv.org/pdf/1301.3781.pdf) est la première concrétisation des réseaux de neurones de type CBOW et Skipgram, elle a été introduite dans la même publication que ces derniers.

L’entraînement de CBOW est réalisé en sommant le modèle à identifier un mot à partir de son contexte, qui ici réfère à son voisinage, sans fournir le mot en question.

L’entraînement de Skip-gram est similaire à celui de CBOW, sauf qu’au lieu de prédire le mot caché à partir du contexte fourni, c’est ce dernier qui doit être prédit en fournissant un mot au modèle.

Les prolongements de mots Word2Vec peuvent être entraînes selon CBOW ou Skip-gram, chaque approche a ses atouts, ses forces et ses faiblesses en fonction du contexte.

#### 2.3 GloVe

Les prolongements de mots GloVe (Global Vectors) [[2]](https://nlp.stanford.edu/pubs/glove.pdf) sont obtenus en entraînant un réseau de neurones calculant une matrice d’encodage (Embedding matrix) en minimisant une fonction de coût faisant intervenir, entre autres, les paramètres du réseau de neurones et leurs biais ainsi que le logarithme de la matrice de co-occurrence <b>Xij</b>, qui représente le nombre de fois qu’un mot i apparaît dans le voisinage d’un mot <b>j</b>.

Cela permet aux prolongements de mots de tenir en compte les ratios de probabilité liés au nombre d’occurrence des mots apparaissant dans la même fenêtre.

#### 2.4 FastText

GloVe (2.3) et Word2Vec (2.2) possèdent néanmoins des faiblesses de taille, à savoir leur incapacité à traiter des mots en dehors du vocabulaire, de reconnaître les préfixes et suffixes, ainsi que de traiter efficacement les mots de la même famille (par exemple : eat eating, eaten...).

Les prolongements de mots de type FastText [[3]](https://arxiv.org/pdf/1607.04606.pdf) répondent à cette problématique en dérivant de précieuses informations à partir de l’extraction des groupes de caractères constituant les mots, puis leur utilisation pour enrichir les prolongements de mots, qui eux sont calculés de la même manière que Word2Vec.

#### 2.5 PSDVec

PSDVec (Positive-Semidefinite Vectors) [[4]](https://arxiv.org/pdf/1606.03192.pdf) est une méthode basée sur la factorisation matricielle à travers la décomposition en valeurs propres de la matrice d’information mutuelle <b>G</b> dérivée de la matrice de co-occurrence.

La méthode a la particularité de fournir de meilleurs résultats qu’a travers la factorisation matricielle via la décomposition en vecteurs propres (SVD), vu que cette dernière se base sur la matrice <b>GGt</b>, ce qui a pour conséquence une perte d’information par rapport à la matrice <b>G</b>.

### <u> 3. Méthodes d’évaluation </u>
#### 3.1 Tâche de similarité

L’évaluation à travers les tâches de similarité consiste à comparer, via l’utilisation d’une mesure de similarité (similarité cosine par exemple), les vecteurs représentatifs des mots entre eux. Un score est attribué à chaque paire de mots, puis comparé à un score de référence.

Au delà de la simple comparaison de scores, un classement peut aussi être formé, a partir duquel le classement des paires de mots à évaluer sont comparées en fonction de leur similarité relationnelle avec un ensemble de paires de mots fournies. C’est le principe de base du dataset SemEval [[5]](https://aclanthology.org/S12-1047.pdf).

#### 3.2 Tâche d'analogie

L’évaluation à travers les tâches d’analogie consiste à mettre à l’épreuve les prolongements de mots en testant leur capacité à modéliser de façon arithmétique les relations entre les paires de mots provenant de contextes similaires.

Un exemple typique serait d’obtenir la même valeur en calculant la distance entre la représentation vectorielle de la paire "France"-"Paris" et de la paire "Allemagne"-"Berlin", ce qui illustre la capacité de l’espace vectoriel contenant les prolongements de mots à modéliser efficacement la relation "Pays-Capitale".

Cette manière d’évaluer les Word Embeddings a fait ses preuves, de par sa simplicité à comprendre et sa manière claire de constater les résultats [[6]](https://aclanthology.org/N13-1090.pdf) .

D’autres jeux de données liés à la tache d’analogie existent, tels que Google et MSR [[6]](https://aclanthology.org/N13-1090.pdf).

La plupart des datasets se basent sur deux métriques : 3COSADD et 3COSMUL [[7]](https://arxiv.org/pdf/1702.02170.pdf), qui calculent la similarité cosine entre un vecteur représentant un mot a, et une opération arithmétique entre d’autres vecteurs représentant d’autres mots analogues (par exemple : France et Paris/Allemagne/Berlin). Cette opération est censée produire le vecteur <b>a</b>.

![Taches!](/pics/Tâches.PNG)

## II. Contributions apportées

### <u> 4. Reproduction des résultats du papier PSDVec </u>
#### 4.1 Explication détaillée de la publication introduisant PSDVec

PSDVec (Positive-Semidefinite Vectors) [[4]](https://arxiv.org/pdf/1606.03192.pdf) [Github repo](https://github.com/askerlee/topicvec/tree/master/psdvec) est une méthode permettant l’apprentissage de prolongements de mots condensant l’information syntaxique et sémantique des mots composant le corpus fourni.

La méthode repose sur la factorisation matricielle d’une matrice calculée à partir de l’information mutuelle (PMI - Pointwise Mutual Information) obtenue via les probabilités d’occurrence des unigrammes et bigrammes des mots composant le corpus. La factorisation est réalisée à travers la décomposition de la matrice PMI, G, en valeurs vecteurs propres.

L’approximation de la matrice G par la matrice d’encodage V est réalisée en usant d’un algorithme appelé BCD (Block Coordinate Descent) [[8]](https://www.aaai.org/Papers/ICML/2003/ICML03-094.pdf). Néanmoins, l’algorithme requiert la décomposition en éléments propres de G, ce qui peut s’avérer coûteux pour de larges vocabulaires.

PSDVec résout cette problématique en transformant l’apprentissage des Word Embeddings en une version incrémentale. Cette version permet d’une part d’accélérer le processus d’entraînement, et permet d’autre part d’interrompre ce dernier, d’ajouter de nouvelles entrées au vocabulaire fourni, et de reprendre l’apprentissage des vecteurs de ces nouveaux mots sans relancer tout le processus.

PSDVec fournit aussi une panoplie logicielle d’outils codés en Python et Perl pour traiter de larges corpus de texte de Wikipéda (extractwiki.py), de calculer la matrice de co-occurrence (gramcount.pl), de lancer le processus d’apprentissage des prolongements de mots (factorize.py), et enfin d’évaluer ces derniers via les tâches de similarité et d’analogie (evaluate.py)

#### 4.2 Exécution de la pipeline de PSDVec sur Wikipédia
#### 4.a Extraction des données de Wikipédia
![4a!](/pics/4a.PNG)

#### 4.b Comptage des unigrammes et bigrammes
![4b!](/pics/4b.PNG)
![4b2!](/pics/4b2.PNG)

On obtient deux matrices de cooccurrence sous la forme de deux fichiers, top1grams-wiki.txt et top2grams-vec.wiki, correspondant respectivement au nombre d’occurrence des unigrammes et bigrammes.

#### 4.c Calcul des prolongements de mots
![4c!](/pics/4c.PNG)
![4c2!](/pics/4c2.PNG)

Les core words résultent de la partition du vocabulaire obtenu à l’étape précédente, et représentent les mots dont la fréquence est la plus élevée, les non-core words désignent le reste du vocabulaire.

#### 4.d Évaluation des prolongements de mots
![4d!](/pics/4d.PNG)

Au vu des excellentes performances obtenues sur la plupart des datasets d’analogie et de similarité, on en conclut que les vecteurs représentant les mots du vocabulaire construit capturent parfaitement leurs nuances sémantiques.

![4d2!](/pics/4d2.PNG)

#### 4.e Remarque

Par la suite, et pour des soucis de consommation de mémoire vive et de vitesse d’exécution, nous nous baserons sur des prolongements de mots calculés sur un échantillon de 10000 articles provenant de Wikipedia.

#### 4.3 Calcul des prolongements de mots via Décomposition en valeurs singulières (SVD) sur Wikipédia

#### 4.a Calcul de la matrice de co-occurrence

On se base sur les résultats des scripts de comptage de PSDVec pour obtenir la matrice de cooccurrence sous format Numpy.

![43a!](/pics/43a.PNG)

#### 4.b Décomposition en valeurs singulières

Comme expliqué dans la section 2.1, la matrice de co-occurrence est décomposée en 3 matrices <b>U</b>, <b>S</b> et <b>V</b>, puis on sélectionne les premières colonnes de V correspondant aux valeurs singulières les plus élevées.

![43b!](/pics/43b.PNG)

#### 4.c Évaluation des prolongements de mots obtenus via SVD

![43c!](/pics/43c.PNG)

A première vue, on observe une baisse de qualité des prolongements de mots comparé a ceux obtenus via PSDVec.

#### 4.d Comparaison entre les performances des prolongements de mots obtenus via SVD et ceux obtenus via PSDVec

Les chiffres en <b>gras</b> réfèrent aux valeurs maximales des métriques de performance pour le jeu de données courant.

![43d!](/pics/43d.PNG)

Le tableau de comparaison confirme la supériorité de PSDvec dans la production de prolongements de mots prenant en compte leur aspect sémantique, et ce, malgré n’avoir utilisé qu’une partie des articles de Wikipédia (10 000 articles). En effet, PSDVec surpasse aisément SVD, en se basant sur les métriques obtenues sur les données de test.


### <u> 5. Modifications apportées à la matrice de co-occurrence </u>

Le tableau 2 illustre l’écart des performances d’évaluation entre PSDVec et SVD, mais en est-il de même lorsqu’on applique des transformations mathématiques sur les matrices de cooccurrence ? Les sections suivantes tentent de répondre à cette question.

#### 5.1 Transformation AFC

La transformation AFC consiste à soustraire la matrice de cooccurrence des bigrammes M à l’inverse de la racine carrée du produit de deux matrices : La première, Dr, est la matrice obtenue en sommant les colonnes de M. La seconde, Dc, est obtenue de façon similaire, en sommant les lignes de M.

La transformation AFC consiste à diviser la matrice de cooccurrence des bigrammes M par la racine carrée du produit de deux matrices : La première, Dr, est la matrice obtenue en sommant les colonnes de M. La seconde, Dc, est obtenue de fa¸con similaire, en sommant les lignes de M.

Cette transformation est une forme de normalisation visant à obtenir une matrice de contingence.

Après transformation de la matrice de cooccurrence puis calcul des prolongements de mots via PSDVec et SVD, on évalue les performances de ceux-ci sur les datasets correspondants aux taches de similarité et d’analogie.

![51!](/pics/51.PNG)

On constate une baisse de performances des vecteurs obtenus par le biais de PSDVec vis-à-vis des tâches d’analogie et de similarité. On remarque aussi qu’une augmentation de la dimension des vecteurs obtenus par le biais de PSDVec se traduit par une amélioration des performances de celles-ci.

#### 5.2 Transformation Logarithme

La transformation Logarithme consiste à appliquer la fonction Log à la matrice de cooccurrence.

Les éléments nuls de la matrice sont ignorés pour éviter des erreurs mathématiques.

![51!](/pics/52.PNG)

On constate que, contrairement au scénario ou PSDVec surpassait SVD dans tous les domaines lorsqu’aucune transformation n’était appliquée, ici, SVD arrive à surpasser PSDVec dans les tâches d’analogie lorsque la transformation logarithme est appliquée.

#### 5.3 Transformation : Puissance α

La transformation Logarithme consiste à élever la matrice de cooccurrence à la puissance α.
Les valeurs de α testées sont 0.1 et 0.4.

![53!](/pics/53.PNG)

![532!](/pics/532.PNG)

On remarque que, malgré une légère baisse de performances de la part des vecteurs produits par PSDVec, ceux produits par SVD ont subi un accroissement significatif au niveau des tâches d’analogie et de similarité, au point de surpasser PSDVec.

#### 5.4 Contraintes et difficultés rencontrées

Ce travail d’initialisation, d’entraînement, et de comparaison des méthodes d’obtention des Word Embeddings ne s’est pas déroulé sans encombres. Parmi les difficultés rencontrées durant ce projet, on peut trouver :

- Présence de bugs dans le code, tels que le calcul des unigrammes/bigrammes basé sur des entiers naturels, et non des chiffres à virgule flottante.
- Difficultés à gérer la mémoire vive, même après avoir eu recours à Google Colaboratory Pro.
- Les fichiers top1grams-wiki.txt et top2grams-wiki.txt stockent les occurrences des unigrammes et bigrammes sous forme de texte brut, et non sous forme de matrice sparse. Ce qui induit le besoin de parser le fichier pour en extraire la matrice de cooccurrence et y appliquer des transformations.

## III. Conclusion Générale et Perspectives

Ce projet fût une occasion de renforcer notre compréhension des Word Embeddings à travers l’étude de l’aspect théorique du domaine, de ses différentes pratiques et procédés, ainsi que les méthodes d’évaluations de ces derniers.

Nous nous sommes ensuite penchés sur certaines implémentations, telles que Word2Vec et Glove, puis nous nous sommes focalisés sur PSDVec, via la lecture et la compréhension de la publication scientifique introduisant la méthode, la reproduction de ses résultats dans un environnement logiciel, et l’enrichissement de ces derniers en appliquant diverses transformations à la matrice de co-occurrence sur laquelle la méthode se base.

Une étude comparative des résultats avec la méthode SVD fut aussi menée, à la suite de laquelle on a constaté que, si PSDVec produit des prolongements de mots largement supérieurs à SVD sans modifier la matrice de cooccurrence, les transformations de type logarithme équilibrent les performances, et les transformations consistant en une élévation à la puissance α font accroître drastiquement les performances des vecteurs produits par SVD.

Une perspective à considérer serait d’inclure plusieurs méthodes de production des prolongements de mots dans notre analyse, ainsi que de tester des configurations plus gourmandes en mémoire vive, afin d’avoir une analyse aussi complète que possible.
