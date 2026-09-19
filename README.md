# HJM Framework — Modèle de Hull-White

Dérivation du modèle de Hull-White comme cas particulier du cadre Heath-Jarrow-Morton
(HJM), et simulation Monte Carlo du taux court sous la mesure risque-neutre, avec
vérification de la calibration sur la courbe zéro-coupon initiale.

## Contenu

| Fichier | Description |
|---|---|
| `Hull_White.ipynb` | Dérivation analytique complète (LaTeX) du drift HJM sans arbitrage jusqu'à la dynamique de Hull-White |
| `Hull_white.py` | Script de simulation Monte Carlo (format cellules `# %%`) |

## Cadre théorique

Dans le cadre HJM, la dynamique du taux forward instantané s'écrit

$$df(t;T) = Q(t;T)\,dt + \sigma(t;T)\,dW^{\mathbb{Q}}(t),
\qquad Q(t;T) = \sigma(t;T)\int_t^T \sigma(t;s)\,ds$$

la seconde égalité étant la condition de non-arbitrage sur le drift.

En spécifiant une volatilité exponentiellement décroissante

$$\sigma(t;T) = \sigma_0 e^{-\lambda(T-t)}$$

le notebook montre pas à pas que le taux court suit un processus d'Ornstein-Uhlenbeck
à moyenne mobile déterministe :

$$dr(t) = \lambda\big(\theta(t) - r(t)\big)dt + \sigma\,dW^{\mathbb{Q}}(t),
\qquad
\theta(t) = \frac{1}{\lambda}\frac{\partial f(0;t)}{\partial t} + f(0;t)
+ \frac{\sigma^2}{2\lambda^2}\left(1 - e^{-2\lambda t}\right)$$

Le terme $\theta(t)$ est ce qui assure l'ajustement exact à la courbe des taux
observée à la date 0.

## Implémentation

`Hull_white.py` simule `NofPath = 500` trajectoires sur `NofSet = 500` pas de temps,
horizon $T = 40$ ans, avec $\sigma = 0{,}25$ et $\lambda = 0{,}5$.

Points d'implémentation :

- **Courbe initiale** : $P(0,T) = e^{-\lambda T}$, avec $f(0;t)$ obtenu par
  différences finies centrées sur $\log P(0,\cdot)$.
- **Réduction de variance** : standardisation des incréments gaussiens colonne par
  colonne (moyenne nulle, variance unitaire) avant construction du brownien.
- **Discrétisation** : schéma d'Euler sur $r(t)$, intégration du facteur
  d'actualisation $P(t)$ par la règle du trapèze sur $r$.
- **Coefficients affines** : $A_0(\tau)$ et $B_0(\tau)$ sont fournis pour la fonction
  caractéristique / le prix analytique du zéro-coupon
  $P(t,T) = e^{A(t,T) + B(t,T)r(t)}$.

## Validation

Le test final compare la courbe zéro-coupon reconstruite par Monte Carlo
$\hat{P}(0,T) = \mathbb{E}\left[e^{-\int_0^T r(s)ds}\right]$ à la courbe de marché
$P_0(T)$. La superposition des deux courbes confirme que le $\theta(t)$ dérivé
recalibre correctement le modèle.

## Dépendances

```
numpy pandas matplotlib
```

## Utilisation

```bash
pip install numpy pandas matplotlib
python Hull_white.py
```

Le script est découpé en cellules `# %%` : il s'exécute aussi bien dans VS Code
(mode interactif) que dans Spyder ou PyCharm.

## Pistes d'extension

- Calibration de $(\sigma, \lambda)$ sur des prix de caps/swaptions plutôt que des
  paramètres fixés.
- Valorisation analytique des zéro-coupons via $A(t,T)$, $B(t,T)$ pour contrôler
  l'erreur Monte Carlo.
- Extension Hull-White à deux facteurs ou volatilité par morceaux.
