@doc doc"""
#### Objet
Cette fonction calcule une solution approchée du problème

```math
\min_{||s||< \Delta}  q(s) = s^{t} g + \frac{1}{2} s^{t}Hs
```

par l'algorithme du gradient conjugué tronqué

#### Syntaxe
```julia
s = Gradient_Conjugue_Tronque(g,H,option)
```

#### Entrées :   
   - g : (Array{Float,1}) un vecteur de ``\mathbb{R}^n``
   - H : (Array{Float,2}) une matrice symétrique de ``\mathbb{R}^{n\times n}``
   - options          : (Array{Float,1})
      - delta    : le rayon de la région de confiance
      - max_iter : le nombre maximal d'iterations
      - tol      : la tolérance pour la condition d'arrêt sur le gradient

#### Sorties:
   - s : (Array{Float,1}) le pas s qui approche la solution du problème : ``min_{||s||< \Delta} q(s)``

#### Exemple d'appel:
```julia
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
xk = [1; 0]
options = []
s = Gradient_Conjugue_Tronque(gradf(xk),hessf(xk),options)
```
"""
function Gradient_Conjugue_Tronque(g,H,options)

    "# Si option est vide on initialise les 3 paramètres par défaut"
    if options == []
        delta = 2
        max_iter = 100
        tol = 1e-6
    else
        delta = options[1]
        max_iter = options[2]
        tol = options[3]
    end

    n = length(g)
    s = zeros(n)
    j = 0
    g0 = g
    p = -g

    q(s) = s'*g + (1/2)*s'*H*s

    while j < 2*n && norm(g) > max(norm(g0)*tol, tol)
        a = norm(p)^2
        b = 2*s'*p
        c = norm(s)^2 - delta^2
        k = transpose(p)* H * p
        D = sqrt(b^2 - 4 * a * c)
        σ_pos = (-b + D) / (2 * a)
        σ_neg = (-b - D) / (2 * a)
        if k <= 0
            #calculer la racine tq q(s+σ*p) min
            if q(s+σ_pos*p) < q(s+σ_neg*p)
                σ = σ_pos
            else
                σ = σ_neg
            end
            return s + σ * p
        end
        α = transpose(g) * g / k
        if norm(s + α * p) > delta
            σ = σ_pos #calculer la racine positive
            return s +  σ * p
        end
        s += α * p
        g_save = g
        g += α * H * p
        β = (transpose(g) * g) / (transpose(g_save) * g_save)
        p = -g + β * p
        j += 1
    end

    return s
end
