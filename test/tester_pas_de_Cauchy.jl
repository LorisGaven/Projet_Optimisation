@doc doc"""
Tester l'algorithme du pas de Cauchy
"""
function tester_pas_de_Cauchy(Pas_De_Cauchy::Function)
    tol_erreur = 1e-8
    @testset "Pas de Cauchy" begin
        @testset "Cas b = 0" begin
            delta = 1
            g = [0; 0]
            H = [7 0 ; 0 2]
            s, e = Pas_De_Cauchy(g, H, delta)
            @test iszero(s) && e == 0
        end
        @testset "Cas a > 0 et Delta < Delta_b" begin
            g = [5, 7]
            H = [10 1; 1 2]
            a = transpose(g) * H * g
            b = -norm(g)^2
            delta_b = - (b/a) * norm(g)
            delta = 0.9 * delta_b
            s, e = Pas_De_Cauchy(g, H, delta)
            @test isapprox(s, - (delta / norm(g)) * g, atol=tol_erreur) && e == -1
        end
        @testset "Cas a > 0 et Delta > Delta_b" begin
            g = [5, 7]
            H = [10 1; 1 2]
            a = transpose(g) * H * g
            b = -norm(g)^2
            delta_b = - (b/a) * norm(g)
            delta = 1.1 * delta_b
            s, e = Pas_De_Cauchy(g, H, delta)
            @test isapprox(s, - ((norm(g)^2) / a) * g, atol=tol_erreur) && e == 1
        end
        @testset "Cas a < 0" begin
            g = [5, 7]
            H = -I
            a = transpose(g) * H * g
            b = -norm(g)^2
            delta_b = - (b/a) * norm(g)
            delta = 1
            s, e = Pas_De_Cauchy(g, H, delta)
            @test isapprox(s, - (delta / norm(g)) * g, atol=tol_erreur) && e == -1
        end
        @testset "Cas a = 0 et b != 0" begin
            g = [5, 7]
            H = zeros(2,2)
            a = transpose(g) * H * g
            b = -norm(g)^2
            delta_b = - (b/a) * norm(g)
            delta = 1
            s, e = Pas_De_Cauchy(g, H, delta)
            @test isapprox(s, - (delta / norm(g)) * g, atol=tol_erreur) && e == -1
        end
    end
end
