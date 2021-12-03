using Pkg
Pkg.add("DataStructures")
Pkg.add("Distributions")

#Hàm phụ trợ tìm argmax lấy từ tài liệu tr.662
function Base.findmax(f::Function, xs)
    f_max = -Inf
    x_max = first(xs)
    for x in xs
        v = f(x)
        if v > f_max
            f_max, x_max = v, x
        end
    end
    return f_max, x_max
end

Base.argmax(f::Function, xs) = findmax(f, xs)[2]
using DataStructures




#Tập người chơi
struct Agent
    a::Vector{Int64}

    function Agent()
        x=[1,2]
        new(x)
    end
end

#Tập hành động có thể thực hiện
struct JointActionSpace
    actionSpaces::Vector{Int64}

    function JointActionSpace()
        x=[]
        for i=2:98
            push!(x,i)
        end
        new(x)
    end
end

#Độ tối ưu của hành động
struct DiscountFactor
    y::Float64

    function DiscountFactor()
        new(0.5)
    end
end

#Struct game travelerDilemma
struct TravelerDilemma
    γ::DiscountFactor# discount factor
    ℐ::Agent # agents
    𝒜::JointActionSpace# joint action space
function TravelerDilemma()
    new(DiscountFactor(),Agent(),JointActionSpace())
end

end

#Chính sách của người chơi 
struct SimpleGamePolicy
    p # dictionary mapping actions to probabilities
    function SimpleGamePolicy(p::Base.Generator)
        return SimpleGamePolicy(Dict(p))
    end
    function SimpleGamePolicy(p::Dict)
        vs = collect(values(p))
        vs ./= sum(vs)
        return new(Dict(k => v for (k,v) in zip(keys(p), vs)))
    end
    SimpleGamePolicy(ai) = new(Dict(ai => 1.0))
end


#Tinh reward của 2 hành động
function resultFunction(a::Tuple{Int64, Int64})
    if a[1]==a[2]
        return [a[1],a[2]]
    elseif a[1]<a[2]
        return [a[1]+2,a[1]-2]
    elseif a[1]>a[2]
        return [a[2]-2,a[2]+2]
    end
end

#Tinh probability
(πi::SimpleGamePolicy)(ai) = get(πi.p, ai, 0.0)
#Tao tập hành động có thể xảy ra. VD: nhận ([1,2]) trả về ([1,1],[1,2],[2,1],[2,2])
joint(X1::Vector{Int64}, X2::Vector{Int64}) = vec(collect(Iterators.product(X1,X2)))

#Tính utility
function utility(m::TravelerDilemma, π, i)
    actSpace = m.𝒜.actionSpaces
    p(a) = prod(πi(ai) for (πi,ai) in zip(π,a))
    return sum(resultFunction(a)[i]*p(a) for a in joint(actSpace,actSpace))
end

#Ghép chính sách mới
joint(π, πi, i) = [i == j ? πi : πj for (j, πj) in enumerate(π)]

#Trả về câu trả lời tốt nhất
function best_response(𝒫::TravelerDilemma, π, i)
    U(ai) = utility(𝒫, joint(π, SimpleGamePolicy(ai), i), i)
    ai = argmax(U, 𝒫.𝒜.actionSpaces)
    return SimpleGamePolicy(ai)
end

#Tính nash theo phương pháp vòng lặp best response
struct IteratedBestResponse
    k_max # number of iterations
    π # initial policy
end

function IteratedBestResponse(𝒫::TravelerDilemma, k_max)
    π = [SimpleGamePolicy(ai => 1.0 for ai in 𝒜i) for 𝒜i in 𝒫.𝒜.actionSpaces]
    return IteratedBestResponse(k_max, π)
end

function solve(M::IteratedBestResponse, 𝒫::TravelerDilemma)
    π = M.π
    for k in 1:M.k_max
        π = [best_response(𝒫, π, i) for i in 𝒫.ℐ.a]
    end
    return π
end

p=TravelerDilemma()
iterRes=IteratedBestResponse(p,5)
solve(iterRes,p)