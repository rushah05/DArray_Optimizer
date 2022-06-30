using DelimitedFiles
using LinearAlgebra

A = readdlm("A.mat", Float32);
LU = readdlm("LU.mat", Float32); 


println("A size", size(A)) 
println("norm(A) ", norm(A))

ip = readdlm("ipiv", Int)[1,:]
perm = LinearAlgebra.ipiv2perm(ip, length(ip)); 
L = tril(LU, -1) + I; 
U = triu(LU); 
pA = A[:, :]
pA[perm,:] = (L*U) 

println("norm(P*L*U-A)/norm(A)=", norm(pA-A)/norm(A))

