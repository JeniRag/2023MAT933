 

k_list=[10 20 30 40 50 60];
for i=1:length(k_list)
    k=k_list(i);
    [W,H]=semi_nmf(DDI_triple, k);
    
    save("../../../Code/W"+int2str(k)+".mat", "W")
    save("../../../Code/H"+int2str(k)+".mat", "H")

end