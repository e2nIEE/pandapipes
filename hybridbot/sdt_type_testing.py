import pandapipes as pps

net = pps.create_empty_network()
j1 = pps.create_junction(net, pn_bar=2, tfluid_k=300)
j0 = pps.create_junction(net, pn_bar=2, tfluid_k=300)
pipe1 = pps.create_pipe(net, 0,1,length_km=0.1, diameter_m=0.05, std_type="80_GGG", alpha_w_per_m2k=50)
leonie = 0