% % sigmoid function pars
% p.pfc_a
% p.pfc_b
% p.pfc_c
% p.pmd_a
% p.pmd_b
% p.pmd_c
% p.gpi_a
% p.gpi_b
% p.gpi_c

% weights for E and I
p.E_PFC_w = 0.004;
p.E_PMd_w1 = 0.02;
p.E_PMd_w2 = 0.05;
p.I_PMd_w = 0.3;
p.E_M1_w1 = 0.01;
p.E_M1_w2 = 0.05;
p.I_M1_w = 0.03;
p.E_GPi_w1 = 0.2;
p.E_GPi_w2 = 0.15;
p.I_GPi_w1 = 0.2;
p.I_GPi_w2 = 0.1;

p.I_GPi_w3 = 0.05;

% kernel pars
p.Kd_pmd_1 = 0.8;
p.Kd_pmd_2 = 0.04;
p.Kd_pmd_3 = 0.2;
p.Kd_gpi_1 = 0.8;
p.Kd_gpi_2 = 0.04;
p.Kd_gpi_3 = 0.2;

p.gpi_threshold = 3;

% p.slow_u0_base = 0.3;
% p.slow_A = 0.002;
% p.fast_u0_base = 1.5;
% p.fast_A = 0.003;
% p.phi_noise_w = 0.3;

p.slow_u0_base = 1;
p.slow_A = 0.002;
p.fast_u0_base = 4;
p.fast_A = 0.0015;
p.phi_noise_w = 0.3;