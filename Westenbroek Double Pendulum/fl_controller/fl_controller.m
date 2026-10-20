function [u, beta_m, alpha_m] = fl_controller(x, v, p_model)

    [M, C_dq, G] = double_pendulum_matrices(x, p_model);

    beta_m  = C_dq + G;
    alpha_m = M;

    u = beta_m + alpha_m*v;

end