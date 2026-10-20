function [u, beta_m, alpha_m] = cepheus_fl_controller(x, v, params)

    ksi  = x(1:4);
    dksi = x(5:8);
    H = cepheus_H(ksi, params);
    c = cepheus_c(ksi, dksi, params);

    beta_m  = c;
    alpha_m = H;

    u = beta_m + alpha_m*v;

end