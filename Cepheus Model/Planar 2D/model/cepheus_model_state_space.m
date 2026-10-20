function dx = cepheus_model_state_space(t, x, u, params)
% State: x = [theta0; q1; q2; q3; dtheta0; dq1; dq2; dq3]
% Input: u = [tau_rw; tau1; tau2; tau3]

    ksi  = x(1:4);
    dksi = x(5:8);

    H = cepheus_H(ksi, params);
    c = cepheus_c(ksi, dksi, params);

    % Generalized accelerations
    ddksi = H \ (u - c);

    % State derivative
    dx = [dksi; ddksi];

end