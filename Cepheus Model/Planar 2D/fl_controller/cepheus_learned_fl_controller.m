function [u, u_nom, u_learned, beta_m, alpha_m, beta_theta, alpha_theta] = ...
    cepheus_learned_fl_controller(x, v, params_nom, theta_vec)

    ksi  = x(1:4);
    dksi = x(5:8);

    Hhat = cepheus_H(ksi, params_nom);
    chat = cepheus_c(ksi, dksi, params_nom);

    beta_m  = chat;   % 4 x 1
    alpha_m = Hhat;   % 4 x 4

    % Feature vectors
    [phi_beta, phi_alpha] = cepheus_policy_features(x);

    n_beta  = length(phi_beta);
    n_alpha = length(phi_alpha);

    % Parameter sizes
    n_theta_beta  = 4*n_beta;   
    n_theta_alpha = 16*n_alpha;  

    n_params = n_theta_beta + n_theta_alpha;

    if numel(theta_vec) ~= n_params
        error('Wrong theta_vec size. Expected %d parameters.', n_params);
    end

    % Split theta vector
    theta_beta_vec  = theta_vec(1:n_theta_beta);
    theta_alpha_vec = theta_vec(n_theta_beta+1:end);

    % Reshape parameters
    Theta_beta  = reshape(theta_beta_vec, 4, n_beta);
    Theta_alpha = reshape(theta_alpha_vec, 16, n_alpha);

    % Learned beta correction
    beta_theta = Theta_beta * phi_beta; % 4 x 1

    % Learned alpha correction
    alpha_entries = Theta_alpha * phi_alpha;  % 16 x 1

    % MATLAB column-major order
    alpha_theta = reshape(alpha_entries, 4, 4);

    % Nominal FL torque
    u_nom = beta_m + alpha_m*v;

    % Learned FL torque
    u = (beta_m + beta_theta) + (alpha_m + alpha_theta)*v;

    % Extra learned torque
    u_learned = u - u_nom;

end