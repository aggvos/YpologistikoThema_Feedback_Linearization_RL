function [u, u_nom, u_learned, beta_m, alpha_m, beta_theta, alpha_theta] = learned_fl_controller(x, v, p_nom, theta_vec)

    % Nominal model terms
    [Mhat, Chat_dq, Ghat] = double_pendulum_matrices(x, p_nom);

    beta_m  = Chat_dq + Ghat;   % 2 x 1
    alpha_m = Mhat;             % 2 x 2

    % Feature vectors
    [phi_beta, phi_alpha] = policy_features(x);

    n_beta  = length(phi_beta);
    n_alpha = length(phi_alpha);

    % Parameter sizes
    n_theta_beta  = 2*n_beta;  
    n_theta_alpha = 4*n_alpha;  

    n_params = n_theta_beta + n_theta_alpha;

    if length(theta_vec) ~= n_params
        error('Wrong theta_vec size. Expected %d parameters.', n_params);
    end

    % Split theta vector
    theta_beta_vec  = theta_vec(1:n_theta_beta);
    theta_alpha_vec = theta_vec(n_theta_beta+1:end);

    % Reshape parameters
    Theta_beta  = reshape(theta_beta_vec, 2, n_beta);
    Theta_alpha = reshape(theta_alpha_vec, 4, n_alpha);

    % Learned beta correction
    beta_theta = Theta_beta * phi_beta;   

    % Learned alpha correction
    alpha_entries = Theta_alpha * phi_alpha; 

    % MATLAB column-major order: alpha_entries = [a11; a21; a12; a22]
    alpha_theta = reshape(alpha_entries, 2, 2);

    % Nominal FL torque
    u_nom = beta_m + alpha_m*v;

    % Learned FL torque
    u = (beta_m + beta_theta) + (alpha_m + alpha_theta)*v;

    % Extra learned torque
    u_learned = u - u_nom;

end