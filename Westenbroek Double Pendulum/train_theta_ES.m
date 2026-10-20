function theta_best = train_theta_ES(p, p_nom)

    rng(1);

    % Parameter size
    [phi_beta, phi_alpha] = policy_features([0; 0; 0; 0]);

    n_params = 2*length(phi_beta) + 4*length(phi_alpha);

    theta = zeros(n_params, 1);

    % Training settings
    n_epochs = 200;
    n_rollouts = 18;
    n_tasks_per_epoch = 4;
    n_validation_tasks = 6;

    sigma = 0.02;
    alpha = 0.01;

    theta_limit = 50;
    grad_max_norm = 10;
    J_cap = 1e5;

    % Fixed validation tasks
    validation_tasks = cell(n_validation_tasks, 1);

    for k = 1:n_validation_tasks
        validation_tasks{k} = sample_training_task();
    end

    best_J = batch_policy_cost(theta, p, p_nom, validation_tasks);
    theta_best = theta;

    fprintf('Number of parameters = %d\n', n_params);
    fprintf('Initial validation cost = %.6f\n', best_J);

    % Training loop
    for epoch = 1:n_epochs

        grad_est = zeros(n_params, 1);

        % Same tasks for plus and minus comparisons in this epoch
        tasks = cell(n_tasks_per_epoch, 1);

        for k = 1:n_tasks_per_epoch
            tasks{k} = sample_training_task();
        end

        for i = 1:n_rollouts

            eps_i = randn(n_params, 1);

            theta_plus  = theta + sigma*eps_i;
            theta_minus = theta - sigma*eps_i;

            theta_plus  = max(min(theta_plus,  theta_limit), -theta_limit);
            theta_minus = max(min(theta_minus, theta_limit), -theta_limit);

            J_plus  = batch_policy_cost(theta_plus,  p, p_nom, tasks);
            J_minus = batch_policy_cost(theta_minus, p, p_nom, tasks);

            J_plus  = min(J_plus,  J_cap);
            J_minus = min(J_minus, J_cap);

            % Cost minimization:
            % If J_plus < J_minus, move toward +eps_i.
            grad_est = grad_est + (J_minus - J_plus)*eps_i;

        end

        grad_est = grad_est / (2*n_rollouts*sigma);

        if norm(grad_est) > grad_max_norm
            grad_est = grad_max_norm * grad_est / norm(grad_est);
        end

        theta = theta + alpha*grad_est;
        theta = max(min(theta, theta_limit), -theta_limit);

        J_train = batch_policy_cost(theta, p, p_nom, tasks);
        J_val   = batch_policy_cost(theta, p, p_nom, validation_tasks);

        if J_val < best_J
            best_J = J_val;
            theta_best = theta;
            save('theta_best_ES.mat', 'theta_best', 'best_J');
        end

        fprintf('Epoch %3d | Train J = %.6f | Val J = %.6f | Best Val J = %.6f\n', ...
            epoch, J_train, J_val, best_J);

    end

    save('theta_final_ES.mat', 'theta', 'theta_best', 'best_J');

end