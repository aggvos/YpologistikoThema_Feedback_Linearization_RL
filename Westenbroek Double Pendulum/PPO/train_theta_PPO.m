function [theta_best, history, info] = train_theta_PPO(p, p_nom)

    rng(1, 'twister');

    % Actor size: identical to train_theta_REINFORCE
    [phi_beta, phi_alpha] = policy_features(zeros(4,1));
    n_actor = 2*length(phi_beta) + 4*length(phi_alpha);

    %% Environment and parameters
    cfg.n_iterations = 5500;
    cfg.n_rollouts = 100;
    cfg.dt = 0.01;
    cfg.T = 0.25;
    cfg.N = round(cfg.T/cfg.dt);
    cfg.gamma = 1.0;

    cfg.q_min = [-pi; -pi];
    cfg.q_max = [ pi;  pi];
    cfg.dq_min = [-1.5; -1.5];
    cfg.dq_max = [ 1.5;  1.5];
    cfg.v_radius = 5;

    cfg.Qlin = diag([2, 1]);
    cfg.loss_cap = 1e4;
    cfg.failure_penalty = 100;
    cfg.state_failure_norm = 1e3;

    %% PPO-Clip
    cfg.clip_epsilon = 0.20;
    cfg.n_update_epochs = 10;
    cfg.minibatch_size = 256;
    cfg.target_kl = 0.02;
    cfg.kl_stop_multiplier = 1.5;
    cfg.max_actor_gradient_norm = 5;
    cfg.theta_limit = 50;

    cfg.adam_beta1 = 0.9;
    cfg.adam_beta2 = 0.999;
    cfg.adam_epsilon = 1e-8;

    %% Critic and validation
    cfg.value_ridge = 1e-4;
    cfg.n_validation = 500;
    cfg.validate_every = 10;
    cfg.target_loss = 0.1;

    %% Parameters
    theta = zeros(n_actor,1);
    theta_best = theta;

    phi_value_0 = value_features(zeros(4,1), zeros(2,1), 1, cfg);
    value_weights = zeros(length(phi_value_0),1);

    actor_adam.m = zeros(n_actor,1);
    actor_adam.v = zeros(n_actor,1);
    actor_adam.t = 0;

    %% Fixed held-out validation tasks
    validation_x = zeros(4,cfg.n_validation);
    validation_v = zeros(2,cfg.n_validation);

    for j = 1:cfg.n_validation
        validation_x(:,j) = sample_state(cfg);
        validation_v(:,j) = sample_v(cfg.v_radius);
    end

    [best_loss, best_rmse] = validate_policy( ...
        theta, p, p_nom, cfg, validation_x, validation_v);

    initial_loss = best_loss;
    best_iteration = 0;
    total_environment_steps = 0;

    %% Training history
    history.mean_return = zeros(cfg.n_iterations,1);
    history.actor_objective = zeros(cfg.n_iterations,1);
    history.value_loss = zeros(cfg.n_iterations,1);
    history.explained_variance = zeros(cfg.n_iterations,1);
    history.exact_kl = zeros(cfg.n_iterations,1);
    history.approx_kl = zeros(cfg.n_iterations,1);
    history.clip_fraction = zeros(cfg.n_iterations,1);
    history.gradient_norm = zeros(cfg.n_iterations,1);
    history.sigma = zeros(cfg.n_iterations,1);
    history.actor_learning_rate = zeros(cfg.n_iterations,1);
    history.policy_updates = zeros(cfg.n_iterations,1);
    history.stopped_by_kl = false(cfg.n_iterations,1);
    history.batch_steps = zeros(cfg.n_iterations,1);
    history.environment_steps = zeros(cfg.n_iterations,1);
    history.validation_epoch = [];
    history.validation_loss = [];

    %% Checkpoint files
    script_directory = fileparts(mfilename('fullpath'));
    best_file = fullfile(script_directory, 'theta_best_PPO.mat');
    final_file = fullfile(script_directory, 'theta_final_PPO.mat');
    algorithm = 'PPO-Clip actor-critic with terminal Monte-Carlo returns';
    value_weights_best = value_weights;

    save(best_file, 'theta_best', 'best_loss', 'best_rmse', ...
        'best_iteration', 'value_weights_best', 'algorithm');

    fprintf('PPO-Clip training with %d actor and %d critic parameters\n', ...
        n_actor, length(value_weights));
    fprintf('Initial validation loss: %.6f\n', initial_loss);

    %% On-policy PPO iterations
    for iteration = 1:cfg.n_iterations
        [sigma, actor_learning_rate] = training_schedule(best_loss);

        % Freeze theta_old and collect the complete on-policy batch
        theta_old = theta;
        batch = collect_batch(theta_old, value_weights, sigma, ...
            p, p_nom, cfg);

        total_environment_steps = total_environment_steps + batch.count;

        % A single normalization is used for every repeated PPO epoch
        advantage_mean = mean(batch.advantages);
        advantage_std = std(batch.advantages);

        if ~isfinite(advantage_std) || advantage_std < 1e-8
            advantage_std = 1;
        end

        batch.advantages = ...
            (batch.advantages - advantage_mean)/advantage_std;

        fisher_half = batch.fisher_half;

        accepted_updates = 0;
        gradient_norm_sum = 0;
        stopped_by_kl = false;

        for update_epoch = 1:cfg.n_update_epochs
            order = randperm(batch.count);

            for first = 1:cfg.minibatch_size:batch.count
                last = min(first + cfg.minibatch_size - 1, batch.count);
                indices = order(first:last);

                [gradient, ~] = ppo_objective_gradient( ...
                    theta, theta_old, batch, indices, sigma, cfg);

                raw_gradient_norm = norm(gradient);

                if ~all(isfinite(gradient))
                    stopped_by_kl = true;
                    break;
                end

                if raw_gradient_norm > cfg.max_actor_gradient_norm
                    gradient = cfg.max_actor_gradient_norm ...
                        * gradient/raw_gradient_norm;
                end

                % Form a candidate without changing the accepted Adam
                % state.  A KL-violating step is rejected completely
                [theta_candidate, adam_candidate] = adam_ascent_candidate( ...
                    theta, gradient, actor_adam, actor_learning_rate, cfg);

                theta_candidate = max(min(theta_candidate, ...
                    cfg.theta_limit), -cfg.theta_limit);

                delta_candidate = theta_candidate - theta_old;
                candidate_kl = delta_candidate' ...
                    * fisher_half * delta_candidate;

                if ~isfinite(candidate_kl) || candidate_kl > ...
                        cfg.kl_stop_multiplier*cfg.target_kl
                    stopped_by_kl = true;
                    break;
                end

                theta = theta_candidate;
                actor_adam = adam_candidate;
                accepted_updates = accepted_updates + 1;
                gradient_norm_sum = gradient_norm_sum + raw_gradient_norm;
            end

            if stopped_by_kl
                break;
            end
        end

        % Diagnostics use the whole frozen batch.
        diagnostics = policy_diagnostics( ...
            theta, theta_old, batch, sigma, cfg);

        value_weights = fit_value_function( ...
            batch.value_features, batch.returns, cfg.value_ridge);

        predicted_values = value_weights'*batch.value_features;
        value_residual = predicted_values - batch.returns;
        value_loss = 0.5*mean(value_residual.^2);
        explained_variance = calculate_explained_variance( ...
            batch.returns, predicted_values);

        history.mean_return(iteration) = mean(batch.episode_returns);
        history.actor_objective(iteration) = diagnostics.objective;
        history.value_loss(iteration) = value_loss;
        history.explained_variance(iteration) = explained_variance;
        history.exact_kl(iteration) = diagnostics.exact_kl;
        history.approx_kl(iteration) = diagnostics.approx_kl;
        history.clip_fraction(iteration) = diagnostics.clip_fraction;
        history.gradient_norm(iteration) = ...
            gradient_norm_sum/max(accepted_updates,1);
        history.sigma(iteration) = sigma;
        history.actor_learning_rate(iteration) = actor_learning_rate;
        history.policy_updates(iteration) = accepted_updates;
        history.stopped_by_kl(iteration) = stopped_by_kl;
        history.batch_steps(iteration) = batch.count;
        history.environment_steps(iteration) = total_environment_steps;

        if iteration == 1 || mod(iteration,cfg.validate_every) == 0
            [validation_loss, validation_rmse] = validate_policy( ...
                theta, p, p_nom, cfg, validation_x, validation_v);

            history.validation_epoch(end+1,1) = iteration;
            history.validation_loss(end+1,1) = validation_loss;

            if validation_loss < best_loss
                best_loss = validation_loss;
                best_rmse = validation_rmse;
                best_iteration = iteration;
                theta_best = theta;
                value_weights_best = value_weights; 

                save(best_file, 'theta_best', 'best_loss', 'best_rmse', ...
                    'best_iteration', 'value_weights_best', 'algorithm');
            end

            fprintf([ ...
                'Iteration %4d | return %9.4f | val %10.6f | ' ...
                'best %10.6f | KL %.5f | clip %.3f | ' ...
                'sigma %.2f | updates %d\n'], ...
                iteration, history.mean_return(iteration), ...
                validation_loss, best_loss, diagnostics.exact_kl, ...
                diagnostics.clip_fraction, sigma, accepted_updates);

            if best_loss <= cfg.target_loss
                fprintf('Target validation loss reached at iteration %d.\n', ...
                    iteration);
                break;
            end
        end
    end

    %% Remove unused preallocation and return a self-contained result
    fields_to_trim = { ...
        'mean_return', 'actor_objective', 'value_loss', ...
        'explained_variance', 'exact_kl', 'approx_kl', ...
        'clip_fraction', 'gradient_norm', 'sigma', ...
        'actor_learning_rate', 'policy_updates', 'stopped_by_kl', ...
        'batch_steps', 'environment_steps'};

    for j = 1:length(fields_to_trim)
        field_name = fields_to_trim{j};
        field_value = history.(field_name);
        history.(field_name) = field_value(1:iteration,:);
    end

    info.initial_validation_loss = initial_loss;
    info.best_validation_loss = best_loss;
    info.rmse = best_rmse;
    info.best_iteration = best_iteration;
    info.total_environment_steps = total_environment_steps;
    info.final_theta = theta;
    info.value_weights = value_weights;
    info.config = cfg;
    info.algorithm = algorithm;

    save(final_file, 'theta', 'theta_best', 'value_weights', ...
        'history', 'info');
end


function batch = collect_batch(theta_old, value_weights, sigma, ...
        p, p_nom, cfg)
%COLLECT_BATCH Generate data only with the frozen old policy.

    n_actor = length(theta_old);
    n_value = length(value_weights);
    max_count = cfg.N*cfg.n_rollouts;

    batch.actions = zeros(2,max_count);
    batch.old_means = zeros(2,max_count);
    batch.old_log_probabilities = zeros(1,max_count);
    batch.jacobians = zeros(n_actor,2,max_count);
    batch.value_features = zeros(n_value,max_count);
    batch.rewards = zeros(1,max_count);
    batch.returns = zeros(1,max_count);
    batch.episode_returns = zeros(cfg.n_rollouts,1);

    fisher_sum = zeros(n_actor,n_actor);
    cursor = 0;

    for rollout = 1:cfg.n_rollouts
        x = sample_state(cfg);
        first_index = cursor + 1;

        for k = 1:cfg.N
            v = sample_v(cfg.v_radius);
            [mu, J_mu] = policy_mean_jacobian(x, v, p_nom, theta_old);
            action = mu + sigma*randn(2,1);

            cursor = cursor + 1;
            batch.actions(:,cursor) = action;
            batch.old_means(:,cursor) = mu;
            batch.old_log_probabilities(cursor) = ...
                gaussian_log_probability(action, mu, sigma);
            batch.jacobians(:,:,cursor) = J_mu';
            batch.value_features(:,cursor) = value_features(x, v, k, cfg);

            fisher_sum = fisher_sum + J_mu'*J_mu;

            if any(~isfinite(action))
                batch.rewards(cursor) = -cfg.failure_penalty;
                break;
            end

            x_next = rk4(x, action, p, cfg.dt);

            if any(~isfinite(x_next)) || ...
                    norm(x_next) > cfg.state_failure_norm
                batch.rewards(cursor) = -cfg.failure_penalty;
                break;
            end

            ddq_measured = (x_next(3:4)-x(3:4))/cfg.dt;
            acceleration_error = ddq_measured-v;
            loss = acceleration_error'*cfg.Qlin*acceleration_error;
            batch.rewards(cursor) = -min(loss,cfg.loss_cap)*cfg.dt;

            x = x_next;
        end

        % Terminal Monte-Carlo return.  G starts at zero even when k=N;
        % there is no value bootstrap beyond the sampled episode.
        last_index = cursor;
        G = 0;

        for index = last_index:-1:first_index
            G = batch.rewards(index) + cfg.gamma*G;
            batch.returns(index) = G;
        end

        batch.episode_returns(rollout) = ...
            sum(batch.rewards(first_index:last_index));
    end

    batch.count = cursor;
    batch.actions = batch.actions(:,1:cursor);
    batch.old_means = batch.old_means(:,1:cursor);
    batch.old_log_probabilities = ...
        batch.old_log_probabilities(1:cursor);
    batch.jacobians = batch.jacobians(:,:,1:cursor);
    batch.value_features = batch.value_features(:,1:cursor);
    batch.rewards = batch.rewards(1:cursor);
    batch.returns = batch.returns(1:cursor);

    old_values = value_weights'*batch.value_features;
    batch.advantages = batch.returns-old_values;

    batch.fisher_half = fisher_sum/(2*sigma^2*cursor);
    batch.fisher_half = 0.5*(batch.fisher_half+batch.fisher_half');
end


function [gradient, objective] = ppo_objective_gradient( ...
        theta, theta_old, batch, indices, sigma, cfg)
%PPO_OBJECTIVE_GRADIENT Exact gradient of the clipped surrogate objective.

    n_actor = length(theta);
    n_samples = length(indices);
    J = batch.jacobians(:,:,indices);
    delta = theta-theta_old;

    delta_mu = reshape(sum(bsxfun(@times, J, ...
        reshape(delta,n_actor,1,1)),1),2,n_samples);
    mu = batch.old_means(:,indices)+delta_mu;
    actions = batch.actions(:,indices);

    new_log_probabilities = ...
        gaussian_log_probability_batch(actions, mu, sigma);
    log_ratio = new_log_probabilities ...
        - batch.old_log_probabilities(indices);
    ratio = exp(max(min(log_ratio,40),-40));

    advantages = batch.advantages(indices);
    clipped_ratio = min(max(ratio,1-cfg.clip_epsilon), ...
        1+cfg.clip_epsilon);

    objective = mean(min(ratio.*advantages, ...
        clipped_ratio.*advantages));

    % The clipped branch is locally constant only in the two cases below.
    active = ~((advantages >= 0 & ratio > 1+cfg.clip_epsilon) | ...
               (advantages <  0 & ratio < 1-cfg.clip_epsilon));

    residual_over_variance = (actions-mu)/(sigma^2);
    score = reshape(sum(bsxfun(@times, J, ...
        reshape(residual_over_variance,1,2,n_samples)),2), ...
        n_actor,n_samples);

    sample_weights = ratio.*advantages.*active;
    gradient = mean(bsxfun(@times,score,sample_weights),2);
end


function diagnostics = policy_diagnostics( ...
        theta, theta_old, batch, sigma, cfg)

    indices = 1:batch.count;
    n_actor = length(theta);
    delta = theta-theta_old;
    J = batch.jacobians;

    delta_mu = reshape(sum(bsxfun(@times,J, ...
        reshape(delta,n_actor,1,1)),1),2,batch.count);
    mu = batch.old_means+delta_mu;

    new_log_probabilities = gaussian_log_probability_batch( ...
        batch.actions, mu, sigma);
    log_ratio = new_log_probabilities ...
        - batch.old_log_probabilities(indices);
    ratio = exp(max(min(log_ratio,40),-40));

    clipped_ratio = min(max(ratio,1-cfg.clip_epsilon), ...
        1+cfg.clip_epsilon);
    diagnostics.objective = mean(min( ...
        ratio.*batch.advantages, clipped_ratio.*batch.advantages));
    diagnostics.approx_kl = mean( ...
        batch.old_log_probabilities(indices)-new_log_probabilities);
    diagnostics.exact_kl = delta'*batch.fisher_half*delta;
    diagnostics.clip_fraction = mean( ...
        abs(ratio-1) > cfg.clip_epsilon);
end


function weights = fit_value_function(features, targets, ridge)
%FIT_VALUE_FUNCTION Ridge regression for the linear critic.

    n_samples = size(features,2);
    n_features = size(features,1);

    regularizer = ridge*eye(n_features);
    regularizer(1,1) = 0;  % Do not penalize the intercept.

    normal_matrix = (features*features')/n_samples + regularizer;
    right_hand_side = (features*targets')/n_samples;
    weights = normal_matrix\right_hand_side;

    if any(~isfinite(weights))
        warning('PPO:InvalidCritic', ...
            'Critic fit was non-finite; replacing it with zeros.');
        weights = zeros(n_features,1);
    end
end


function phi = value_features(x, v, k, cfg)
%VALUE_FEATURES Bounded quadratic features, including normalized time-to-go.

    z = [ ...
        sin(x(1));
        cos(x(1));
        sin(x(2));
        cos(x(2));
        tanh(x(3)/1.5);
        tanh(x(4)/1.5);
        v(1)/cfg.v_radius;
        v(2)/cfg.v_radius;
        (cfg.N-k+1)/cfg.N
    ];

    n = length(z);
    phi = zeros(1+n+n*(n+1)/2,1);
    phi(1) = 1;
    phi(2:1+n) = z;

    cursor = 1+n;

    for i = 1:n
        for j = i:n
            cursor = cursor+1;
            phi(cursor) = z(i)*z(j);
        end
    end
end


function [candidate, new_state] = adam_ascent_candidate( ...
        parameters, gradient, old_state, learning_rate, cfg)

    new_state = old_state;
    new_state.t = old_state.t+1;
    new_state.m = cfg.adam_beta1*old_state.m ...
        + (1-cfg.adam_beta1)*gradient;
    new_state.v = cfg.adam_beta2*old_state.v ...
        + (1-cfg.adam_beta2)*(gradient.^2);

    m_hat = new_state.m/(1-cfg.adam_beta1^new_state.t);
    v_hat = new_state.v/(1-cfg.adam_beta2^new_state.t);

    candidate = parameters + learning_rate*m_hat ...
        ./ (sqrt(v_hat)+cfg.adam_epsilon);
end


function [sigma, learning_rate] = training_schedule(best_loss)
%Exploration and step-size schedule for this plant.

    if best_loss > 100
        sigma = 1.2;
        learning_rate = 2e-3;
    % if best_loss > 100
    %     sigma = 0.20;
    %     learning_rate = 3e-5;
    elseif best_loss > 3
        sigma = 1.0;
        learning_rate = 2e-3;
    elseif best_loss > 1
        sigma = 0.4;
        learning_rate = 8e-4;
    elseif best_loss > 0.2
        sigma = 0.2;
        learning_rate = 4e-4;
    else
        sigma = 0.1;
        learning_rate = 2e-4;
    end
end


function [mu, J_mu] = policy_mean_jacobian(x, v, p_nom, theta)
%Actor mean and exact derivative with respect to theta.

    [mu,~,~,~,~,~,~] = learned_fl_controller(x,v,p_nom,theta);
    [phi_beta,phi_alpha] = policy_features(x);

    n_beta = length(phi_beta);
    n_alpha = length(phi_alpha);
    n_beta_parameters = 2*n_beta;
    n_actor = n_beta_parameters+4*n_alpha;

    J_mu = zeros(2,n_actor);

    for j = 1:n_beta
        index = 2*(j-1);
        J_mu(1,index+1) = phi_beta(j);
        J_mu(2,index+2) = phi_beta(j);
    end

    base = n_beta_parameters;

    for j = 1:n_alpha
        index = base+4*(j-1);
        feature = phi_alpha(j);

        J_mu(1,index+1) = feature*v(1);
        J_mu(2,index+2) = feature*v(1);
        J_mu(1,index+3) = feature*v(2);
        J_mu(2,index+4) = feature*v(2);
    end
end


function log_probability = gaussian_log_probability(action, mean, sigma)

    dimension = length(action);
    standardized = (action-mean)/sigma;
    log_probability = -0.5*sum(standardized.^2) ...
        - dimension*log(sigma) ...
        - 0.5*dimension*log(2*pi);
end


function log_probabilities = gaussian_log_probability_batch( ...
        actions, means, sigma)

    dimension = size(actions,1);
    standardized = (actions-means)/sigma;
    log_probabilities = -0.5*sum(standardized.^2,1) ...
        - dimension*log(sigma) ...
        - 0.5*dimension*log(2*pi);
end


function explained_variance = calculate_explained_variance( ...
        targets, predictions)

    target_variance = var(targets,1);

    if target_variance < 1e-12 || ~isfinite(target_variance)
        explained_variance = 0;
    else
        explained_variance = 1-var(targets-predictions,1) ...
            / target_variance;
    end
end


function [mean_loss, joint_rmse] = validate_policy( ...
        theta, p, p_nom, cfg, validation_x, validation_v)

    n = size(validation_x,2);
    errors = zeros(2,n);
    losses = zeros(n,1);

    for j = 1:n
        x = validation_x(:,j);
        v = validation_v(:,j);

        [u,~,~,~,~,~,~] = learned_fl_controller(x,v,p_nom,theta);

        if any(~isfinite(u))
            errors(:,j) = sqrt(cfg.loss_cap/2)*ones(2,1);
            losses(j) = cfg.loss_cap;
            continue;
        end

        x_next = rk4(x,u,p,cfg.dt);

        if any(~isfinite(x_next)) || ...
                norm(x_next) > cfg.state_failure_norm
            errors(:,j) = sqrt(cfg.loss_cap/2)*ones(2,1);
            losses(j) = cfg.loss_cap;
        else
            ddq_measured = (x_next(3:4)-x(3:4))/cfg.dt;
            errors(:,j) = ddq_measured-v;
            losses(j) = errors(:,j)'*cfg.Qlin*errors(:,j);
        end
    end

    mean_loss = mean(losses);
    joint_rmse = sqrt(mean(errors.^2,2));
end


function x = sample_state(cfg)

    q = cfg.q_min+(cfg.q_max-cfg.q_min).*rand(2,1);
    dq = cfg.dq_min+(cfg.dq_max-cfg.dq_min).*rand(2,1);
    x = [q;dq];
end


function v = sample_v(radius)

    direction = randn(2,1);
    direction = direction/(norm(direction)+eps);
    v = radius*sqrt(rand())*direction;
end


function x_next = rk4(x, u, p, dt)

    k1 = double_pendulum_model_state_space(0,x,u,p);
    k2 = double_pendulum_model_state_space(0,x+dt*k1/2,u,p);
    k3 = double_pendulum_model_state_space(0,x+dt*k2/2,u,p);
    k4 = double_pendulum_model_state_space(0,x+dt*k3,u,p);

    x_next = x+(dt/6)*(k1+2*k2+2*k3+k4);
end
