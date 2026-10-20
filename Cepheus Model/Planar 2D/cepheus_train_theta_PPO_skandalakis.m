function [theta_best, history, info] = cepheus_train_theta_PPO_skandalakis(params, params_nom, options)

    if nargin < 3
        options = struct();
    end
    if ~isstruct(options) || ~isscalar(options)
        error('options must be a scalar struct.');
    end
    seed = 1;
    if isfield(options,'seed'), seed = options.seed; end
    validateattributes(seed,{'numeric'}, ...
        {'scalar','integer','nonnegative'},mfilename,'options.seed');
    rng(seed, 'twister');

    %% Actor size: identical to cepheus_train_theta_REINFORCE
    [phi_beta, phi_alpha] = cepheus_policy_features(zeros(8,1));
    n_actor = 4*length(phi_beta) + 16*length(phi_alpha);

    %% Environment and finite-horizon objective
    cfg.n_iterations = 1000;
    cfg.n_rollouts = 200;
    if isfield(options,'n_iterations')
        cfg.n_iterations = options.n_iterations;
    end
    if isfield(options,'n_rollouts')
        cfg.n_rollouts = options.n_rollouts;
    end
    validateattributes(cfg.n_iterations,{'numeric'}, ...
        {'scalar','integer','positive'},mfilename,'options.n_iterations');
    validateattributes(cfg.n_rollouts,{'numeric'}, ...
        {'scalar','integer','positive'},mfilename,'options.n_rollouts');
    if mod(cfg.n_rollouts,2) ~= 0
        error('options.n_rollouts must be even for paired samples.');
    end
    cfg.sigma_schedule = [0.05, 0.025, 0.0125, 0.006, 0.003];
    cfg.learning_rate_schedule = [1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6];
    cfg.loss_stage_thresholds = [10, 2, 0.5, 0.2];
    cfg.patience_validations = 15;
    if isfield(options,'patience_validations')
        cfg.patience_validations = options.patience_validations;
    end
    validateattributes(cfg.patience_validations,{'numeric'}, ...
        {'scalar','integer','positive'},mfilename,'options.patience_validations');
    schedule_stage = 1;
    if isfield(options,'initial_stage')
        schedule_stage = options.initial_stage;
    end
    validateattributes(schedule_stage,{'numeric'}, ...
        {'scalar','integer','>=',1,'<=',numel(cfg.sigma_schedule)}, ...
        mfilename,'options.initial_stage');
    validations_without_improvement = 0;
    cfg.dt = 0.01;
    cfg.T = 0.01;  % Change this to 0.25 for 25 steps, as in REINFORCE.
    cfg.N = round(cfg.T/cfg.dt);
    if isfield(options,'episode_steps')
        cfg.N = options.episode_steps;
        cfg.T = cfg.N*cfg.dt;
    end
    validateattributes(cfg.N,{'numeric'}, ...
        {'scalar','integer','positive'},mfilename,'options.episode_steps');
    cfg.gamma = 1.0;

    cfg.q_min = deg2rad([-10;-30;55;15]);
    cfg.q_max = deg2rad([10;90;110;80]);
    cfg.dq_min = [-0.02;-0.12;-0.12;-0.12];
    cfg.dq_max = -cfg.dq_min;
    cfg.v_radius = 0.03;
    cfg.plant.torque_limits = [0.1;0.5;0.5;0.5];
    cfg.plant.joint_damping = 0.9*ones(3,1);
    if isfield(options,'torque_limits')
        cfg.plant.torque_limits = options.torque_limits(:);
    end
    if isfield(options,'joint_damping')
        cfg.plant.joint_damping = options.joint_damping(:);
    end
    validateattributes(cfg.plant.torque_limits,{'numeric'}, ...
        {'column','numel',4,'positive','finite'},mfilename,'options.torque_limits');
    validateattributes(cfg.plant.joint_damping,{'numeric'}, ...
        {'column','numel',3,'nonnegative','finite'},mfilename,'options.joint_damping');
    cfg.paper_torque_limits = cfg.plant.torque_limits;
    if isfield(options,'v_radius'), cfg.v_radius = options.v_radius; end
    validateattributes(cfg.v_radius,{'numeric'}, ...
        {'scalar','positive','finite'},mfilename,'options.v_radius');

    cfg.Qlin = diag([1, 1, 1, 1]);
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
    cfg.backtracking_factor = 0.5;
    cfg.max_backtracking_steps = 12;

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
    if isfield(options,'initial_theta')
        theta = options.initial_theta(:);
        if numel(theta) ~= n_actor || any(~isfinite(theta))
            error('options.initial_theta must contain %d finite values.',n_actor);
        end
    end
    theta_best = theta;

    phi_value_0 = value_features(zeros(8,1), zeros(4,1), 1, cfg);
    value_weights = zeros(length(phi_value_0),1);

    actor_adam.m = zeros(n_actor,1);
    actor_adam.v = zeros(n_actor,1);
    actor_adam.t = 0;

    %% Fixed held-out validation tasks
    validation_x = zeros(8,cfg.n_validation);
    validation_v = zeros(4,cfg.n_validation);

    for j = 1:cfg.n_validation
        validation_x(:,j) = sample_state(cfg);
        validation_v(:,j) = sample_v(cfg.v_radius);
    end

    [best_loss, best_rmse] = validate_policy( ...
        theta, params, params_nom, cfg, validation_x, validation_v);

    initial_loss = best_loss;
    [true_fl_validation_loss, torque_feasible_fraction] = validate_true_fl( ...
        params, cfg, validation_x, validation_v);
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
    history.schedule_stage = zeros(cfg.n_iterations,1);
    history.policy_updates = zeros(cfg.n_iterations,1);
    history.stopped_by_kl = false(cfg.n_iterations,1);
    history.backtracking_steps = zeros(cfg.n_iterations,1);
    history.max_candidate_kl = zeros(cfg.n_iterations,1);
    history.reward_cap_fraction = zeros(cfg.n_iterations,1);
    history.failed_rollouts = zeros(cfg.n_iterations,1);
    history.max_abs_velocity = zeros(cfg.n_iterations,1);
    history.batch_steps = zeros(cfg.n_iterations,1);
    history.environment_steps = zeros(cfg.n_iterations,1);
    history.validation_epoch = [];
    history.validation_loss = [];

    %% Checkpoint files
    script_directory = fileparts(mfilename('fullpath'));
    checkpoint_tag = 'skandalakis';
    if cfg.N > 1
        checkpoint_tag = sprintf('skandalakis_steps_%d',cfg.N);
    end
    if isfield(options,'initial_theta') && ...
            ~isfield(options,'checkpoint_tag')
        error(['Set options.checkpoint_tag when fine-tuning so the ' ...
            'original best checkpoint is preserved.']);
    end
    if isfield(options,'checkpoint_tag')
        checkpoint_tag = char(options.checkpoint_tag);
    end
    if isempty(regexp(checkpoint_tag,'^[A-Za-z0-9_]+$','once'))
        error('options.checkpoint_tag must contain only letters, numbers and underscores.');
    end
    best_file = fullfile(script_directory, ...
        ['theta_best_cepheus_PPO_' checkpoint_tag '.mat']);
    final_file = fullfile(script_directory, ...
        ['theta_final_cepheus_PPO_' checkpoint_tag '.mat']);
    algorithm = 'PPO-Clip with antithetic one-step episodes and adaptive schedule';
    if cfg.N > 1
        algorithm = sprintf('PPO-Clip with paired %d-step episodes and adaptive schedule',cfg.N);
    end
    value_weights_best = value_weights;

    save(best_file, 'theta_best', 'best_loss', 'best_rmse', ...
        'best_iteration', 'value_weights_best', 'algorithm', ...
        'schedule_stage', 'params', 'params_nom', 'cfg');

    fprintf('PPO-Clip training with %d actor and %d critic parameters\n', ...
        n_actor, length(value_weights));
    fprintf('Initial validation loss: %.6f\n', initial_loss);
    fprintf('True FL finite-step validation loss: %.6f\n', ...
        true_fl_validation_loss);
    fprintf('Unclipped true FL feasibility in training domain: %.2f%%.\n', ...
        100*torque_feasible_fraction);
    if cfg.N == 1
        fprintf('Paired one-step samples; %d transitions per iteration.\n', ...
            cfg.n_rollouts);
    else
        fprintf('Paired %d-step episodes; %d trajectories, up to %d transitions per iteration.\n', ...
            cfg.N,cfg.n_rollouts,cfg.N*cfg.n_rollouts);
    end

    %% On-policy PPO iterations
    for iteration = 1:cfg.n_iterations
        [sigma, actor_learning_rate] = training_schedule(cfg, schedule_stage);

        % Freeze theta_old and collect the complete on-policy batch.
        theta_old = theta;
        batch = collect_batch(theta_old, value_weights, sigma, ...
            params, params_nom, cfg);

        total_environment_steps = total_environment_steps + batch.count;

        % A single normalization is used for every repeated PPO epoch.
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
        total_backtracking_steps = 0;
        max_candidate_kl = 0;

        for update_epoch = 1:cfg.n_update_epochs
            % Keep +/- actions from the same context in one minibatch.
            pair_order = randperm(batch.count/2);
            order = reshape([2*pair_order-1; 2*pair_order],1,[]);

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

                % Form an Adam candidate without changing the accepted
                % optimizer state. The policy change depends on both the
                % sampled features (including squared velocities) and sigma.
                [raw_candidate, adam_candidate] = adam_ascent_candidate( ...
                    theta, gradient, actor_adam, actor_learning_rate, cfg);

                raw_candidate = max(min(raw_candidate, ...
                    cfg.theta_limit), -cfg.theta_limit);

                raw_step = raw_candidate-theta;
                raw_delta = raw_candidate-theta_old;
                raw_candidate_kl = raw_delta'*fisher_half*raw_delta;

                if isfinite(raw_candidate_kl)
                    max_candidate_kl = max(max_candidate_kl, ...
                        raw_candidate_kl);
                else
                    max_candidate_kl = Inf;
                end

                kl_limit = cfg.kl_stop_multiplier*cfg.target_kl;
                candidate_accepted = isfinite(raw_candidate_kl) && ...
                    raw_candidate_kl <= kl_limit;
                theta_candidate = raw_candidate;
                candidate_kl = raw_candidate_kl;
                backtracking_step = 0;

                if ~candidate_accepted && accepted_updates == 0
                    for backtracking_step = 1:cfg.max_backtracking_steps
                        step_scale = ...
                            cfg.backtracking_factor^backtracking_step;
                        theta_candidate = theta+step_scale*raw_step;

                        delta_candidate = theta_candidate-theta_old;
                        candidate_kl = delta_candidate' ...
                            * fisher_half*delta_candidate;

                        if isfinite(candidate_kl) && ...
                                candidate_kl <= kl_limit
                            candidate_accepted = true;
                            break;
                        end
                    end
                end

                total_backtracking_steps = total_backtracking_steps ...
                    + backtracking_step;

                if ~candidate_accepted
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

        % Fit V_w(x,v,k) to sampled returns. The
        % newly fitted critic becomes the baseline for the next batch.
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
        history.schedule_stage(iteration) = schedule_stage;
        history.policy_updates(iteration) = accepted_updates;
        history.stopped_by_kl(iteration) = stopped_by_kl;
        history.backtracking_steps(iteration) = total_backtracking_steps;
        history.max_candidate_kl(iteration) = max_candidate_kl;
        history.reward_cap_fraction(iteration) = batch.reward_cap_fraction;
        history.failed_rollouts(iteration) = batch.failed_rollouts;
        history.max_abs_velocity(iteration) = batch.max_abs_velocity;
        history.batch_steps(iteration) = batch.count;
        history.environment_steps(iteration) = total_environment_steps;

        if iteration == 1 || mod(iteration,cfg.validate_every) == 0
            [validation_loss, validation_rmse] = validate_policy( ...
                theta, params, params_nom, cfg, validation_x, validation_v);

            history.validation_epoch(end+1,1) = iteration;
            history.validation_loss(end+1,1) = validation_loss;

            improved = validation_loss < best_loss;
            if improved
                best_loss = validation_loss;
                best_rmse = validation_rmse;
                best_iteration = iteration;
                theta_best = theta;
                value_weights_best = value_weights; 

                save(best_file, 'theta_best', 'best_loss', 'best_rmse', ...
                    'best_iteration', 'value_weights_best', 'algorithm', ...
                    'schedule_stage', 'params', 'params_nom', 'cfg');
                validations_without_improvement = 0;
            else
                validations_without_improvement = ...
                    validations_without_improvement + 1;
            end

            fprintf([ ...
                'Iteration %4d | return %9.4f | val %10.6f | ' ...
                'best %10.6f | KL %.5f | clip %.3f | ' ...
                'sigma %.3f | lr %.1e | updates %d | ' ...
                'reward cap %.1f%% | failed %d\n'], ...
                iteration, history.mean_return(iteration), ...
                validation_loss, best_loss, diagnostics.exact_kl, ...
                diagnostics.clip_fraction, sigma, actor_learning_rate, ...
                accepted_updates, 100*batch.reward_cap_fraction, ...
                batch.failed_rollouts);

            if best_loss <= cfg.target_loss
                fprintf('Target validation loss reached at iteration %d.\n', ...
                    iteration);
                break;
            end

            % Thresholds and sustained plateaus move toward lower exploration.
            % Only the next on-policy batch uses the new sigma and rate.
            threshold_stage = 1 + sum(best_loss <= cfg.loss_stage_thresholds);
            next_stage = max(schedule_stage, threshold_stage);
            plateau = validations_without_improvement >= ...
                cfg.patience_validations;
            if plateau
                next_stage = min(next_stage+1,numel(cfg.sigma_schedule));
                validations_without_improvement = 0;
            end
            if next_stage > schedule_stage
                schedule_stage = next_stage;
                if plateau

                    theta = theta_best;
                    actor_adam.m(:) = 0;
                    actor_adam.v(:) = 0;
                    actor_adam.t = 0;
                    fprintf('Plateau: restoring best policy before next batch.\n');
                end
                fprintf('Next PPO batch: stage %d, sigma %.4f, lr %.1e.\n', ...
                    schedule_stage, cfg.sigma_schedule(schedule_stage), ...
                    cfg.learning_rate_schedule(schedule_stage));
            end
        end
    end

    %% Remove unused preallocation and return a self-contained result
    fields_to_trim = { ...
        'mean_return', 'actor_objective', 'value_loss', ...
        'explained_variance', 'exact_kl', 'approx_kl', ...
        'clip_fraction', 'gradient_norm', 'sigma', ...
        'actor_learning_rate', 'schedule_stage', 'policy_updates', 'stopped_by_kl', ...
        'backtracking_steps', 'max_candidate_kl', ...
        'reward_cap_fraction', 'failed_rollouts', 'max_abs_velocity', ...
        'batch_steps', 'environment_steps'};

    for j = 1:length(fields_to_trim)
        field_name = fields_to_trim{j};
        field_value = history.(field_name);
        history.(field_name) = field_value(1:iteration,:);
    end

    info.initial_validation_loss = initial_loss;
    info.true_fl_validation_loss = true_fl_validation_loss;
    info.torque_feasible_fraction = torque_feasible_fraction;
    info.best_validation_loss = best_loss;
    info.rmse = best_rmse;
    info.best_iteration = best_iteration;
    info.total_environment_steps = total_environment_steps;
    info.final_theta = theta;
    info.value_weights = value_weights;
    info.config = cfg;
    info.algorithm = algorithm;
    info.checkpoint_tag = checkpoint_tag;
    info.final_schedule_stage = schedule_stage;
    info.objective = ['finite-step one-step acceleration error, ' ...
        'training loss capped at cfg.loss_cap, plant torques saturated'];
    if cfg.N > 1
        info.objective = ['finite-step acceleration error summed across episode, ' ...
            'training loss capped at cfg.loss_cap, plant torques saturated'];
    end

    save(final_file, 'theta', 'theta_best', 'value_weights', ...
        'history', 'info', 'params', 'params_nom');
end


function batch = collect_batch(theta_old, value_weights, sigma, ...
        params, params_nom, cfg)
%COLLECT_BATCH Paired on-policy data from identical simulator contexts.
    if cfg.N > 1
        batch = collect_batch_multistep(theta_old,value_weights,sigma, ...
            params,params_nom,cfg);
        return;
    end

    n_actor = length(theta_old);
    n_value = length(value_weights);
    max_count = cfg.n_rollouts;

    batch.actions = zeros(4,max_count);
    batch.old_means = zeros(4,max_count);
    batch.old_log_probabilities = zeros(1,max_count);
    batch.jacobians = zeros(n_actor,4,max_count);
    batch.value_features = zeros(n_value,max_count);
    batch.rewards = zeros(1,max_count);
    batch.returns = zeros(1,max_count);
    batch.episode_returns = zeros(cfg.n_rollouts,1);
    batch.failed_rollouts = 0;
    batch.max_abs_velocity = 0;
    reward_cap_count = 0;

    fisher_sum = zeros(n_actor,n_actor);
    cursor = 0;

    for pair = 1:cfg.n_rollouts/2
        x = sample_state(cfg);
        v = sample_v(cfg.v_radius);
        [mu, J_mu] = policy_mean_jacobian(x, v, params_nom, theta_old);
        if any(~isfinite(mu)) || any(~isfinite(J_mu(:)))
            error('Nonfinite policy mean/Jacobian in pair %d.',pair);
        end
        noise = sigma*randn(4,1);
        context_value_features = value_features(x, v, 1, cfg);

        fisher_sum = fisher_sum + 2*(J_mu'*J_mu);
        batch.max_abs_velocity = max(batch.max_abs_velocity, ...
            max(abs(x(5:8))));


        for sign_index = 1:2
            sign_value = 3-2*sign_index;  % +1, then -1
            action = mu + sign_value*noise;

            cursor = cursor + 1;
            batch.actions(:,cursor) = action;
            batch.old_means(:,cursor) = mu;
            batch.old_log_probabilities(cursor) = ...
                gaussian_log_probability(action, mu, sigma);
            batch.jacobians(:,:,cursor) = J_mu';
            batch.value_features(:,cursor) = context_value_features;

            x_next = rk4(x, action, params, cfg.dt, cfg.plant);

            if any(~isfinite(x_next)) || ...
                    norm(x_next) > cfg.state_failure_norm
                batch.rewards(cursor) = -cfg.failure_penalty;
                batch.failed_rollouts = batch.failed_rollouts + 1;
            else
                ddq_measured = (x_next(5:8)-x(5:8))/cfg.dt;
                acceleration_error = ddq_measured-v;
                loss = acceleration_error'*cfg.Qlin*acceleration_error;
                if isfinite(loss)
                    reward_cap_count = reward_cap_count + ...
                        (loss >= cfg.loss_cap);
                    batch.rewards(cursor) = -min(loss,cfg.loss_cap)*cfg.dt;
                else
                    batch.rewards(cursor) = -cfg.failure_penalty;
                    batch.failed_rollouts = batch.failed_rollouts + 1;
                end
            end
            batch.returns(cursor) = batch.rewards(cursor);
            batch.episode_returns(cursor) = batch.rewards(cursor);
        end
    end

    batch.count = cursor;
    batch.reward_cap_fraction = reward_cap_count/cursor;
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


function batch = collect_batch_multistep(theta_old, value_weights, sigma, ...
        params, params_nom, cfg)
%COLLECT_BATCH_MULTISTEP Paired trajectories with Monte-Carlo returns.

    n_actor = length(theta_old);
    n_value = length(value_weights);
    max_count = cfg.n_rollouts*cfg.N;

    batch.actions = zeros(4,max_count);
    batch.old_means = zeros(4,max_count);
    batch.old_log_probabilities = zeros(1,max_count);
    batch.jacobians = zeros(n_actor,4,max_count);
    batch.value_features = zeros(n_value,max_count);
    batch.rewards = zeros(1,max_count);
    batch.returns = zeros(1,max_count);
    batch.episode_returns = zeros(cfg.n_rollouts,1);
    batch.failed_rollouts = 0;
    batch.max_abs_velocity = 0;
    reward_cap_count = 0;

    fisher_sum = zeros(n_actor,n_actor);
    cursor = 0;

    for pair = 1:cfg.n_rollouts/2
        x0 = sample_state(cfg);
        x = [x0, x0];
        first_index = cursor+1;
        for k = 1:cfg.N
            % Common random inputs make the trajectories comparable. Each
            % individual rollout still sees Gaussian noise at every step.
            v = sample_v(cfg.v_radius);
            noise = sigma*randn(4,1);
            x_next = zeros(8,2);
            pair_failed = false;

            for sign_index = 1:2
                sign_value = 3-2*sign_index;  % +1, then -1
                state = x(:,sign_index);
                [mu, J_mu] = policy_mean_jacobian( ...
                    state, v, params_nom, theta_old);
                if any(~isfinite(mu)) || any(~isfinite(J_mu(:)))
                    error('Nonfinite policy mean/Jacobian at pair %d, step %d.', ...
                        pair,k);
                end
                action = mu+sign_value*noise;
                cursor = cursor+1;
                batch.actions(:,cursor) = action;
                batch.old_means(:,cursor) = mu;
                batch.old_log_probabilities(cursor) = ...
                    gaussian_log_probability(action,mu,sigma);
                batch.jacobians(:,:,cursor) = J_mu';
                batch.value_features(:,cursor) = value_features(state,v,k,cfg);
                fisher_sum = fisher_sum+J_mu'*J_mu;
                batch.max_abs_velocity = max(batch.max_abs_velocity, ...
                    max(abs(state(5:8))));

                if any(~isfinite(action))
                    failed = true;
                else
                    x_next(:,sign_index) = rk4(state,action,params,cfg.dt,cfg.plant);
                    failed = any(~isfinite(x_next(:,sign_index))) || ...
                        norm(x_next(:,sign_index))>cfg.state_failure_norm;
                end
                if failed
                    batch.rewards(cursor) = -cfg.failure_penalty;
                    batch.failed_rollouts = batch.failed_rollouts+1;
                    pair_failed = true;
                else
                    ddq = (x_next(5:8,sign_index)-state(5:8))/cfg.dt;
                    error_acc = ddq-v;
                    loss = error_acc'*cfg.Qlin*error_acc;
                    if isfinite(loss)
                        reward_cap_count = reward_cap_count+(loss>=cfg.loss_cap);
                        batch.rewards(cursor) = -min(loss,cfg.loss_cap)*cfg.dt;
                    else
                        batch.rewards(cursor) = -cfg.failure_penalty;
                        batch.failed_rollouts = batch.failed_rollouts+1;
                        pair_failed = true;
                    end
                end
            end
            if pair_failed
                % End both rollouts so every stored transition has a mate
                % in the same PPO minibatch. Failures are separately logged.
                break;
            end
            x = x_next;
        end
        last_index = cursor;
        for sign_index = 1:2
            G = 0;
            for index = last_index-2+sign_index:-2:first_index-1+sign_index
                G = batch.rewards(index)+cfg.gamma*G;
                batch.returns(index) = G;
            end
            batch.episode_returns(2*pair-2+sign_index) = ...
                sum(batch.rewards(first_index-1+sign_index:2:last_index));
        end
    end

    batch.count = cursor;
    batch.reward_cap_fraction = reward_cap_count/cursor;
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
%Exact gradient of the clipped surrogate objective.

    n_actor = length(theta);
    n_samples = length(indices);
    J = batch.jacobians(:,:,indices);
    delta = theta-theta_old;

    delta_mu = reshape(sum(bsxfun(@times, J, ...
        reshape(delta,n_actor,1,1)),1),4,n_samples);
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
        reshape(residual_over_variance,1,4,n_samples)),2), ...
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
        reshape(delta,n_actor,1,1)),1),4,batch.count);
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
%Ridge regression for the linear critic.

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
%Original 153-feature critic at N=1; time-to-go at N>1.

    z = [ ...
        sin(x(1));
        cos(x(1));
        sin(x(2));
        cos(x(2));
        sin(x(3));
        cos(x(3));
        sin(x(4));
        cos(x(4));
        tanh(x(5)/1.5);
        tanh(x(6)/1.5);
        tanh(x(7)/1.5);
        tanh(x(8)/1.5);
        v(1)/cfg.v_radius;
        v(2)/cfg.v_radius;
        v(3)/cfg.v_radius;
        v(4)/cfg.v_radius
    ];
    if cfg.N > 1
        z = [z; (cfg.N-k+1)/cfg.N];
    end

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


function [sigma, learning_rate] = training_schedule(cfg, stage)
    sigma = cfg.sigma_schedule(stage); % N.m, fixed inside this PPO batch
    learning_rate = cfg.learning_rate_schedule(stage);
end


function [mu, J_mu] = policy_mean_jacobian(x, v, params_nom, theta)
%Actor mean and exact derivative with respect to theta.

    [mu,~,~,~,~,~,~] = cepheus_learned_fl_controller(x,v,params_nom,theta);
    [phi_beta,phi_alpha] = cepheus_policy_features(x);

    n_beta = length(phi_beta);
    n_alpha = length(phi_alpha);
    n_beta_parameters = 4*n_beta;
    n_actor = n_beta_parameters + 16*n_alpha;

    J_mu = zeros(4,n_actor);

    for j = 1:n_beta
        index = 4*(j-1);
        for row = 1:4
            J_mu(row,index+row) = phi_beta(j);
        end
    end

    base = n_beta_parameters;

    for j = 1:n_alpha
        index = base + 16*(j-1);
        feature = phi_alpha(j);

        % Match reshape(alpha_entries,4,4): entries are column-major.
        for col = 1:4
            for row = 1:4
                J_mu(row,index+4*(col-1)+row) = feature*v(col);
            end
        end
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
        theta, params, params_nom, cfg, validation_x, validation_v)

    n = size(validation_x,2);
    errors = zeros(4,n);
    losses = zeros(n,1);

    for j = 1:n
        x = validation_x(:,j);
        v = validation_v(:,j);

        [u,~,~,~,~,~,~] = cepheus_learned_fl_controller(x,v,params_nom,theta);

        if any(~isfinite(u))
            errors(:,j) = sqrt(cfg.loss_cap/trace(cfg.Qlin))*ones(4,1);
            losses(j) = cfg.loss_cap;
            continue;
        end

        x_next = rk4(x,u,params,cfg.dt,cfg.plant);

        if any(~isfinite(x_next)) || ...
                norm(x_next) > cfg.state_failure_norm
            errors(:,j) = sqrt(cfg.loss_cap/trace(cfg.Qlin))*ones(4,1);
            losses(j) = cfg.loss_cap;
        else
            ddq_measured = (x_next(5:8)-x(5:8))/cfg.dt;
            errors(:,j) = ddq_measured-v;
            losses(j) = errors(:,j)'*cfg.Qlin*errors(:,j);
        end
    end

    mean_loss = mean(losses);
    joint_rmse = sqrt(mean(errors.^2,2));
end


function [mean_loss, feasible_fraction] = validate_true_fl( ...
        params, cfg, validation_x, validation_v)
% Exact continuous-time FL benchmark and paper-limit feasibility diagnostic.
    n = size(validation_x,2);
    losses = zeros(n,1);
    feasible = false(n,1);
    for j = 1:n
        x = validation_x(:,j);
        v = validation_v(:,j);
        u = cepheus_fl_controller(x,v,params) ...
            + [0;cfg.plant.joint_damping.*x(6:8)];
        if any(~isfinite(u))
            losses(j) = cfg.loss_cap;
            continue;
        end
        feasible(j) = all(abs(u) <= cfg.paper_torque_limits);
        x_next = rk4(x,u,params,cfg.dt,cfg.plant);
        if any(~isfinite(x_next)) || ...
                norm(x_next) > cfg.state_failure_norm
            losses(j) = cfg.loss_cap;
        else
            error_acc = (x_next(5:8)-x(5:8))/cfg.dt-v;
            losses(j) = error_acc'*cfg.Qlin*error_acc;
        end
    end
    mean_loss = mean(losses);
    feasible_fraction = mean(feasible);
end


function x = sample_state(cfg)

    q = cfg.q_min+(cfg.q_max-cfg.q_min).*rand(4,1);
    dq = cfg.dq_min+(cfg.dq_max-cfg.dq_min).*rand(4,1);
    x = [q;dq];
end


function v = sample_v(radius)

    direction = randn(4,1);
    direction = direction/(norm(direction)+eps);
    v = radius*rand()^(1/4)*direction;
end


function x_next = rk4(x, u, params, dt, plant)

    k1 = cepheus_constrained_state_space(0,x,u,params,plant);
    k2 = cepheus_constrained_state_space(0,x+dt*k1/2,u,params,plant);
    k3 = cepheus_constrained_state_space(0,x+dt*k2/2,u,params,plant);
    k4 = cepheus_constrained_state_space(0,x+dt*k3,u,params,plant);

    x_next = x+(dt/6)*(k1+2*k2+2*k3+k4);
end
