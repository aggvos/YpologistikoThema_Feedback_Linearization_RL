function [theta_best, history, info] = cepheus_train_theta_REINFORCE_skandalakis(params, params_nom, options)
% Cepheus: x=[theta0;q1;q2;q3;dtheta0;dq1;dq2;dq3], with four torque inputs


if nargin < 3, options = struct(); end
seed = 1;
if isfield(options,'seed'), seed = options.seed; end
rng(seed);

[phi_beta, phi_alpha] = cepheus_policy_features(zeros(8,1));
n_params = 4*length(phi_beta) + 16*length(phi_alpha);


cfg.n_epochs = 3000;
cfg.n_rollouts = 200;
cfg.dt = 0.01;
cfg.T = 0.01;
cfg.N = round(cfg.T/cfg.dt);
if isfield(options,'episode_steps'), cfg.N = options.episode_steps; end
if isfield(options,'n_epochs'), cfg.n_epochs = options.n_epochs; end
if isfield(options,'n_rollouts'), cfg.n_rollouts = options.n_rollouts; end
validateattributes(cfg.N,{'numeric'},{'scalar','integer','positive'});
validateattributes(cfg.n_epochs,{'numeric'},{'scalar','integer','positive'});
validateattributes(cfg.n_rollouts,{'numeric'},{'scalar','integer','positive'});
cfg.T = cfg.N*cfg.dt;
cfg.gamma = 1;

cfg.q_min = deg2rad([-10;-30;55;15]);
cfg.q_max = deg2rad([10;90;110;80]);
cfg.dq_min = [-0.02;-0.12;-0.12;-0.12];
cfg.dq_max = -cfg.dq_min;
cfg.v_radius = 0.03;
if isfield(options,'v_radius'), cfg.v_radius = options.v_radius; end
cfg.plant.torque_limits = [0.1;0.5;0.5;0.5];
cfg.plant.joint_damping = 0.9*ones(3,1);
if isfield(options,'torque_limits')
    cfg.plant.torque_limits = options.torque_limits(:);
end
if isfield(options,'joint_damping')
    cfg.plant.joint_damping = options.joint_damping(:);
end
validateattributes(cfg.v_radius,{'numeric'},{'scalar','positive','finite'});
validateattributes(cfg.plant.torque_limits,{'numeric'}, ...
    {'column','numel',4,'positive','finite'});
validateattributes(cfg.plant.joint_damping,{'numeric'}, ...
    {'column','numel',3,'nonnegative','finite'});

cfg.target_loss = 0.1;

cfg.grad_max = 10;
cfg.theta_limit = 50;
cfg.loss_cap = 1e4;
cfg.failure_penalty = 100;
cfg.Qlin = eye(4);
% cfg.Qlin = diag([2,1,1,1]);

cfg.n_validation = 500;
cfg.validate_every = 10;

theta = zeros(n_params,1);
theta_best = theta;

validation_x = zeros(8,cfg.n_validation);
validation_v = zeros(4,cfg.n_validation);

for j = 1:cfg.n_validation
    validation_x(:,j) = sample_state(cfg);
    validation_v(:,j) = sample_v(cfg.v_radius);
end

[best_loss, best_rmse] = validate_policy(theta,params,params_nom,cfg,validation_x,validation_v);
initial_loss = best_loss;

history.mean_return = zeros(cfg.n_epochs,1);
history.gradient_norm = zeros(cfg.n_epochs,1);
history.validation_epoch = [];
history.validation_loss = [];

adam_m = zeros(n_params,1);
adam_s = zeros(n_params,1);
beta1 = 0.9;
beta2 = 0.999;
adam_epsilon = 1e-8;

fprintf('REINFORCE training with %d parameters\n',n_params);
fprintf('Initial validation loss: %.6f\n',initial_loss);

for epoch = 1:cfg.n_epochs
    if best_loss > 10
        alpha = 1e-4; sigma = 0.05;
    elseif best_loss > 2
        alpha = 5e-5; sigma = 0.025;
    elseif best_loss > 0.5
        alpha = 2.5e-5; sigma = 0.0125;
    elseif best_loss > 0.2
        alpha = 1e-5; sigma = 0.006;
    else
        alpha = 5e-6; sigma = 0.003;
    end

    rewards = zeros(cfg.N,cfg.n_rollouts);
    scores = zeros(n_params,cfg.N,cfg.n_rollouts);

    for rollout = 1:cfg.n_rollouts
        [rewards(:,rollout),scores(:,:,rollout)] = training_rollout(theta,params,params_nom,cfg,sigma);
    end

    returns = zeros(size(rewards));

    for rollout = 1:cfg.n_rollouts
        G = 0;

        for k = cfg.N:-1:1
            G = rewards(k,rollout) + cfg.gamma*G;
            returns(k,rollout) = G;
        end
    end

    baseline = mean(returns,2);
    advantages = returns - baseline;
    advantages = advantages/(std(advantages(:)) + 1e-8);

    gradient = zeros(n_params,1);

    for rollout = 1:cfg.n_rollouts
        gradient = gradient + sum(scores(:,:,rollout).*advantages(:,rollout)',2);
    end

    gradient = gradient/cfg.n_rollouts;
    gradient_norm = norm(gradient);

    history.gradient_norm(epoch) = gradient_norm;
    history.mean_return(epoch) = mean(sum(rewards,1));

    if gradient_norm > cfg.grad_max
        gradient = cfg.grad_max*gradient/gradient_norm;
    end

    adam_m = beta1*adam_m + (1-beta1)*gradient;
    adam_s = beta2*adam_s + (1-beta2)*(gradient.^2);

    adam_m_hat = adam_m/(1-beta1^epoch);
    adam_s_hat = adam_s/(1-beta2^epoch);

    theta = theta + alpha*adam_m_hat./(sqrt(adam_s_hat) + adam_epsilon);
    theta = max(min(theta,cfg.theta_limit),-cfg.theta_limit);

    if epoch == 1 || mod(epoch,cfg.validate_every) == 0

        [validation_loss,validation_rmse] = validate_policy(theta,params,params_nom,cfg,validation_x,validation_v);

        history.validation_epoch(end+1,1) = epoch;
        history.validation_loss(end+1,1) = validation_loss;

        if validation_loss < best_loss
            best_loss = validation_loss;
            best_rmse = validation_rmse;
            theta_best = theta;

            save('theta_best_cepheus_REINFORCE_skandalakis.mat', ...
                'theta_best','best_loss','best_rmse','epoch','params','params_nom','cfg');
        end

        fprintf('Epoch %4d | sigma %.3f | alpha %.5f | return %.4f | val %.6f | best %.6f\n',...
            epoch,sigma,alpha,history.mean_return(epoch),validation_loss,best_loss);
        if best_loss <= cfg.target_loss
            fprintf('Target validation loss reached at epoch %d.\n',epoch);
            break;
        end
    end
end


history.mean_return = history.mean_return(1:epoch);
history.gradient_norm = history.gradient_norm(1:epoch);

info.initial_validation_loss = initial_loss;
info.best_validation_loss = best_loss;
info.rmse = best_rmse;
info.config = cfg;

save('theta_final_cepheus_REINFORCE_skandalakis.mat', ...
    'theta','theta_best','history','info','params','params_nom');

end


function [rewards,scores] = training_rollout(theta,params,params_nom,cfg,sigma)

n_params = length(theta);

rewards = zeros(cfg.N,1);
scores = zeros(n_params,cfg.N);

x = sample_state(cfg);

for k = 1:cfg.N

    v = sample_v(cfg.v_radius);
    [mu,J_mu] = policy_mean_jacobian(x,v,params_nom,theta);

    u = mu + sigma*randn(4,1);
    scores(:,k) = J_mu'*(u-mu)/(sigma^2);

    x_next = rk4(x,u,params,cfg.dt,cfg.plant);

    if any(~isfinite(x_next)) || norm(x_next) > 1e3
        rewards(k) = -cfg.failure_penalty;
        break;
    end

    ddq_measured = (x_next(5:8)-x(5:8))/cfg.dt;
    acceleration_error = ddq_measured-v;

    loss = acceleration_error'*cfg.Qlin*acceleration_error;
    rewards(k) = -min(loss,cfg.loss_cap)*cfg.dt;

    x = x_next;
end

end


function [mu,J_mu] = policy_mean_jacobian(x,v,params_nom,theta)

[mu,~,~,~,~,~,~] = cepheus_learned_fl_controller(x,v,params_nom,theta);
[phi_beta,phi_alpha] = cepheus_policy_features(x);

n_beta = length(phi_beta);
n_alpha = length(phi_alpha);
n_beta_parameters = 4*n_beta;
n_params = n_beta_parameters + 16*n_alpha;

J_mu = zeros(4,n_params);

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


function [mean_loss,joint_rmse] = validate_policy(theta,params,params_nom,cfg,validation_x,validation_v)

n = size(validation_x,2);
errors = zeros(4,n);
losses = zeros(n,1);

for j = 1:n

    x = validation_x(:,j);
    v = validation_v(:,j);

    [u,~,~,~,~,~,~] = cepheus_learned_fl_controller(x,v,params_nom,theta);
    x_next = rk4(x,u,params,cfg.dt,cfg.plant);

    if any(~isfinite(x_next)) || norm(x_next) > 1e3
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


function x = sample_state(cfg)

q = cfg.q_min + (cfg.q_max-cfg.q_min).*rand(4,1);
dq = cfg.dq_min + (cfg.dq_max-cfg.dq_min).*rand(4,1);

x = [q;dq];

end


function v = sample_v(radius)

direction = randn(4,1);
direction = direction/(norm(direction) + eps);

v = radius*rand()^(1/4)*direction;

end


function x_next = rk4(x,u,params,dt,plant)

k1 = cepheus_constrained_state_space(0,x,u,params,plant);
k2 = cepheus_constrained_state_space(0,x + dt*k1/2,u,params,plant);
k3 = cepheus_constrained_state_space(0,x + dt*k2/2,u,params,plant);
k4 = cepheus_constrained_state_space(0,x + dt*k3,u,params,plant);

x_next = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);

end
