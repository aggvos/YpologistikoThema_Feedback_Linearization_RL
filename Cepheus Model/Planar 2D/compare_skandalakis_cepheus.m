clear; clc; close all;


algorithm = 'PPO';               
% algorithm = 'REINFORCE'; 
ppo_options = struct();       
% ppo_options.n_iterations = 1000;
% ppo_options.episode_steps = 1;

algorithm = upper(strtrim(char(algorithm)));
assert(any(strcmp(algorithm,{'PPO','REINFORCE'})), ...
    'algorithm must be PPO or REINFORCE.');

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
rng(1);

needed = {'cepheus_H','cepheus_c','cepheus_fl_controller', ...
    'cepheus_learned_fl_controller','cepheus_policy_features', ...
    'cepheus_constrained_state_space',['cepheus_train_theta_' algorithm]};

for k = 1:numel(needed)
    assert(exist(needed{k},'file') == 2, ...
        'Missing function: %s',needed{k});
end

p.m0 = 54.00; p.I0 = 2.1837; p.r0 = 0.1645;
p.m1 = 0.23; p.I1 = 6.81e-3; p.l1 = 0.185; p.r1 = 0.185;
p.m2 = 0.10; p.I2 = 1.487e-5; p.l2 = 0.143; p.r2 = 0.143;
p.m3 = 2.50; p.I3 = 6.5262e-2; p.l3 = 0.0411; p.r3 = 0.0261;

% p_nom = p;
% p_nom.m0 = (10/3)*p.m0; p_nom.I0 = (1/3)*p.I0;
% p_nom.m1 = (1/3)*p.m1; p_nom.I1 = (1/30)*p.I1;
% p_nom.m2 = (10/3)*p.m2; p_nom.I2 = (1/3)*p.I2;
% p_nom.m3 = (6/3)*p.m3; p_nom.I3 = (1/23)*p.I3;
% p_nom.l1 = (4)*p.l1; p_nom.l2 = (2)*p.l2; p_nom.l3 = (1/2)*p.l3;
% p_nom.r0 = (1/3)*p.r0; p_nom.r1 = (2)*p.r1; p_nom.r2 = (1)*p.r2; p_nom.r3 = (1.6)*p.r3;

p_nom = p;
p_nom.m0 = 59.40; p_nom.I0 = 2.4021;
p_nom.m1 = 0.22;  p_nom.I1 = 7.491e-3;
p_nom.m2 = 0.11;  p_nom.I2 = 1.4126e-5;
p_nom.m3 = 2.25;  p_nom.I3 = 6.8525e-2;

pack = @(a) [a.m0;a.I0;a.m1;a.I1;a.m2;a.I2; ...
             a.m3;a.I3;a.r0;a.l1;a.r1;a.l2;a.r2;a.l3;a.r3];

params = pack(p);
params_nom = pack(p_nom);

plant.torque_limits = [0.1;0.5;0.5;0.5];
plant.joint_damping = 0.9*ones(3,1);

[phi_beta,phi_alpha] = cepheus_policy_features(zeros(8,1));
assert(numel(phi_beta)==64 && numel(phi_alpha)==7, ...
    'Expected 64 beta and 7 alpha features, including velocity/damping terms.');
feature_probe = [zeros(4,1);0.01;-0.02;0.03;-0.04];
[probe_beta,~] = cepheus_policy_features(feature_probe);
assert(norm(probe_beta(end-3:end)-feature_probe(5:8),Inf)<1e-12, ...
    'Use the current 64-feature version with four linear velocity features.');

theta_zero = zeros(4*numel(phi_beta)+16*numel(phi_alpha),1);

task.T = 50;
task.move_time = 50;
task.sample_time = 0.05;
task.start = deg2rad([0;-20;65;25]);
task.goal = deg2rad([0;80;100;70]);
task.x0 = [deg2rad([0;-15;60;30]);zeros(4,1)];
task.Kp = 0.36*eye(4);
task.Kd = 1.2*eye(4);

assert(task.move_time > 0 && task.T >= task.move_time);
assert(all(plant.torque_limits > 0));

baseline = cell(3,1);
baseline{1} = simulate_case(1,theta_zero,params,params_nom,plant,task);
baseline{2} = simulate_case(2,theta_zero,params,params_nom,plant,task);
baseline{3} = simulate_case(3,theta_zero,params,params_nom,plant,task);

fprintf('Before training: oracle %.6f, nominal %.6f, learned theta=0 %.6f rad^2 s\n', ...
    baseline{1}.cost,baseline{2}.cost,baseline{3}.cost);

assert(abs(baseline{2}.cost-baseline{3}.cost) < ...
       1e-5*max(1,baseline{2}.cost), ...
       'Learned controller at theta=0 does not match nominal FL.');

% A separate run directory prevents overwriting a previous experiment.
run_parent = fullfile(script_dir,['skandalakis_' lower(algorithm) '_runs']);
if ~exist(run_parent,'dir'), mkdir(run_parent); end
run_dir = tempname(run_parent);
mkdir(run_dir);
fprintf('Training method: %s\nRun directory: %s\n',algorithm,run_dir);
[theta_best,history,info] = run_training( ...
    algorithm,params,params_nom,plant,ppo_options,run_dir);

assert(numel(theta_best)==numel(theta_zero) && ...
       isvector(theta_best) && all(isfinite(theta_best(:))), ...
       'Training returned an invalid policy.');
theta_best = theta_best(:);

assert(isequal(info.config.plant.torque_limits(:),plant.torque_limits) && ...
       isequal(info.config.plant.joint_damping(:),plant.joint_damping), ...
       'Training and comparison use different plant settings.');

result = cell(3,1);
result{1} = baseline{1};
result{2} = baseline{2};
result{3} = simulate_case(3,theta_best,params,params_nom,plant,task);

names = {'True FL','Nominal FL','Learned FL'};
colors = [0 0.45 0.74; 0.85 0.33 0.1; 0.5 0.2 0.65];
labels = {'\theta_0','q_1','q_2','q_3'};

fprintf('\nHeld-out acceleration loss: %.6f -> %.6f\n', ...
    info.initial_validation_loss,info.best_validation_loss);

for j = 1:3
    fprintf('%s: tracking cost %.6f rad^2 s | RMS error [deg] %.3f %.3f %.3f %.3f\n', ...
        names{j},result{j}.cost,result{j}.rmse_deg);

    fprintf('  Saturated samples [%%] base/q1/q2/q3: %.1f %.1f %.1f %.1f\n', ...
        result{j}.saturation_percent);
    fprintf('  Max |dq| [rad/s]: %s | max ||v||: %.5f\n', ...
        mat2str(max(abs(result{j}.x(:,5:8)),[],1),4), ...
        max(sqrt(sum(result{j}.virtual_input.^2,2))));
    fprintf('  RMS instantaneous acceleration residual [rad/s^2]: %s\n', ...
        mat2str(result{j}.acceleration_rmse,4));
end

L = plant.torque_limits;
du_raw = zeros(numel(result{2}.t),4);
du_applied = du_raw;

for k = 1:numel(result{2}.t)
    tt = result{2}.t(k);
    xx = result{2}.x(k,:)';

    un = controller(tt,xx,2,theta_best,params,params_nom,plant,task);
    ul = controller(tt,xx,3,theta_best,params,params_nom,plant,task);

    du_raw(k,:) = (ul-un)';
    du_applied(k,:) = ...
        (max(-L,min(L,ul))-max(-L,min(L,un)))';
end

fprintf('Max raw nominal-to-learned torque change [Nm]: %s\n', ...
    mat2str(max(abs(du_raw),[],1),4));

fprintf('Max applied nominal-to-learned torque change [Nm]: %s\n', ...
    mat2str(max(abs(du_applied),[],1),4));


if isfield(info,'total_environment_steps')
    fprintf('PPO training environment steps: %d\n',info.total_environment_steps);
else
    fprintf('REINFORCE maximum training steps: %d (less if rollouts failed).\n', ...
        numel(history.mean_return)*info.config.n_rollouts*info.config.N);
end
cfg = info.config;
learned_x = result{3}.x;
in_state = all(bsxfun(@ge,learned_x(:,1:4),cfg.q_min(:)'-1e-12) & ...
               bsxfun(@le,learned_x(:,1:4),cfg.q_max(:)'+1e-12),2) & ...
           all(bsxfun(@ge,learned_x(:,5:8),cfg.dq_min(:)'-1e-12) & ...
               bsxfun(@le,learned_x(:,5:8),cfg.dq_max(:)'+1e-12),2);
in_v = sqrt(sum(result{3}.virtual_input.^2,2)) <= cfg.v_radius+1e-12;
fprintf('Learned tracking samples outside training state/input region: %.2f%%\n', ...
    100*mean(~(in_state & in_v)));
save(fullfile(run_dir,'comparison_results.mat'), ...
    'algorithm','params','params_nom','plant','task','theta_best', ...
    'history','info','result','baseline','du_raw','du_applied','in_state','in_v');

figure('Name',[algorithm ' learning']);
tiledlayout(2,1);

nexttile;
plot(1:numel(history.mean_return),history.mean_return, ...
    'Color',[0.75 0.75 0.75]);
hold on;
plot(1:numel(history.mean_return), ...
    movmean(history.mean_return,min(25,numel(history.mean_return))), ...
    'b','LineWidth',1.5);
grid on;
ylabel('Average episode reward');

nexttile;
validation_epoch = history.validation_epoch(:);
validation_loss = history.validation_loss(:);
if isempty(validation_epoch) || validation_epoch(1) ~= 0
    validation_epoch = [0;validation_epoch];
    validation_loss = [info.initial_validation_loss;validation_loss];
end
semilogy(validation_epoch,validation_loss,'LineWidth',1.5);
hold on;
if isfield(info,'true_fl_validation_loss')
    yline(info.true_fl_validation_loss,':','True FL finite-step reference');
end
grid on;
ylabel('Held-out acceleration loss');
xlabel([algorithm ' iterations']);

figure('Name','Tracking angles [deg]');
tiledlayout(4,1);

for i = 1:4
    nexttile;
    plot(result{1}.t,rad2deg(result{1}.qd(:,i)), ...
        'k--','LineWidth',1.5);
    hold on;

    for j = 1:3
        plot(result{j}.t,rad2deg(result{j}.x(:,i)), ...
            'Color',colors(j,:),'LineWidth',1.2);
    end

    grid on;
    ylabel([labels{i} ' [deg]']);

    if i == 1
        legend([{'Desired'},names],'Location','best');
    end
end

xlabel('Time [s]');

figure('Name','Tracking error [deg]');
tiledlayout(4,1);

for i = 1:4
    nexttile;
    hold on;

    for j = 1:3
        plot(result{j}.t,rad2deg(result{j}.error(:,i)), ...
            'Color',colors(j,:),'LineWidth',1.2);
    end

    grid on;
    ylabel([labels{i} ' error [deg]']);

    if i == 1
        legend(names,'Location','best');
    end
end

xlabel('Time [s]');

figure('Name','Applied torque [Nm]');
tiledlayout(4,1);

for i = 1:4
    nexttile;
    hold on;

    for j = 1:3
        plot(result{j}.t,result{j}.torque(:,i), ...
            'Color',colors(j,:),'LineWidth',1.2);
    end

    yline( L(i),'k:');
    yline(-L(i),'k:');

    grid on;
    ylabel(sprintf('u_%d [Nm]',i-1));

    if i == 1
        legend(names,'Location','best');
    end
end

xlabel('Time [s]');

function out = simulate_case(kind,theta,params,params_nom,plant,task)

    t_requested = (0:task.sample_time:task.T)';
    if t_requested(end) < task.T
        t_requested(end+1,1) = task.T;
    end

    ode_options = odeset('RelTol',1e-7,'AbsTol',1e-9, ...
                         'MaxStep',task.sample_time);

    [t,x] = ode45( ...
        @(tt,xx) rhs(tt,xx,kind,theta,params,params_nom,plant,task), ...
        t_requested,task.x0,ode_options);

    assert(numel(t)==numel(t_requested) && all(isfinite(x(:))), ...
        'Incomplete simulation.');

    n = numel(t);
    qd = zeros(n,4);
    raw = zeros(n,4);
    torque = zeros(n,4);
    virtual_input = zeros(n,4);
    acceleration_residual = zeros(n,4);

    for k = 1:n
        qd(k,:) = reference(t(k),task)';

        [u,v] = controller(t(k),x(k,:)',kind, ...
                       theta,params,params_nom,plant,task);

        raw(k,:) = u';
        virtual_input(k,:) = v';
        dx = cepheus_constrained_state_space(t(k),x(k,:)',u,params,plant);
        acceleration_residual(k,:) = (dx(5:8)-v)';
        torque(k,:) = ...
            max(-plant.torque_limits,min(plant.torque_limits,u))';
    end

    error_rad = x(:,1:4)-qd;

    out = struct( ...
        't',t,'x',x,'qd',qd,'raw',raw,'torque',torque, ...
        'error',error_rad,'virtual_input',virtual_input, ...
        'acceleration_residual',acceleration_residual, ...
        'acceleration_rmse',sqrt(trapz(t,acceleration_residual.^2,1)/task.T), ... ...
        'cost',trapz(t,sum(error_rad.^2,2)), ...
        'rmse_deg',rad2deg(sqrt(trapz(t,error_rad.^2,1)/task.T)), ...
        'saturation_percent', ...
            100*mean(abs(raw)>plant.torque_limits'+1e-12,1));
end

function dx = rhs(t,x,kind,theta,params,params_nom,plant,task)

    u = controller(t,x,kind,theta,params,params_nom,plant,task);
    dx = cepheus_constrained_state_space(t,x,u,params,plant);

end

function [u,v] = controller(t,x,kind,theta,params,params_nom,plant,task)

    [qd,dqd,ddqd] = reference(t,task);
    v = ddqd-task.Kd*(x(5:8)-dqd)-task.Kp*(x(1:4)-qd);

    switch kind
        case 1
            u = cepheus_fl_controller(x,v,params) + ...
                [0;plant.joint_damping.*x(6:8)];

        case 2
            u = cepheus_fl_controller(x,v,params_nom);

        case 3
            u = cepheus_learned_fl_controller(x,v,params_nom,theta);

        otherwise
            error('Unknown controller.');
    end

    u = u(:);
end

function [qd,dqd,ddqd] = reference(t,task)

    z = min(max(t/task.move_time,0),1);

    s = 10*z^3-15*z^4+6*z^5;
    sd = (30*z^2-60*z^3+30*z^4)/task.move_time;
    sdd = (60*z-180*z^2+120*z^3)/task.move_time^2;

    d = task.goal-task.start;

    qd = task.start+s*d;
    dqd = sd*d;
    ddqd = sdd*d;
end

function [theta_best,history,info] = run_training( ...
        algorithm,params,params_nom,plant,ppo_options,run_dir)
% Function scope ensures restoration even if training raises an exception.
    old_dir = pwd;
    restore_directory = onCleanup(@() cd(old_dir)); 
    cd(run_dir);
    switch algorithm
        case 'PPO'
            ppo_options.output_dir = run_dir;
            [theta_best,history,info] = ...
                cepheus_train_theta_PPOo(params,params_nom,plant,ppo_options);
        case 'REINFORCE'
            [theta_best,history,info] = ...
                cepheus_train_theta_REINFORCE(params,params_nom,plant);
    end
end
