clear; clc; close all;

% addpath(genpath(fileparts(mfilename('fullpath'))));
addpath(genpath(pwd));
rng(1);

% True plant parameters

p.m0 = 54.00; p.I0 = 2.1837; p.r0 = 0.1645;
p.m1 = 0.23;  p.I1 = 6.81e-3;
p.m2 = 0.10;  p.I2 = 1.487e-5;
p.m3 = 2.50;  p.I3 = 6.5262e-2;

p.l1 = 0.185;  p.r1 = 0.185;
p.l2 = 0.143;  p.r2 = 0.143;
p.l3 = 0.0411; p.r3 = 0.0261;

% %% Nominal controller parameters (Skandalakis paper)
% p_nom = p;
% 
% p_nom.m0 = 59.40; p_nom.I0 = 2.4021;
% p_nom.m1 = 0.22;  p_nom.I1 = 7.491e-3;
% p_nom.m2 = 0.11;  p_nom.I2 = 1.4126e-5;
% p_nom.m3 = 2.25;  p_nom.I3 = 6.8525e-2;
% 
% % Numerical vector order must match params_sym in the derivation.
% pack_params = @(p) [p.m0; p.I0; p.m1; p.I1; p.m2; p.I2; p.m3; p.I3; ...
%                     p.r0; p.l1; p.r1; p.l2; p.r2; p.l3; p.r3];
% params_true = pack_params(p);
% params_nom = pack_params(p_nom);

 %% Nominal controller parameters

% p_nom = p;
% 
% p_nom.m0 = 0.5*p.m0;  p_nom.r0 = 1.2*p.r0;
% p_nom.I0 = p.I0*(p_nom.m0/p.m0)*(p_nom.r0/p.r0)^2;
% 
% p_nom.m1 = 1.5*p.m1;  p_nom.l1 = 0.9*p.l1;  p_nom.r1 = 0.8*p.r1;
% p_nom.I1 = p.I1*(p_nom.m1/p.m1)*((p_nom.l1+p_nom.r1)/(p.l1+p.r1))^2;
% 
% p_nom.m2 = 0.5*p.m2;  p_nom.l2 = 1.1*p.l2;  p_nom.r2 = 1.2*p.r2;
% p_nom.I2 = p.I2*(p_nom.m2/p.m2)*((p_nom.l2+p_nom.r2)/(p.l2+p.r2))^2;
% 
% p_nom.m3 = 1.5*p.m3;  p_nom.l3 = 0.9*p.l3;  p_nom.r3 = 0.8*p.r3;
% p_nom.I3 = p.I3*(p_nom.m3/p.m3)*((p_nom.l3+p_nom.r3)/(p.l3+p.r3))^2;
% 
% % Numerical vector order must match params_sym in the derivation.
% pack_params = @(p) [p.m0; p.I0; p.m1; p.I1; p.m2; p.I2; ...
%                     p.m3; p.I3; p.r0; p.l1; p.r1; p.l2; ...
%                     p.r2; p.l3; p.r3];
% 
% params_true = pack_params(p);
% params_nom  = pack_params(p_nom);

%  %% Nominal controller parameters
% 
% p_nom = p;
% 
% p_nom.m0 = 0.3*p.m0;  p_nom.r0 = 1.6*p.r0;
% p_nom.I0 = p.I0*(p_nom.m0/p.m0)*(p_nom.r0/p.r0)^2;
% 
% p_nom.m1 = 1.7*p.m1;  p_nom.l1 = 0.6*p.l1;  p_nom.r1 = 0.7*p.r1;
% p_nom.I1 = p.I1*(p_nom.m1/p.m1)*((p_nom.l1+p_nom.r1)/(p.l1+p.r1))^2;
% 
% p_nom.m2 = 0.5*p.m2;  p_nom.l2 = 1.6*p.l2;  p_nom.r2 = 1.7*p.r2;
% p_nom.I2 = p.I2*(p_nom.m2/p.m2)*((p_nom.l2+p_nom.r2)/(p.l2+p.r2))^2;
% 
% p_nom.m3 = 1.8*p.m3;  p_nom.l3 = 0.7*p.l3;  p_nom.r3 = 0.6*p.r3;
% p_nom.I3 = p.I3*(p_nom.m3/p.m3)*((p_nom.l3+p_nom.r3)/(p.l3+p.r3))^2;
% 
% % Numerical vector order must match params_sym in the derivation.
% pack_params = @(p) [p.m0; p.I0; p.m1; p.I1; p.m2; p.I2; ...
%                     p.m3; p.I3; p.r0; p.l1; p.r1; p.l2; ...
%                     p.r2; p.l3; p.r3];
% 
% params_true = pack_params(p);
% params_nom  = pack_params(p_nom);


%  %% Nominal controller parameters
% 
% p_nom = p;
% 
% p_nom.m0 = (1/2)*p.m0;  p_nom.r0 = (1/2)*p.r0;
% p_nom.I0 = p.I0*(p_nom.m0/p.m0)*(p_nom.r0/p.r0)^2;
% 
% p_nom.m1 = (1/2)*p.m1;  p_nom.l1 = (1/2)*p.l1;  p_nom.r1 = (1/2)*p.r1;
% p_nom.I1 = p.I1*(p_nom.m1/p.m1)*((p_nom.l1+p_nom.r1)/(p.l1+p.r1))^2;
% 
% p_nom.m2 = (2)*p.m2;  p_nom.l2 = (1/2)*p.l2;  p_nom.r2 = (1/2)*p.r2;
% p_nom.I2 = p.I2*(p_nom.m2/p.m2)*((p_nom.l2+p_nom.r2)/(p.l2+p.r2))^2;
% 
% p_nom.m3 = (1/2)*p.m3;  p_nom.l3 = (1/2)*p.l3;  p_nom.r3 = (2)*p.r3;
% p_nom.I3 = p.I3*(p_nom.m3/p.m3)*((p_nom.l3+p_nom.r3)/(p.l3+p.r3))^2;
% 
% % Numerical vector order must match params_sym in the derivation.
% pack_params = @(p) [p.m0; p.I0; p.m1; p.I1; p.m2; p.I2; ...
%                     p.m3; p.I3; p.r0; p.l1; p.r1; p.l2; ...
%                     p.r2; p.l3; p.r3];
% 
% params_true = pack_params(p);
% params_nom  = pack_params(p_nom);

 %% Nominal controller parameters

p_nom = p;

p_nom.m0 = (1/5)*p.m0;  p_nom.r0 = (1/5)*p.r0;
p_nom.I0 = p.I0*(p_nom.m0/p.m0)*(p_nom.r0/p.r0)^2;

p_nom.m1 = (1/5)*p.m1;  p_nom.l1 = (1/5)*p.l1;  p_nom.r1 = (1/5)*p.r1;
p_nom.I1 = p.I1*(p_nom.m1/p.m1)*((p_nom.l1+p_nom.r1)/(p.l1+p.r1))^2;

p_nom.m2 = (1/5)*p.m2;  p_nom.l2 = (1/5)*p.l2;  p_nom.r2 = (1/2)*p.r2;
p_nom.I2 = p.I2*(p_nom.m2/p.m2)*((p_nom.l2+p_nom.r2)/(p.l2+p.r2))^2;

p_nom.m3 = (1/2)*p.m3;  p_nom.l3 = (1/2)*p.l3;  p_nom.r3 = (1/5)*p.r3;
p_nom.I3 = p.I3*(p_nom.m3/p.m3)*((p_nom.l3+p_nom.r3)/(p.l3+p.r3))^2;

% Numerical vector order must match params_sym in the derivation.
pack_params = @(p) [p.m0; p.I0; p.m1; p.I1; p.m2; p.I2; ...
                    p.m3; p.I3; p.r0; p.l1; p.r1; p.l2; ...
                    p.r2; p.l3; p.r3];

params_true = pack_params(p);
params_nom  = pack_params(p_nom);



%% Policy parameter size
[phi_beta, phi_alpha] = cepheus_policy_features(zeros(8,1));

% Full, nonsymmetric 4x4 alpha correction
n_params = 4*length(phi_beta) + 16*length(phi_alpha);

theta_zero = zeros(n_params,1);

fprintf('Beta features: %d\n', length(phi_beta));
fprintf('Alpha features: %d\n', length(phi_alpha));
fprintf('Number of learned parameters: %d\n', n_params);

%% Fixed 10-second tracking task
% This trajectory is only for testing, it is not used during REINFORCE training

task.T = 10;

% Vector order: [spacecraft attitude; joint 1; joint 2; joint 3].

task.A   = [0; 0.5; 0.5; 0.4];
task.w   = [0; 1.0; 1.0; 0.8];
task.ph  = [0; 0.0; pi/2; 0.3];
task.off = [0; 0.0; 0.0; 0.2];

[qd0, dqd0, ~] = cepheus_reference_from_task(0, task);

% Small initial tracking error.
task.x0 = [
    qd0 + [0.02; 0.10; -0.10; 0.10];
    dqd0];


%% Test before training
[t_true, x_true, qd_true, ~, J_true] = ...
    cepheus_rollout_simulation(params_true, params_nom, theta_zero, task, 'true');

[t_nom, x_nom, qd_nom, ~, J_nom] = ...
    cepheus_rollout_simulation(params_true, params_nom, theta_zero, task, 'nominal');

[t_l0, x_l0, qd_l0, ~, J_l0] = cepheus_rollout_simulation(params_true, params_nom, theta_zero, task, 'learned');

fprintf('\nBefore training:\n');
fprintf('True FL cost:          %.6f\n', J_true);
fprintf('Nominal FL cost:       %.6f\n', J_nom);
fprintf('Learned theta=0 cost:  %.6f\n', J_l0);

%% Choose learning algorithm

% algorithm = 'REINFORCE';
algorithm = 'PPO';

switch upper(algorithm)

    case 'REINFORCE'

        [theta_best, training_history, training_info] = ...
            cepheus_train_theta_REINFORCE(params_true, params_nom);

        fprintf('\nLinearization-learning results:\n');
        fprintf('Initial held-out loss: %.6f\n', ...
            training_info.initial_validation_loss);
        fprintf('Best held-out loss:    %.6f\n', ...
            training_info.best_validation_loss);
        fprintf('Acceleration RMSE:     [%.6f, %.6f, %.6f, %.6f] rad/s^2\n', ...
            training_info.rmse(1), training_info.rmse(2), ...
            training_info.rmse(3), training_info.rmse(4));

    case 'PPO'

    [theta_best, training_history, training_info] = ...
        cepheus_train_theta_PPO(params_true, params_nom);

    fprintf('\nPPO linearization-learning results:\n');
    fprintf('Initial held-out loss: %.6f\n', ...
        training_info.initial_validation_loss);
    fprintf('Best held-out loss:    %.6f\n', ...
        training_info.best_validation_loss);
    fprintf('Acceleration RMSE:     [%.6f, %.6f, %.6f, %.6f] rad/s^2\n', ...
        training_info.rmse(1), training_info.rmse(2), ...
            training_info.rmse(3), training_info.rmse(4));
    fprintf('Environment steps:     %d\n', ...
        training_info.total_environment_steps);

    otherwise

        error('Unknown learning algorithm: %s', algorithm);

end

%% Test after training on the fixed task
[t_learned, x_learned, qd_learned, ~, J_learned] = ...
    cepheus_rollout_simulation(params_true, params_nom, theta_best, task, 'learned');

fprintf('\nAfter training:\n');
fprintf('True FL cost:          %.6f\n', J_true);
fprintf('Nominal FL cost:       %.6f\n', J_nom);
fprintf('Learned FL cost:       %.6f\n', J_learned);

%% Plot spacecraft attitude
figure;
% figure('WindowState','maximized');

plot(t_true, qd_true(:,1), 'k--', 'LineWidth', 1.5);
hold on;

plot(t_true, x_true(:,1), 'LineWidth', 1.5);
plot(t_nom, x_nom(:,1), 'LineWidth', 1.2);
plot(t_learned, x_learned(:,1), 'LineWidth', 1.5);

grid on;

xlabel('Time [s]');
ylabel('\theta_0 [rad]');

legend( ...
    '\theta_0 desired', ...
    'Correct FL', ...
    'Wrong model FL', ...
    'Learned FL');

title('Comparison of \theta_0 Tracking');

%% Plot joint 1
figure;
% figure('WindowState','maximized');

plot(t_true, qd_true(:,2), 'k--', 'LineWidth', 1.5);
hold on;

plot(t_true, x_true(:,2), 'LineWidth', 1.5);
plot(t_nom, x_nom(:,2), 'LineWidth', 1.2);
plot(t_learned, x_learned(:,2), 'LineWidth', 1.5);

grid on;

xlabel('Time [s]');
ylabel('q_1 [rad]');

legend( ...
    'q_1 desired', ...
    'Correct FL', ...
    'Wrong model FL', ...
    'Learned FL');

title('Comparison of q_1 Tracking');

%% Plot joint 2
figure;
% figure('WindowState','maximized');

plot(t_true, qd_true(:,3), 'k--', 'LineWidth', 1.5);
hold on;

plot(t_true, x_true(:,3), 'LineWidth', 1.5);
plot(t_nom, x_nom(:,3), 'LineWidth', 1.2);
plot(t_learned, x_learned(:,3), 'LineWidth', 1.5);

grid on;

xlabel('Time [s]');
ylabel('q_2 [rad]');

legend( ...
    'q_2 desired', ...
    'Correct FL', ...
    'Wrong model FL', ...
    'Learned FL');

title('Comparison of q_2 Tracking');

%% Plot joint 3
figure;
% figure('WindowState','maximized');

plot(t_true, qd_true(:,4), 'k--', 'LineWidth', 1.5);
hold on;

plot(t_true, x_true(:,4), 'LineWidth', 1.5);
plot(t_nom, x_nom(:,4), 'LineWidth', 1.2);
plot(t_learned, x_learned(:,4), 'LineWidth', 1.5);

grid on;

xlabel('Time [s]');
ylabel('q_3 [rad]');

legend( ...
    'q_3 desired', ...
    'Correct FL', ...
    'Wrong model FL', ...
    'Learned FL');

title('Comparison of q_3 Tracking');

%% Plot errors
e_true = x_true(:,1:4) - qd_true;
e_nom = x_nom(:,1:4) - qd_nom;
e_learned = x_learned(:,1:4) - qd_learned;

figure;
% figure('WindowState','maximized');

subplot(4,1,1);

plot(t_true, e_true(:,1), 'LineWidth', 1.5);
hold on;

plot(t_nom, e_nom(:,1), 'LineWidth', 1.2);
plot(t_learned, e_learned(:,1), 'LineWidth', 1.5);

grid on;

xlabel('Time [s]');
ylabel('Error [rad]');

legend( ...
    'True FL error', ...
    'Nominal FL error', ...
    'Learned FL error');

title('\theta_0 Tracking Error Comparison');

subplot(4,1,2);

plot(t_true, e_true(:,2), 'LineWidth', 1.5);
hold on;

plot(t_nom, e_nom(:,2), 'LineWidth', 1.2);
plot(t_learned, e_learned(:,2), 'LineWidth', 1.5);

grid on;

xlabel('Time [s]');
ylabel('Error [rad]');

legend( ...
    'True FL error', ...
    'Nominal FL error', ...
    'Learned FL error');

title('q_1 Tracking Error Comparison');

subplot(4,1,3);

plot(t_true, e_true(:,3), 'LineWidth', 1.5);
hold on;

plot(t_nom, e_nom(:,3), 'LineWidth', 1.2);
plot(t_learned, e_learned(:,3), 'LineWidth', 1.5);

grid on;

xlabel('Time [s]');
ylabel('Error [rad]');

legend( ...
    'True FL error', ...
    'Nominal FL error', ...
    'Learned FL error');

title('q_2 Tracking Error Comparison');

subplot(4,1,4);

plot(t_true, e_true(:,4), 'LineWidth', 1.5);
hold on;

plot(t_nom, e_nom(:,4), 'LineWidth', 1.2);
plot(t_learned, e_learned(:,4), 'LineWidth', 1.5);

grid on;

xlabel('Time [s]');
ylabel('Error [rad]');

legend( ...
    'True FL error', ...
    'Nominal FL error', ...
    'Learned FL error');

title('q_3 Tracking Error Comparison');

%% Plot REINFORCE training
if strcmpi(algorithm, 'REINFORCE')

    figure;
    % figure('WindowState','maximized');

    subplot(2,1,1);

    semilogy( ...
        training_history.validation_epoch, ...
        training_history.validation_loss, ...
        'LineWidth', 1.5);

    grid on;

    xlabel('Epoch');
    ylabel('Mean ||ddq-v||^2');
    title('Held-out linearization loss');

    subplot(2,1,2);

    plot( ...
        1:length(training_history.mean_return), ...
        training_history.mean_return, ...
        'LineWidth', 1.0);

    grid on;

    xlabel('Epoch');
    ylabel('Mean episodic return');
    title('REINFORCE training return');

end

%% Generalization tests

test_tasks = cell(3,1);
test_names = cell(3,1);

% Original sinusoidal task
test_tasks{1} = task;
test_names{1} = 'Original sinusoid';

% Different amplitudes, frequencies, phases and offsets
test_tasks{2}.T = 10;

test_tasks{2}.A   = [0; 0.35; 0.45; 0.30];
test_tasks{2}.w   = [0; 0.65; 1.35; 0.90];
test_tasks{2}.ph  = [0; 0.40; -0.70+pi/2; 0.50];
test_tasks{2}.off = [0; 0.15; -0.10; 0.25];
test_names{2} = 'Different sinusoid';

% Constant set point
test_tasks{3}.T = 10;

test_tasks{3}.A   = zeros(4,1);
test_tasks{3}.w   = zeros(4,1);
test_tasks{3}.ph  = zeros(4,1);
test_tasks{3}.off = [0; 0.40; -0.30; 0.50];
test_names{3} = 'Set point';

fprintf('\nGeneralization tests:\n');

for test_id = 1:length(test_tasks)

    test_task = test_tasks{test_id};

    [qd_test_0, dqd_test_0, ~] = ...
        cepheus_reference_from_task(0, test_task);

    test_task.x0 = [
        qd_test_0 + [0.02; 0.10; -0.10; 0.10];
        dqd_test_0
    ];

    [t_test_true, x_test_true, qd_test, ~, J_test_true] = ...
        cepheus_rollout_simulation( ...
            params_true, params_nom, theta_zero, test_task, 'true');

    [t_test_nom, x_test_nom, ~, ~, J_test_nom] = ...
        cepheus_rollout_simulation( ...
            params_true, params_nom, theta_zero, test_task, 'nominal');

    [t_test_learned, x_test_learned, ~, ~, J_test_learned] = ...
        cepheus_rollout_simulation( ...
            params_true, params_nom, theta_best, test_task, 'learned');

    fprintf('\n%s:\n', test_names{test_id});
    fprintf('  Correct FL cost:    %.6f\n', J_test_true);
    fprintf('  Wrong model FL cost: %.6f\n', J_test_nom);
    fprintf('  Learned FL cost: %.6f\n', J_test_learned);

    if test_id > 1

        figure;
        % figure('WindowState','maximized');

        subplot(4,1,1);

        plot(t_test_true, qd_test(:,1), ...
            'k--', 'LineWidth', 1.5);

        hold on;

        plot(t_test_true, x_test_true(:,1), ...
            'LineWidth', 1.3);

        plot(t_test_nom, x_test_nom(:,1), ...
            'LineWidth', 1.1);

        plot(t_test_learned, x_test_learned(:,1), ...
            'LineWidth', 1.5);

        grid on;

        xlabel('Time [s]');
        ylabel('\theta_0 [rad]');

        legend( ...
            'Desired', ...
            'Correct FL', ...
            'Wrong model FL', ...
            'Learned FL');

        title([test_names{test_id}, ': spacecraft attitude']);

        subplot(4,1,2);

        plot(t_test_true, qd_test(:,2), ...
            'k--', 'LineWidth', 1.5);

        hold on;

        plot(t_test_true, x_test_true(:,2), ...
            'LineWidth', 1.3);

        plot(t_test_nom, x_test_nom(:,2), ...
            'LineWidth', 1.1);

        plot(t_test_learned, x_test_learned(:,2), ...
            'LineWidth', 1.5);

        grid on;

        xlabel('Time [s]');
        ylabel('q_1 [rad]');

        legend( ...
            'Desired', ...
            'Correct FL', ...
            'Wrong model FL', ...
            'Learned FL');

        title([test_names{test_id}, ': joint 1']);

        subplot(4,1,3);

        plot(t_test_true, qd_test(:,3), ...
            'k--', 'LineWidth', 1.5);

        hold on;

        plot(t_test_true, x_test_true(:,3), ...
            'LineWidth', 1.3);

        plot(t_test_nom, x_test_nom(:,3), ...
            'LineWidth', 1.1);

        plot(t_test_learned, x_test_learned(:,3), ...
            'LineWidth', 1.5);

        grid on;

        xlabel('Time [s]');
        ylabel('q_2 [rad]');

        legend( ...
            'Desired', ...
            'Correct FL', ...
            'Wrong model FL', ...
            'Learned FL');

        title([test_names{test_id}, ': joint 2']);

        subplot(4,1,4);

        plot(t_test_true, qd_test(:,4), ...
            'k--', 'LineWidth', 1.5);

        hold on;

        plot(t_test_true, x_test_true(:,4), ...
            'LineWidth', 1.3);

        plot(t_test_nom, x_test_nom(:,4), ...
            'LineWidth', 1.1);

        plot(t_test_learned, x_test_learned(:,4), ...
            'LineWidth', 1.5);

        grid on;

        xlabel('Time [s]');
        ylabel('q_3 [rad]');

        legend( ...
            'Desired', ...
            'Correct FL', ...
            'Wrong model FL', ...
            'Learned FL');

        title([test_names{test_id}, ': joint 3']);

    end
end

tests = repmat(struct( ...
    'T',10,'A',zeros(4,1),'w',zeros(4,1), ...
    'ph',zeros(4,1),'off',zeros(4,1),'x0',zeros(8,1)),4,1);

names = {
    'Shifted unequal-frequency sinusoid'
    'Faster sinusoid'
    'Nonzero set point'
    'Initial-condition perturbation'
};

tests(1).A   = [0; 0.4; 0.3; 0.25];
tests(1).w   = [0; 0.7; 1.3; 0.9];
tests(1).ph  = [0; 0.8; -0.5+pi/2; 0.6];
tests(1).off = [0.1; 0.6; -0.4; 0.3];

tests(2).A   = [0; 0.35; 0.30; 0.25];
tests(2).w   = [0; 1.4; 1.5; 1.2];
tests(2).ph  = [0; 0.3; 1.0+pi/2; 0.4];

tests(3).off = [0.1; 1.0; -0.8; 0.6];

tests(4).A   = task.A;
tests(4).w   = task.w;
tests(4).ph  = task.ph;
tests(4).off = task.off;

for j = 1:length(tests)
    [qd0,dqd0,~] = cepheus_reference_from_task(0,tests(j));
    tests(j).x0 = [qd0;dqd0];
end

tests(3).x0 = tests(3).x0 + [0;-0.1;0.1;0;0;0;0;0];
% tests(4).x0 = tests(4).x0 + [0.05;1;-2;0.2;0;-2;1.3;0];
tests(4).x0 = [0.05;1;-2;0.4;0;-2;1.3;0];


for j = 1:length(tests)

    task_test = tests(j);

    [t_true,x_true,qd_true,~,J_true] = ...
        cepheus_rollout_simulation(params_true,params_nom,theta_zero,task_test,'true');

    [t_nom,x_nom,qd_nom,~,J_nom] = ...
        cepheus_rollout_simulation(params_true,params_nom,theta_zero,task_test,'nominal');

    [t_learned,x_learned,qd_learned,~,J_learned] = ...
        cepheus_rollout_simulation(params_true,params_nom,theta_best,task_test,'learned');

    fprintf('\n%s:\n',names{j});
    fprintf('  Correct FL:    %.6f\n',J_true);
    fprintf('  Wrong model FL: %.6f\n',J_nom);
    fprintf('  Learned FL: %.6f\n',J_learned);

    figure;
    % figure('WindowState','maximized');
    tiledlayout(4,1);

    nexttile;
    plot(t_true,qd_true(:,1),'k--','LineWidth',1.5); hold on;
    plot(t_true,x_true(:,1),'LineWidth',1.3);
    plot(t_nom,x_nom(:,1),'LineWidth',1.3);
    plot(t_learned,x_learned(:,1),'LineWidth',1.5);
    grid on;
    ylabel('\theta_0 [rad]');
    % title(names{j});
    legend('Desired','Correct FL','Wrong model FL','Learned FL');

    nexttile;
    plot(t_true,qd_true(:,2),'k--','LineWidth',1.5); hold on;
    plot(t_true,x_true(:,2),'LineWidth',1.3);
    plot(t_nom,x_nom(:,2),'LineWidth',1.3);
    plot(t_learned,x_learned(:,2),'LineWidth',1.5);
    grid on;
    ylabel('q_1 [rad]');
    % title(names{j});
    legend('Desired','Correct FL','Wrong model FL','Learned FL');

    nexttile;
    plot(t_true,qd_true(:,3),'k--','LineWidth',1.5); hold on;
    plot(t_true,x_true(:,3),'LineWidth',1.3);
    plot(t_nom,x_nom(:,3),'LineWidth',1.3);
    plot(t_learned,x_learned(:,3),'LineWidth',1.5);
    grid on;
    ylabel('q_2 [rad]');
    % title(names{j});
    legend('Desired','Correct FL','Wrong model FL','Learned FL');

    nexttile;
    plot(t_true,qd_true(:,4),'k--','LineWidth',1.5); hold on;
    plot(t_true,x_true(:,4),'LineWidth',1.3);
    plot(t_nom,x_nom(:,4),'LineWidth',1.3);
    plot(t_learned,x_learned(:,4),'LineWidth',1.5);
    grid on;
    ylabel('q_3 [rad]');
    % title(names{j});
    legend('Desired','Correct FL','Wrong model FL','Learned FL');

    xlabel('Time [s]');

end
