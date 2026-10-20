clear; clc; close all;

addpath(genpath(pwd));
rng(1);

%% True plant parameters
p.m1 = 1;
p.m2 = 1;

p.l1 = 1;
p.l2 = 1;

p.lc1 = p.l1/2;
p.lc2 = p.l2/2;

p.I1 = (1/12)*p.m1*p.l1^2;
p.I2 = (1/12)*p.m2*p.l2^2;

p.g = 9.81;

%% Nominal/wrong controller parameters
p_nom = p;

p_nom.m1 = p.m1/3;
p_nom.m2 = p.m2/3;

p_nom.l1 = p.l1/3;
p_nom.l2 = p.l2/3;

p_nom.lc1 = p_nom.l1/2;
p_nom.lc2 = p_nom.l2/2;

p_nom.I1 = (1/12)*p_nom.m1*p_nom.l1^2;
p_nom.I2 = (1/12)*p_nom.m2*p_nom.l2^2;

%% Policy parameter size
[phi_beta, phi_alpha] = policy_features(zeros(4,1));

% Full, nonsymmetric 2x2 alpha correction
n_params = 2*length(phi_beta) + 4*length(phi_alpha);

theta_zero = zeros(n_params,1);

fprintf('Beta features: %d\n', length(phi_beta));
fprintf('Alpha features: %d\n', length(phi_alpha));
fprintf('Number of learned parameters: %d\n', n_params);

%% Fixed 10-second tracking task
% This trajectory is only for testing

task.T = 10;

task.A1 = 0.5;
task.A2 = 0.5;

task.w1 = 1.0;
task.w2 = 1.0;

task.ph1 = 0.0;
task.ph2 = 0.0;

task.off1 = 0.0;
task.off2 = 0.0;

[qd0, dqd0, ~] = reference_from_task(0, task);

% Small initial tracking error

task.x0 = [qd0 + [0.10; -0.10];dqd0];

%% Test before training
[t_true, x_true, qd_true, ~, J_true] = ...
    rollout_simulation(p, p_nom, theta_zero, task, 'true');

[t_nom, x_nom, qd_nom, ~, J_nom] = ...
    rollout_simulation(p, p_nom, theta_zero, task, 'nominal');

[t_l0, x_l0, qd_l0, ~, J_l0] = rollout_simulation(p, p_nom, theta_zero, task, 'learned');

fprintf('\nBefore training:\n');
fprintf('True FL cost:          %.6f\n', J_true);
fprintf('Nominal FL cost:       %.6f\n', J_nom);
fprintf('Learned theta=0 cost:  %.6f\n', J_l0);

%% Choose learning algorithm

algorithm = 'REINFORCE';
% algorithm = 'PPO';so 

switch upper(algorithm)

    case 'REINFORCE'

        [theta_best, training_history, training_info] = ...
            train_theta_REINFORCE(p, p_nom);

        fprintf('\nLinearization-learning results:\n');
        fprintf('Initial held-out loss: %.6f\n', ...
            training_info.initial_validation_loss);
        fprintf('Best held-out loss:    %.6f\n', ...
            training_info.best_validation_loss);
        fprintf('Acceleration RMSE:     [%.6f, %.6f] rad/s^2\n', ...
            training_info.rmse(1), training_info.rmse(2));

    case 'ES'

        theta_best = train_theta_ES(p, p_nom);
        training_history = [];

    case 'PPO'

    [theta_best, training_history, training_info] = ...
        train_theta_PPO(p, p_nom);

    fprintf('\nPPO linearization-learning results:\n');
    fprintf('Initial held-out loss: %.6f\n', ...
        training_info.initial_validation_loss);
    fprintf('Best held-out loss:    %.6f\n', ...
        training_info.best_validation_loss);
    fprintf('Acceleration RMSE:     [%.6f, %.6f] rad/s^2\n', ...
        training_info.rmse(1), training_info.rmse(2));
    fprintf('Environment steps:     %d\n', ...
        training_info.total_environment_steps);

    otherwise

        error('Unknown learning algorithm: %s', algorithm);

end

%% Test after training on the fixed task
[t_learned, x_learned, qd_learned, ~, J_learned] = ...
    rollout_simulation(p, p_nom, theta_best, task, 'learned');

fprintf('\nAfter training:\n');
fprintf('True FL cost:          %.6f\n', J_true);
fprintf('Nominal FL cost:       %.6f\n', J_nom);
fprintf('Learned FL cost:       %.6f\n', J_learned);

%% Plot joint 1
figure;

plot(t_true, qd_true(:,1), 'k--', 'LineWidth', 1.5);
hold on;

plot(t_true, x_true(:,1), 'LineWidth', 1.5);
plot(t_nom, x_nom(:,1), 'LineWidth', 1.2);
plot(t_learned, x_learned(:,1), 'LineWidth', 1.5);

grid on;

xlabel('Time [s]');
ylabel('\theta_1 [rad]');

legend( ...
    '\theta_{1,desired}', ...
    'Correct FL', ...
    'Wrong model FL', ...
    'Learned FL');

title('Comparison of \theta_1 Tracking');

%% Plot joint 2
figure;

plot(t_true, qd_true(:,2), 'k--', 'LineWidth', 1.5);
hold on;

plot(t_true, x_true(:,2), 'LineWidth', 1.5);
plot(t_nom, x_nom(:,2), 'LineWidth', 1.2);
plot(t_learned, x_learned(:,2), 'LineWidth', 1.5);

grid on;

xlabel('Time [s]');
ylabel('\theta_2 [rad]');

legend( ...
    '\theta_{2,desired}', ...
    'Correct FL', ...
    'Wrong model FL', ...
    'Learned FL');

title('Comparison of \theta_2 Tracking');

%% Plot errors
e_true = x_true(:,1:2) - qd_true;
e_nom = x_nom(:,1:2) - qd_nom;
e_learned = x_learned(:,1:2) - qd_learned;

figure;

subplot(2,1,1);

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

title('\theta_1 Tracking Error Comparison');

subplot(2,1,2);

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

title('\theta_2 Tracking Error Comparison');

%% Plot REINFORCE training
if strcmpi(algorithm, 'REINFORCE')

    figure;

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
% None of these trajectories were used during training.

test_tasks = cell(3,1);
test_names = cell(3,1);

% Original sinusoidal task
test_tasks{1} = task;
test_names{1} = 'Original sinusoid';

% Different amplitudes, frequencies, phases and offsets
test_tasks{2}.T = 10;

test_tasks{2}.A1 = 0.35;
test_tasks{2}.A2 = 0.45;

test_tasks{2}.w1 = 0.65;
test_tasks{2}.w2 = 1.35;

test_tasks{2}.ph1 = 0.40;
test_tasks{2}.ph2 = -0.70;

test_tasks{2}.off1 = 0.15;
test_tasks{2}.off2 = -0.10;
test_names{2} = 'Different sinusoid';

% Constant set point
test_tasks{3}.T = 10;

test_tasks{3}.A1 = 0.0;
test_tasks{3}.A2 = 0.0;

test_tasks{3}.w1 = 1.0;
test_tasks{3}.w2 = 1.0;

test_tasks{3}.ph1 = 0.0;
test_tasks{3}.ph2 = 0.0;

test_tasks{3}.off1 = 0.40;
test_tasks{3}.off2 = -0.30;

test_names{3} = 'Set point';

fprintf('\nGeneralization tests:\n');

for test_id = 1:length(test_tasks)

    test_task = test_tasks{test_id};

    [qd_test_0, dqd_test_0, ~] = ...
        reference_from_task(0, test_task);

    test_task.x0 = [
        qd_test_0 + [0.10; -0.10];
        dqd_test_0
    ];

    [t_test_true, x_test_true, qd_test, ~, J_test_true] = ...
        rollout_simulation( ...
            p, p_nom, theta_zero, test_task, 'true');

    [t_test_nom, x_test_nom, ~, ~, J_test_nom] = ...
        rollout_simulation( ...
            p, p_nom, theta_zero, test_task, 'nominal');

    [t_test_learned, x_test_learned, ~, ~, J_test_learned] = ...
        rollout_simulation( ...
            p, p_nom, theta_best, test_task, 'learned');

    fprintf('\n%s:\n', test_names{test_id});
    fprintf('  Correct FL cost:    %.6f\n', J_test_true);
    fprintf('  Wrong model FL cost: %.6f\n', J_test_nom);
    fprintf('  Learned FL cost: %.6f\n', J_test_learned);

    if test_id > 1

        figure;

        subplot(2,1,1);

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
        ylabel('q_1 [rad]');

        legend( ...
            'Desired', ...
            'Correct FL', ...
            'Wrong model FL', ...
            'Learned FL');

        title([test_names{test_id}, ': joint 1']);

        subplot(2,1,2);

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
        ylabel('q_2 [rad]');

        legend( ...
            'Desired', ...
            'Correct FL', ...
            'Wrong model FL', ...
            'Learned FL');

        title([test_names{test_id}, ': joint 2']);

    end
end

tests = repmat(struct( ...
    'T',10,'A1',0,'A2',0,'w1',0,'w2',0, ...
    'ph1',0,'ph2',0,'off1',0,'off2',0,'x0',zeros(4,1)),4,1);

names = {
    'Shifted unequal-frequency sinusoid'
    'Faster sinusoid'
    'Nonzero set point'
    'Initial-condition perturbation'
};

tests(1).A1 = 0.4;
tests(1).A2 = 0.3;
tests(1).w1 = 0.7;
tests(1).w2 = 1.3;
tests(1).ph1 = 0.8;
tests(1).ph2 = -0.5;
tests(1).off1 = 0.6;
tests(1).off2 = -0.4;

tests(2).A1 = 0.35;
tests(2).A2 = 0.30;
tests(2).w1 = 1.4;
tests(2).w2 = 1.5;
tests(2).ph1 = 0.3;
tests(2).ph2 = 1.0;

tests(3).off1 = 1.0;
tests(3).off2 = -0.8;

tests(4).A1 = 0.5;
tests(4).A2 = 0.5;
tests(4).w1 = 1.0;
tests(4).w2 = 1.0;
tests(4).ph1 = 0.0;
tests(4).ph2 = 0.0;
tests(4).off1 = 0.0;
tests(4).off2 = 0.0;

for j = 1:length(tests)
    [qd0,dqd0,~] = reference_from_task(0,tests(j));
    tests(j).x0 = [qd0;dqd0];
end

tests(3).x0 = tests(3).x0 + [-0.1;0.1;0;0];
% tests(4).x0 = tests(4).x0 + [1;-2;-2;1.3];
tests(4).x0 = [1;-2;-2;1.3];


for j = 1:length(tests)

    task_test = tests(j);

    [t_true,x_true,qd_true,~,J_true] = ...
        rollout_simulation(p,p_nom,theta_zero,task_test,'true');

    [t_nom,x_nom,qd_nom,~,J_nom] = ...
        rollout_simulation(p,p_nom,theta_zero,task_test,'nominal');

    [t_learned,x_learned,qd_learned,~,J_learned] = ...
        rollout_simulation(p,p_nom,theta_best,task_test,'learned');

    fprintf('\n%s:\n',names{j});
    fprintf('  Correct FL:    %.6f\n',J_true);
    fprintf('  Wrong model FL: %.6f\n',J_nom);
    fprintf('  Learned FL: %.6f\n',J_learned);

    figure;
    tiledlayout(2,1);

    nexttile;
    plot(t_true,qd_true(:,1),'k--','LineWidth',1.5); hold on;
    plot(t_true,x_true(:,1),'LineWidth',1.3);
    plot(t_nom,x_nom(:,1),'LineWidth',1.3);
    plot(t_learned,x_learned(:,1),'LineWidth',1.5);
    grid on;
    ylabel('\theta_1 [rad]');
    % title(names{j});
    legend('Desired','Correct FL','Wrong model FL','Learned FL');

    nexttile;
    plot(t_true,qd_true(:,2),'k--','LineWidth',1.5); hold on;
    plot(t_true,x_true(:,2),'LineWidth',1.3);
    plot(t_nom,x_nom(:,2),'LineWidth',1.3);
    plot(t_learned,x_learned(:,2),'LineWidth',1.5);
    grid on;
    xlabel('Time [s]');
    ylabel('\theta_2 [rad]');
end
