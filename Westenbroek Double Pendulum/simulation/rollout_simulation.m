function [t, x, qd_all, u_all, cost] = rollout_simulation(p, p_nom, theta_vec, task, controller_type)

    tspan = [0 task.T];

    Kp = diag([25 25]);
    Kd = diag([10 10]);

    % Start on or near desired trajectory
    [qd0, dqd0, ~] = reference_from_task(0, task);

    if isfield(task, 'x0')
        x0 = task.x0;
    else
        x0 = [qd0; dqd0];
    end

    % Closed-loop dynamics
    odefun = @(t, x) closed_loop_rhs(t, x, p, p_nom, theta_vec, task, controller_type, Kp, Kd);

    [t, x] = ode45(odefun, tspan, x0);

    if t(end) < task.T - 1e-3
        cost = 1e6;
        qd_all = zeros(length(t), 2);
        u_all  = zeros(length(t), 2);
        return;
    end

    % Store desired trajectory and input along solution
    qd_all = zeros(length(t), 2);
    u_all  = zeros(length(t), 2);

    cost = 0;

    for k = 1:length(t)

        [qd, dqd, ddqd] = reference_from_task(t(k), task);

        [v, e, de] = compute_virtual_input(x(k,:)', qd, dqd, ddqd, Kp, Kd);

        switch controller_type

            case 'true'
                [u, ~, ~] = fl_controller(x(k,:)', v, p);

            case 'nominal'
                [u, ~, ~] = fl_controller(x(k,:)', v, p_nom);

            case 'learned'
                [u, ~, ~, ~, ~, ~, ~] = learned_fl_controller(x(k,:)', v, p_nom, theta_vec);

            otherwise
                error('Unknown controller_type');
        end

        qd_all(k,:) = qd';
        u_all(k,:)  = u';

        % running_cost = e'*e + 0.01*(de'*de) + 1e-5*(u'*u);

        Q = diag([5 5]);
        R = 0.01*eye(2);

        running_cost = e'*Q*e + de'*R*de + 1e-5*(u'*u);

        if k > 1
            dt = t(k) - t(k-1);
            cost = cost + running_cost*dt;
        end
    end

end


function dx = closed_loop_rhs(t, x, p, p_nom, theta_vec, task, controller_type, Kp, Kd)

    [qd, dqd, ddqd] = reference_from_task(t, task);

    [v, ~, ~] = compute_virtual_input(x, qd, dqd, ddqd, Kp, Kd);

    switch controller_type

        case 'true'
            [u, ~, ~] = fl_controller(x, v, p);

        case 'nominal'
            [u, ~, ~] = fl_controller(x, v, p_nom);

        case 'learned'
            [u, ~, ~, ~, ~, ~, ~] = learned_fl_controller(x, v, p_nom, theta_vec);

        otherwise
            error('Unknown controller_type');
    end

    dx = double_pendulum_model_state_space(t, x, u, p);

end