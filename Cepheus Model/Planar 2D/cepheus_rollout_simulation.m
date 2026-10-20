function [t, x, qd_all, u_all, cost] = cepheus_rollout_simulation( ...
    params, params_nom, theta_vec, task, controller_type)


    assert(isscalar(task.T) && isfinite(task.T) && task.T > 0, ...
        'task.T must be positive and finite.');
    params = params(:);
    params_nom = params_nom(:);
    assert(numel(params) == 15 && numel(params_nom) == 15, ...
        'Expected 15-element numerical parameter vectors.');
    controller_type = validatestring(controller_type, ...
        {'true','nominal','learned'});

    Kp = 25*eye(4);
    Kd = 10*eye(4);
    Q = 5*eye(4);
    R = 0.01*eye(4);
    if isfield(task,'Kp'), Kp = task.Kp; end
    if isfield(task,'Kd'), Kd = task.Kd; end
    if isfield(task,'Q'), Q = task.Q; end
    if isfield(task,'R'), R = task.R; end
    assert(isequal(size(Kp),[4 4]) && isequal(size(Kd),[4 4]) && ...
           isequal(size(Q),[4 4]) && isequal(size(R),[4 4]), ...
           'Kp, Kd, Q and R must be 4-by-4 matrices.');

    [qd0, dqd0, ~] = cepheus_reference_from_task(0, task);
    if isfield(task,'x0')
        x0 = task.x0(:);
    else
        x0 = [qd0; dqd0];
    end
    assert(numel(x0) == 8, 'task.x0 must have eight entries.');

    odefun = @(t,x) closed_loop_rhs(t,x,params,params_nom, ...
        theta_vec,task,controller_type,Kp,Kd);
    options = odeset('RelTol',1e-6,'AbsTol',1e-8);
    if isfield(task,'ode_options')
        options = odeset(options,task.ode_options);
    end
    [t,x] = ode45(odefun,[0 task.T],x0,options);

    % An incomplete/nonfinite rollout is penalized. NaN marks invalid logs.
    if t(end) < task.T - 1e-3 || any(~isfinite(x(:)))
        cost = 1e6;
        qd_all = nan(length(t),4);
        u_all = nan(length(t),4);
        return;
    end

    qd_all = zeros(length(t),4);
    u_all = zeros(length(t),4);
    running_cost = zeros(length(t),1);
    for k = 1:length(t)
        [qd,dqd,ddqd] = cepheus_reference_from_task(t(k),task);
        [v,e,de] = cepheus_virtual_input(x(k,:)',qd,dqd,ddqd,Kp,Kd);
        u = controller_torque(x(k,:)',v,params,params_nom, ...
            theta_vec,controller_type);
        qd_all(k,:) = qd';
        u_all(k,:) = u';
        running_cost(k) = e'*Q*e + de'*R*de + 1e-5*(u'*u);
    end
    % Integrate over ode45's nonuniform output times.
    cost = trapz(t,running_cost);
end

function dx = closed_loop_rhs(t,x,params,params_nom,theta_vec,task, ...
    controller_type,Kp,Kd)
    [qd,dqd,ddqd] = cepheus_reference_from_task(t,task);
    [v,~,~] = cepheus_virtual_input(x,qd,dqd,ddqd,Kp,Kd);
    u = controller_torque(x,v,params,params_nom,theta_vec,controller_type);
    dx = cepheus_model_state_space(t,x,u,params);
end

function u = controller_torque(x,v,params,params_nom,theta_vec,controller_type)
    switch controller_type
        case 'true'
            u = cepheus_fl_controller(x,v,params);
        case 'nominal'
            u = cepheus_fl_controller(x,v,params_nom);
        case 'learned'
            u = cepheus_learned_fl_controller(x,v,params_nom,theta_vec);
        otherwise
            error('Unknown controller_type.');
    end
end
