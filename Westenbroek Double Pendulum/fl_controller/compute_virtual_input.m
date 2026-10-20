function [v, e, de] = compute_virtual_input(x, qd, dqd, ddqd, Kp, Kd)
% Computes the virtual input v for trajectory tracking
% v = ddqd - Kd*(dq - dqd) - Kp*(q - qd)

    q  = x(1:2);
    dq = x(3:4);

    e  = q  - qd;
    de = dq - dqd;

    v = ddqd - Kd*de - Kp*e;

end