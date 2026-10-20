function [M, C_dq, G] = double_pendulum_matrices(x, p)

    th1  = x(1);
    th2  = x(2);
    dth1 = x(3);
    dth2 = x(4);

    m1  = p.m1;
    m2  = p.m2;
    l1  = p.l1;
    lc1 = p.lc1;
    lc2 = p.lc2;
    I1  = p.I1;
    I2  = p.I2;
    g   = p.g;

    % Inertia matrix M(q)
    M11 = m1*lc1^2 + I1 + m2*(l1^2 + lc2^2 + 2*l1*lc2*cos(th2)) + I2;
    M12 = m2*(lc2^2 + l1*lc2*cos(th2)) + I2;
    M21 = M12;
    M22 = m2*lc2^2 + I2;

    M = [M11 M12;
         M21 M22];

    % Coriolis / centrifugal vector C(q,dq)*dq
    h = m2*l1*lc2*sin(th2);

    C_dq = [-h*(2*dth1*dth2 + dth2^2);
         h*dth1^2];

    % Gravity vector G(q)
    G = [(m1*lc1 + m2*l1)*g*cos(th1) + m2*lc2*g*cos(th1 + th2);
         m2*lc2*g*cos(th1 + th2)];
    
end