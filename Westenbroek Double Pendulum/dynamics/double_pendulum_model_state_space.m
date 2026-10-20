function dx = double_pendulum_model(t, x, u, p)
% State: x = [th1; th2; dth1; dth2]
% Input: u = [u1; u2]

    q  = x(1:2);
    dq = x(3:4);

    [M, C_dq, G] = double_pendulum_matrices(x, p);
    ddq = M \ (u - C_dq - G);
    dx = [dq; ddq];

end