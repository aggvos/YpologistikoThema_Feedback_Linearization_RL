function [phi_beta, phi_alpha] = policy_features(x)

    th1  = x(1);
    th2  = x(2);
    dth1 = x(3);
    dth2 = x(4);

    z = [
        sin(th1);
        cos(th1);
        sin(th2);
        cos(th2);
        dth1;
        dth2
    ];

    phi_beta = 1;
    phi_beta = [phi_beta; z];

    n = length(z);

    for i = 1:n
        for j = i:n
            phi_beta = [phi_beta; z(i)*z(j)];
        end
    end

    phi_beta = [
        phi_beta;
        sin(th2)*dth1^2;
        sin(th2)*dth1*dth2;
        sin(th2)*dth2^2
    ];

    phi_alpha = [
        1;
        cos(th2)
    ];

end
