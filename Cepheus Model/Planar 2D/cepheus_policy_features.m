function [phi_beta, phi_alpha] = cepheus_policy_features(x)

    x = x(:);
    assert(numel(x) == 8, 'Expected an eight-element state.');

    q1 = x(2);
    q2 = x(3);
    q3 = x(4);
    dksi = x(5:8);

    % Angle combinations appearing in H and c
    angles = [
        q1;
        q2;
        q3;
        q1 + q2;
        q2 + q3;
        q1 + q2 + q3];

    % Inertia-matrix features: 7 x 1
    phi_alpha = [1; cos(angles)];

    % Ten distinct quadratic velocity products
    velocity_products = zeros(10,1);

    k = 0;
    for i = 1:4
        for j = i:4
            k = k + 1;
            velocity_products(k) = dksi(i)*dksi(j);
        end
    end

    % Coriolis/centrifugal features: 60 x 1
    % phi_beta = kron(sin(angles), velocity_products); % για όταν δεν βάζω damping
    % Coriolis/centrifugal features: 64 x 1
    phi_beta = [kron(sin(angles), velocity_products); dksi];

end