function [v, e, de] = cepheus_virtual_input( x, ksi_d, dksi_d, ddksi_d, Kp, Kd)
% v = ddksi_d - Kd*(dksi - dksi_d) - Kp*(ksi - ksi_d)

    ksi  = x(1:4);
    dksi = x(5:8);

    e  = ksi  - ksi_d;
    de = dksi - dksi_d;

    v = ddksi_d - Kd*de - Kp*e;

end