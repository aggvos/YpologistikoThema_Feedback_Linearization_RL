function dx = cepheus_constrained_state_space(~,x,u,params,plant)
% The simulator enforces torque limits and joint viscous damping

    u = max(-plant.torque_limits,min(plant.torque_limits,u(:)));
    H = cepheus_H(x(1:4),params);
    c = cepheus_c(x(1:4),x(5:8),params);
    dx = [x(5:8);H\(u-c-[0;plant.joint_damping(:).*x(6:8)])];
end
