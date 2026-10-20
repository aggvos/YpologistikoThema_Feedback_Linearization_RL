clear; clc;
% clear all;

% Generalized coordinates and velocities
syms theta0 q1 q2 q3 real
syms dtheta0 dq1 dq2 dq3 real

ksi  = [theta0; q1; q2; q3];
dksi = [dtheta0; dq1; dq2; dq3];

% Geometric and intertial parameters 
syms r0 l1 r1 l2 r2 l3 r3 real
syms m0 I0 m1 I1 m2 I2 m3 I3 real


% Absolute angles of links
theta1 = theta0 + q1;
theta2 = theta0 + q1 + q2;
theta3 = theta0 + q1 + q2 + q3;

% Positions of center of mass of each link relative to spacecraft center of mass
p1 = r0*[cos(theta0); sin(theta0)] + l1*[cos(theta1); sin(theta1)];
p2 = r0*[cos(theta0); sin(theta0)] + (l1+r1)*[cos(theta1); sin(theta1)] + ...
     l2*[cos(theta2); sin(theta2)];
p3 = r0*[cos(theta0); sin(theta0)] + (l1+r1)*[cos(theta1); sin(theta1)] + ...
    (l2+r2)*[cos(theta2); sin(theta2)] + l3*[cos(theta3); sin(theta3)];

% Angular velocities
omega0 = dtheta0;
omega1 = dtheta0 + dq1;
omega2 = dtheta0 + dq1 + dq2;
omega3 = dtheta0 + dq1 + dq2 + dq3;

% Linear velocities relative to center of mass of spacecraft (body)
v1_rel = simplify(jacobian(p1, ksi)*dksi);
v2_rel = simplify(jacobian(p2, ksi)*dksi);
v3_rel = simplify(jacobian(p3, ksi)*dksi);


%Conservation of linear momentum
Mtot = m0 + m1 + m2 + m3;
v0 = simplify(-(m1*v1_rel + m2*v2_rel + m3*v3_rel)/Mtot);

% Linear velocities
v1 = simplify(v0 + v1_rel);
v2 = simplify(v0 + v2_rel);
v3 = simplify(v0 + v3_rel);

% Kinetic energy
T0 = (1/2)*m0*(v0.'*v0) + (1/2)*I0*omega0^2;
T1 = (1/2)*m1*(v1.'*v1) + (1/2)*I1*omega1^2;
T2 = (1/2)*m2*(v2.'*v2) + (1/2)*I2*omega2^2;
T3 = (1/2)*m3*(v3.'*v3) + (1/2)*I3*omega3^2;

T = simplify(T0 + T1 + T2 + T3);

% H = simplify(hessian(T, dksi));
% H_symmetry_check = simplify(H - H.');


%% Euler - Lagrange

syms ddtheta0 ddq1 ddq2 ddq3 real

ddksi = [ddtheta0; ddq1; ddq2; ddq3];

% Lagrangian
V = 0;
L = T - V;  

% Partial derivative dL/ddksi
dL_ddksi = jacobian(L, dksi).';

% Time derivative d/dt(dL/ddksi)
dt_dL_ddksi = jacobian(dL_ddksi, ksi)*dksi + jacobian(dL_ddksi, dksi)*ddksi;

% Partial derivative dL/dksi
dL_dksi = jacobian(L, ksi).';

% Euler-Lagrange left-hand side
EL = simplify(dt_dL_ddksi - dL_dksi);

% Inertial matrix from Euler-Lagrange
H = simplify(jacobian(EL,ddksi));

% Centrifugal and Coriolis Matrix 
c = simplify(EL - H*ddksi);
% c = simplify(subs(EL, ddksi, zeros(4,1)));

% model: H(q)ξ'' + c(q,q') = τ 


% Export symbolic H and c as numerical MATLAB functions for simulation
params_sym = [m0; I0; m1; I1; m2; I2; m3; I3; r0; l1; r1; l2; r2; l3; r3];
matlabFunction(H, 'File', 'cepheus_H', 'Vars', {ksi, params_sym});
matlabFunction(c, 'File', 'cepheus_c', 'Vars', {ksi, dksi, params_sym});