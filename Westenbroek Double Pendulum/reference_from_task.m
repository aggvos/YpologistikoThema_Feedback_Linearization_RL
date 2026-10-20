function [qd, dqd, ddqd] = reference_from_task(t, task)

    A1 = task.A1;
    A2 = task.A2;

    w1 = task.w1;
    w2 = task.w2;

    ph1 = task.ph1;
    ph2 = task.ph2;

    off1 = task.off1;
    off2 = task.off2;

    qd = [
        off1 + A1*sin(w1*t + ph1);
        off2 + A2*cos(w2*t + ph2)
    ];

    dqd = [
        A1*w1*cos(w1*t + ph1);
       -A2*w2*sin(w2*t + ph2)
    ];

    ddqd = [
       -A1*w1^2*sin(w1*t + ph1);
       -A2*w2^2*cos(w2*t + ph2)
    ];

end