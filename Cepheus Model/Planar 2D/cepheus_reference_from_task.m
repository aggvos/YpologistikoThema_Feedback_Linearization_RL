function [ksi_d, dksi_d, ddksi_d] = cepheus_reference_from_task(t, task)

    A   = task.A(:);
    w   = task.w(:);
    ph  = task.ph(:);
    off = task.off(:);

    phase = w*t + ph;

    ksi_d   = off + A.*sin(phase);
    dksi_d  = A.*w.*cos(phase);
    ddksi_d = -A.*(w.^2).*sin(phase);

end