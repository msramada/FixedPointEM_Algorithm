%% Measurement update of the particle filter
function Likelihood= Measurement_Likelihood(gx,y,R_y)
% Calculates unscaled likelihood
p=size(R_y);
helper1=-1/2*(y-gx)'*R_y^(-1)*(y-gx); % (N x N)
Likelihood=exp(diag(helper1)); % Off diagonal are cross-terms
end