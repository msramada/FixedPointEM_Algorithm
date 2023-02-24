%% State-Space model as defined in the paper.
function [x,gx] = SSM(x0,Q_x,H,alpha,i)
    omega=sqrtm(Q_x)*randn(size(x0));
    x=sin(2*pi/20*i)*tanh(alpha*x0)+omega;
    gx=H*x0;
end
