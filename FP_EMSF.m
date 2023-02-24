clear all; close all; clc
%% Created by Mohammad Ramadan, msramada@eng.ucsd.edu.

%% For more context about this example, please check our paper:
% Maximum Likelihood recursive state estimation using the Expectation Maximization algorithm
% https://doi.org/10.1016/j.automatica.2022.110482

%% number of particles N, Terminal time T, state dim n, output dim p
T=40;
N=500;
n=1;
p=1;
%% nonlinear system parameters
H=1/2;
omega=20;
alpha=pi;

%% Initializing recording matrices
w=NaN(1,N);     % Importance weights for each sample and time step
x0=NaN(n,N);     % Samples from importance distribution at each time step
y=NaN(p,T);       % Output at each time step
Py_given_x=NaN(1,N); % Measurement Likelihood

%% Initializing initial distributions/particles
w(:,1)=1/N; % initial importance weights at t=1
mu_x0=zeros(n,1); % initial distribution of the state
Px0=1*eye(n);
Q=1/5*eye(n);
R=1*eye(p);
x0(:,:)=mvnrnd(mu_x0,Px0,N)'; %samples from initial dist.

x_traj=NaN(n,T); %dim, number of particles.
x_traj(:,1)=zeros;
x_rec_EMSF=NaN(1,T);
x_rec_Nikolay=NaN(1,T);

%% Generate an output sequence realization.
for jj=1:T
[x_traj(:,jj+1),gx] = SSM(x_traj(:,jj),Q,H,alpha,jj);
y(:,jj)=mvnrnd(gx,R,1)';
end
clear gx
%% Begin Particle Filter with EMSF algorithm

xtGivent=NaN(n,N);
xt=x0; % samples of the target distribution
X=[-2:0.05:2];
HIST=NaN(1,length(X),T);
tic
for i=1:T
% xt without resampling p(x(t)|Y(t-1))
% x after resampling p(x(t)|Y(t))
[~,gx]=SSM(xt(:,:),Q,H,alpha,i); % Towards Measurement Update
Py_given_x= Measurement_Likelihood(gx,y(:,i),R)'; % Measurement Likelihood
w(1,:)=Py_given_x/sum(Py_given_x); %Normalize importance weights
xtGivent(:,:) = MassDistribution(xt(:,:),w(1,:));
[xt(:,:),~]=SSM(xtGivent(:,:),Q,H,alpha,i); % Towards Time Update
x_pf(:,i)=sum(xtGivent(:,:),2)/N;
for index_X=1:length(X)
    if i<T
HIST(1,index_X,i+1)=log( normpdf(y(:,i+1)-H*X(index_X),0,sqrt(R)).*sum(normpdf(X(index_X)-((1+0.5*sin(2*pi/omega*i))*tanh(alpha*xtGivent(:,:))),0,sqrt(Q))));
    end
end
HIST(1,:,i)=HIST(1,:,i)+1-max(HIST(1,:,i));

%% EMSF Algorithm Starts:
x_EMSF=NaN(5,1); %5 random initializations
x_EMSF_likelihood1=-Inf;
if i<T

for trial=1:length(x_EMSF)
x_EMSF(trial)=6*(rand-0.5);

for jj=1:10
lamdas=Measurement_Likelihood(0,x_EMSF(trial)*ones(1,N)-...
    (1+0.5*sin(2*pi/omega*i))*tanh(alpha*xtGivent(:,:)),Q)';
lamdas=lamdas/sum(lamdas);

helper1=sum((2*H/R*y(:,i+1)+2/Q*(1+0.5*sin(2*pi/omega*i))*tanh(alpha*xtGivent(:,:))).*lamdas);
x_EMSF(trial)=(2*H^2/R+2/Q)^-1*helper1;
end
% Evaluating each guess
% by picking the one with largest log-likelihood convergence point
x_EMSF_likelihood2=log( normpdf(y(:,i+1)-H*x_EMSF(trial),0,sqrt(R))*sum( normpdf(x_EMSF(trial)-...
    (1+0.5*sin(2*pi/omega*i))*tanh(alpha*xtGivent(:,:)),0,sqrt(Q))));
if x_EMSF_likelihood2>x_EMSF_likelihood1
x_rec_EMSF(:,i+1)=x_EMSF(trial);
x_EMSF_likelihood1=x_EMSF_likelihood2;
end
end
end
end
toc
%% Plotting section

Hist=reshape(HIST,[length(X),T,1]);
[XX,YY]=meshgrid([1:T],X);
figure(1)
contourf(XX,YY,exp(Hist),'LineStyle','none','LevelStep',0.2)
hold all
xlim([2 T])
ylim([-2.0 2.0])
mycolormap = customcolormap([0 .25 .5 .75 1], {'#9d0142','#f66e45','#ffffbb','#65c0ae','#5e4f9f'});
colormap(mycolormap);
pp=[1.00,0.41,0.16];
plot(x_pf(1,:),'-*','LineWidth',1,'MarkerSize',3)

% plot(x_traj(1,1:end-1),'LineWidth',1.5)
% plot(y,'LineWidth',1.5)
h=colorbar('northoutside');
set(gca, 'CLim', [min(min(exp(Hist))), max(max(exp(Hist)))]);
set(h, 'XTick', [min(min(exp(Hist))), max(max(exp(Hist)))]);
set(h,'XTickLabel',{'Low' ,'High'});
% plot(x_rec_EMSF(1,:),'LineWidth',1)

plot(x_rec_EMSF(:,1:end)','-*k','LineWidth',1,'MarkerSize',3)

ylabel('x_{k}','fontweight','bold')
xlabel('time-step (k)','fontweight','bold')
box on
ax = gca
ax.LineWidth = 1.1
SavePDF('Example2',3.0,3.0,9,'Times Roman')

figure(2)
pp=26;
zpf=interp1q(X,exp(Hist(:,pp)),x_pf(pp));
zEMSF=interp1q(X,exp(Hist(:,pp)),x_rec_EMSF(pp));
plot(X,exp(Hist(:,pp)),'LineWidth',2)
hold all
plot([x_pf(pp) x_pf(pp)],[0  zpf],'-.k')
plot([x_rec_EMSF(pp) x_rec_EMSF(pp)],[0  zEMSF],'-.k')
xlabel('x_{k}','fontweight','bold')
Ax = gca;
if x_rec_EMSF(pp)<x_pf(pp)
set(gca, 'XTick', [-2.0,x_rec_EMSF(pp),x_pf(pp),2.0])
xticklabels({'-2.0','x_{EMSF}','x_{CM}','2.0'})
else
set(gca, 'XTick', [-2.0,x_pf(pp),x_rec_EMSF(pp),2.0])
xticklabels({'-2.0','x_{CM}','x_{EMSF}','2.0'})
end
ylabel('Empirical Filtered Density','fontweight','bold')
box on
xlim([-2.0 2.0])
ax = gca
ax.LineWidth = 1.1
SavePDF('Example230',3.0,2.0,9,'Times Roman')

figure(3)
pp=10;
zpf=interp1q(X,exp(Hist(:,pp)),x_pf(pp));
zEMSF=interp1q(X,exp(Hist(:,pp)),x_rec_EMSF(pp));
plot(X,exp(Hist(:,pp)),'LineWidth',2)
hold all
plot([x_pf(pp) x_pf(pp)],[0  zpf],'-.k')
plot([x_rec_EMSF(pp) x_rec_EMSF(pp)],[0  zEMSF],'-.k')
xlabel('x_{k}','fontweight','bold')
Ax = gca;
if x_rec_EMSF(pp)<x_pf(pp)
set(gca, 'XTick', [-2.0,x_rec_EMSF(pp),x_pf(pp),2.0])
xticklabels({'-2.0','x_{EMSF}','x_{CM}','2.0'})
else
set(gca, 'XTick', [-2.0,x_pf(pp),x_rec_EMSF(pp),2.0])
xticklabels({'-2.0','x_{CM}','x_{EMSF}','2.0'})
end
ylabel('Empirical Filtered Density','fontweight','bold')
box on
xlim([-2.0 2.0])
ax = gca
ax.LineWidth = 1.1

