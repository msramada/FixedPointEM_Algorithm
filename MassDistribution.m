%% Resampling step of the bootstrap particle filter.
function [x_tilda] = MassDistribution(x,p)
p=p./sum(p);
sample=rand(size(p));
u=cumsum(p);

for i=1:length(p)
I=find(u>=sample(i));
x_tilda(:,i,1)=x(:,min(I));
end


end

