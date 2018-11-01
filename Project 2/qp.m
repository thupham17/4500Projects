function qp(a)
%
% This example formulates and solves the following simple QP model:
%      lambda*x'Qx + cx
%  subject to
%      sum(x) = 1
%      lb <= x <= ub

names = {'x1', 'x2', 'x3', 'x4'};
model.varnames = names;
model.Q = 10*sparse(a);
model.obj = [-20.0 -0.04 -0.10 0.05];
model.A = sparse([1 1 1 1]);
model.rhs = [1];
model.lb = [0.010 0.000 0.005 0.030];
model.ub = [0.5 1.0 1.0 0.4];
model.modelsense = 'min';
model.sense = '=';

gurobi_write(model, 'qp.lp');

results = gurobi(model);

for v=1:length(names)
    fprintf('%s %e\n', names{v}, results.x(v));
end

fprintf('Obj: %e\n', results.objval);

end
