test
a = [4.2769    0.3705    0.2679    0.5052;
0.3705    4.3171    0.6659    0.2621;
0.2679    0.6659    4.7655    0.6204;
0.5052    0.2621    0.6204    4.6463]

Q = [54.00 -0.30 -0.02 0.00; -0.30  12.00 0.50  1.00; -0.02   0.50 4.00  0.3; 0.00   1.0  0.3   0.02];

subs(10*x.'*Q*x + m*x, [x1 x2 x3 x4], [0 0 0 0]);

x = quadprog(20*a,m,[],[],[1 1 1 1],1,lb,ub)

f = @(x) 10*x.'*a*x + m*x
z = fmincon(f,[0.1 0.2 0.3 0.4].',[],[],[1 1 1 1],1,[0.010 0.000 0.005 0.030],[0.5 1.0 1.0 0.4]);

%% step size 
syms y1 y2 y3 y4 s
syms x1 x2 x3 x4 
y = [y1; y2;y3;y4];
x = [x1;x2;x3;x4];
G = @(x,y,s) 10*(x+s*y).'*Q*(x+s*y) + m*(x+s*y);
G1 = @(x,y,s) (x+s*y).'*Q*(x+s*y) + m*(x+s*y);
diff(f(x+s*y),s)
solve(diff(G(x0,y0,s),s)==0,s)
solve(subs(diff(f(x+s*y),s),[x1 x2 x3 x4 y1 y2 y3 y4],[0.5 0.465 0.005 0.03 -0.49 -0.465 0.955 0])==0,s)