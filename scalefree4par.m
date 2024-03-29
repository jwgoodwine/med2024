% just makes one network and computed order(s)
close all; clear all; more off;
format shortG;
rng(1)

tfinish = 10;
opts = odeset('RelTol',1e-5,'AbsTol',1e-8);
for run = 1:1
    disp(run)
    b = (150);
    k = 2500+1;
    mincon = ceil(rand()*2)+1;
    mincon = 3;
    N = ceil(rand()*2000)+50;
    N = 2000;
    A = zeros(N,N);
    bigA = zeros(2*N,2*N);
    A(1:mincon,1:mincon) = ones(mincon,mincon)-eye(mincon);
    bigA(1+N:mincon+N,1:mincon) = k*(ones(mincon,mincon)  - (mincon)*eye(mincon));
    bigA(N+1:N+(mincon),N+1:N+(mincon)) = b*(ones((mincon),(mincon))  - (mincon)*eye((mincon)));
    for n=mincon+1:N
        adj = sum(A');
        for i=1:mincon
            flag = 0;
            while(flag<1)
                target = floor(rand()*(n-1))+1;
                thresh = rand()*(n+mincon-1);
                if(adj(target) >  thresh && target ~= n && A(target,n) ~= 1)
                    A(target,n) = 1;
                    A(n,target) = 1;
                    flag = 1;
                    if(rand() > 0.5)
                        bigA(target+N,n) = k;
                        bigA(n+N,target) = k;
                        bigA(n+N,n) = bigA(n+N,n) - k;
                        bigA(target+N,target) = bigA(target+N,target) - k;
                    else
                        bigA(target+N,n+N) = b;
                        bigA(n+N,target+N) = b;
                        bigA(n+N,n+N) = bigA(n+N,n+N) - b;
                        bigA(N+target,N+target) = bigA(N+target,N+target) - b;
                    end
                end
            end
        end
    end
    for n=1:N
        bigA(n,n+N) = 1;
    end
    G = graph(A);

    ic = zeros(1,2*N);
    moved = floor(rand()*N+1);
    moved = 100;
    ic(moved) = 1;
    t = linspace(0,tfinish,101);
    [t, y] = ode45(@(t,y)scalefreerhs(t,y,N,bigA,moved),t,ic',opts);
    target = floor(rand()*N+1);
    while (target == moved)
        target = floor(rand()*N+1);
    end
    target = 1011;
    thedist = length(shortestpath(G,moved,target))-1;
    len = length(t);
    cutbeginning = 1;
    soln = y(cutbeginning:len,target)';
    newtime = t(cutbeginning:len);
plot(newtime,soln)
    if soln(length(soln)) > 0.1

        % bounds for fractional search
        lb = [1.001 5];   % order is [alpha,c]
        ub = [1.999 9];
        % bounds for second-order integer search
        lb2 = [1 0.00001];  % order is [omega_n, zeta]
        ub2 = [4 .5];

        options = optimoptions("patternsearch",'Display','none',MaxIterations=5000);
        [xd,fvald,exitflag,output] = patternsearch(@(x)phi3(x,soln,newtime),[1 .1],[],[],[],[],lb,ub,[],options)
        [xd2,fvald2,exitflag,output] = patternsearch(@(x)phi2(x,soln,newtime),[1 .1],[],[],[],[],lb2,ub2,[],options)

        options = optimoptions('particleswarm','Display','none','SwarmSize',100);
        [x,fval,exitflag,output] = particleswarm(@(x)phi3(x,soln,newtime),2,lb,ub,options)
        [x2,fval2,exitflag,output] = particleswarm(@(x)phi2(x,soln,newtime),2,lb2,ub2,options)
        disp([xd,xd2,x,x2])

    end
end
