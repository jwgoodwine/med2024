function obj = phi3(x,soln,t)
    alpha = x(1);
    c = x(2);
    beta = alpha + 1;
	mysoln = c*t.^(alpha).*ml(-c*t.^alpha,alpha,beta);
                %figure; plot(t,soln);plot(t,mysoln); title(x)
	obj = (soln' - mysoln)'*(soln' - mysoln);
end
