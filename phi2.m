function obj = phi2(x,soln,t)
    poles = roots([1 2*x(1)*x(2) x(1)^2]);
    if ~isreal(poles(1))
        %disp('complex')
        omegan = x(1);
        zeta = x(2);
        omegad = omegan*sqrt(1-zeta^2);
	    mysoln = 1-exp(-zeta*omegan*t).*(cos(omegad*t)+zeta/sqrt(1-zeta^2)*sin(omegad*t));
    else
        %disp('real')
        p1 = poles(1);
        p2 = poles(2);
        mysoln = 1 - p2*exp(p1*t)/(p2 - p1) + p1*exp(p2*t)/(p2-p1);
    end
	obj = (soln' - mysoln)'*(soln' - mysoln);
end
