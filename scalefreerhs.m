function xdot = scalefreerhs(t,x,N,A,moved)
  xdot = A*x;
  xdot(moved+N,1) = 0;
end
