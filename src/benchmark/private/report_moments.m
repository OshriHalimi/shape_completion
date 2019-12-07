function report_moments(name,arr,appender)
mu = 100*mean(arr);
sigma = 100*std(arr);
fprintf('%s Error : %g +- %g %s\n',name,mu,sigma,appender);
end