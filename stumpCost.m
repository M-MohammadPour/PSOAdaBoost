function stump=stumpCost(thr,x,y,w)
N = length(x);
y_predict = zeros(N, 1);

idx = logical(x >= x(thr));
y_predict(idx) = 1;
y_predict(~idx) = -1;


err_label = logical(y ~= y_predict);
err1 = sum(err_label.*w)/sum(w);

y_predict=y_predict*-1;
err_label = logical(y ~= y_predict);
err2 = sum(err_label.*w)/sum(w);

if err1<err2
    stump.error = err1;
    stump.threshold = thr;
    stump.less = -1;
    stump.more = 1;
else
    stump.error = err2;
    stump.threshold = thr;
    stump.less = 1;
    stump.more = -1;
end

end
