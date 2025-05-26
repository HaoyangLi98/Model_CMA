% trans the initial pars into array
function array = pars2array(p)
fields = fieldnames(p);
array = zeros(1, numel(fields));
for i = 1:numel(fields)
    array(i) = p.(fields{i});
end
end