acts = ans.acts;
fe = acts{1,1}.urgency;
se = acts{1,4}.urgency;

fa = acts{1,2}.urgency;
sa = acts{1,5}.urgency;

figure;
plot(fe,'r');
hold on;
plot(se,'b');

figure;
plot(fa,'r');
hold on;
plot(sa,'b');