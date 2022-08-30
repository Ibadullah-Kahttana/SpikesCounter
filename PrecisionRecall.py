Detect = 325;
TP = 322;
FP=21;
FN=18;
TN = 0;

Precision = TP / (TP+FP);

print(Precision)

Recall = TP/(TP+FN)

Acc1 = TP+TN;
Acc2 = TP+TN+FP+FN
Accuraccy = Acc1/Acc2 *100

print(Accuraccy)

PRecal = Precision*Recall;
Rpre = Precision+Recall;

DoviideThe = PRecal/Rpre;
F1= 2*DoviideThe

print(F1)


