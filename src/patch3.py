import os
SRC=os.path.expanduser("~/protoecgnet/src")
def read(f): return open(os.path.join(SRC,f)).read()
def write(f,s): open(os.path.join(SRC,f),"w").write(s)
def edit(f, old, new):
    s=read(f)
    if old in s:
        write(f, s.replace(old,new,1)); print("  [OK]  ", f)
    else:
        print("  [FAIL]", f, "-- anchor not found")

edit("proto_models1D.py",
     "_ws=getattr(model,'_weight_sep',False)",
     "_ws=getattr(model,'_weight_sep',False); _wb=getattr(model,'_weight_bce',False)")

edit("proto_models1D.py",
"    classification_loss = F.binary_cross_entropy_with_logits(logits, y_true, pos_weight=class_weights)",
"""    if _wb and _sw is not None:
        _perbce = F.binary_cross_entropy_with_logits(logits, y_true, pos_weight=class_weights, reduction='none')
        _wv = _sw.to(device).view(-1, 1)
        classification_loss = (_wv * _perbce).sum() / (_wv.sum() * _perbce.shape[1] + 1e-9)
    else:
        classification_loss = F.binary_cross_entropy_with_logits(logits, y_true, pos_weight=class_weights)""")

edit("main.py",
     "parser.add_argument('--weight_sep', type=str2bool, default=False)",
     "parser.add_argument('--weight_sep', type=str2bool, default=False)\n    parser.add_argument('--weight_bce', type=str2bool, default=False)")

edit("training_functions.py",
     "self.model._weight_sep=getattr(self.args,'weight_sep',False)",
     "self.model._weight_sep=getattr(self.args,'weight_sep',False)\n        self.model._weight_bce=getattr(self.args,'weight_bce',False)")
print("done.")