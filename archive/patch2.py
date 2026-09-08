import os
SRC = os.path.expanduser("~/protoecgnet/src")
def read(f): return open(os.path.join(SRC, f)).read()
def write(f, s): open(os.path.join(SRC, f), "w").write(s)
def ind_of(l): return l[:len(l) - len(l.lstrip())]

# ---------------- proto_models1D.py ----------------
lines = read("proto_models1D.py").split("\n"); out = []; c_clst = c_sep = 0
for l in lines:
    s = l.strip()
    if s == "clst_loss = -torch.mean(correct_class_prototype_activations)":
        ind = ind_of(l)
        out += [ind+"_sw=getattr(model,'_cur_w',None); _wc=getattr(model,'_weight_clst',False); _ws=getattr(model,'_weight_sep',False)",
                ind+"if _wc and _sw is not None:",
                ind+"    _w=_sw.to(device).view(-1,1)",
                ind+"    clst_loss = -(_w*correct_class_prototype_activations).sum()/(_w.sum()*correct_class_prototype_activations.shape[1]+1e-9)",
                ind+"else:",
                ind+"    clst_loss = -torch.mean(correct_class_prototype_activations)"]; c_clst += 1
    elif s == "sep_loss = torch.mean(incorrect_class_prototype_activations)":
        ind = ind_of(l)
        out += [ind+"if _ws and _sw is not None:",
                ind+"    _w=_sw.to(device).view(-1,1)",
                ind+"    sep_loss = (_w*incorrect_class_prototype_activations).sum()/(_w.sum()*incorrect_class_prototype_activations.shape[1]+1e-9)",
                ind+"else:",
                ind+"    sep_loss = torch.mean(incorrect_class_prototype_activations)"]; c_sep += 1
    else:
        out.append(l)
write("proto_models1D.py", "\n".join(out)); print("proto  clst=%d sep=%d" % (c_clst, c_sep))

# ---------------- training_functions.py ----------------
lines = read("training_functions.py").split("\n"); out = []; cur = None; c_init = c_meth = 0
for l in lines:
    s = l.strip()
    if s.startswith("def training_step(self, batch):"): cur = "train"
    elif s.startswith("def validation_step(self, batch):"): cur = "val"
    elif s.startswith("def test_step(self, batch):"): cur = "test"
    elif s.startswith("def "): cur = None
    if s.startswith("self.class_weights = class_weights.to(self.device)"):
        out.append(l); ind = ind_of(l)
        out += [ind+"self.wdict={}",
                ind+"if getattr(self.args,'sample_weights_path',None):",
                ind+"    _d=__import__('numpy').load(self.args.sample_weights_path)",
                ind+"    self.wdict={int(a):float(b) for a,b in zip(_d['ecg_id'],_d['weight'])}",
                ind+"self.model._cur_w=None",
                ind+"self.model._weight_clst=getattr(self.args,'weight_clst',False)",
                ind+"self.model._weight_sep=getattr(self.args,'weight_sep',False)"]; c_init += 1; continue
    if s == "x, y = batch" and cur in ("train", "val", "test"):
        ind = ind_of(l)
        out += [ind+"if len(batch)==3:", ind+"    x, y, _ids = batch"]
        if cur == "train":
            out.append(ind+"    self.model._cur_w = __import__('torch').tensor([self.wdict.get(int(_i),1.0) for _i in _ids], dtype=__import__('torch').float32) if self.wdict else None")
        out += [ind+"else:", ind+"    x, y = batch"]
        if cur != "train":
            out.append(ind+"self.model._cur_w = None")
        c_meth += 1; cur = None; continue
    out.append(l)
write("training_functions.py", "\n".join(out)); print("trainer init=%d methods=%d" % (c_init, c_meth))

# ---------------- main.py (return_sample_ids) ----------------
lines = read("main.py").split("\n"); out = []; c = 0
for l in lines:
    if 'if args.training_stage == "projection":' in l and "sample_weights_path" not in l:
        out.append(l.replace('== "projection":', '== "projection" or args.sample_weights_path:')); c += 1
    else:
        out.append(l)
write("main.py", "\n".join(out)); print("main  return_sample_ids=%d" % c)
