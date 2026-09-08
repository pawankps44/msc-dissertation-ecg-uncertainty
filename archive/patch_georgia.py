import os
SRC=os.path.expanduser("~/protoecgnet/src")
def edit(f, old, new, marker):
    p=os.path.join(SRC,f); s=open(p).read()
    if marker in s:
        print("  [SKIP]",f,"(already has georgia)"); return
    if old in s:
        open(p,"w").write(s.replace(old,new,1)); print("  [OK]  ",f)
    else:
        print("  [FAIL]",f,"anchor not found")

edit("main.py",
"'4', 'chapman'], default='superdiagnostic')",
"'4', 'chapman', 'georgia'], default='superdiagnostic')",
"'chapman', 'georgia']")

edit("main.py",
"""        num_classes = len(CHAPMAN_CLASSES)
    else:
        print("Loading label mappings...")""",
"""        num_classes = len(CHAPMAN_CLASSES)
    elif args.dataset == 'georgia':
        from ecg_utils_georgia import GEORGIA_CLASSES
        num_classes = len(GEORGIA_CLASSES)
    else:
        print("Loading label mappings...")""",
"num_classes = len(GEORGIA_CLASSES)")

edit("main.py",
"""            train_loader, val_loader, test_loader, class_weights = get_chapman_dataloaders(batch_size=args.batch_size, work_num=args.num_workers, return_sample_ids=return_sample_ids, remove_baseline=args.remove_baseline, seed=args.seed)
        else:
            train_loader, val_loader, test_loader, class_weights = get_dataloaders(""",
"""            train_loader, val_loader, test_loader, class_weights = get_chapman_dataloaders(batch_size=args.batch_size, work_num=args.num_workers, return_sample_ids=return_sample_ids, remove_baseline=args.remove_baseline, seed=args.seed)
        elif args.dataset == 'georgia':
            from ecg_utils_georgia import get_georgia_dataloaders
            train_loader, val_loader, test_loader, class_weights = get_georgia_dataloaders(batch_size=args.batch_size, work_num=args.num_workers, return_sample_ids=return_sample_ids, remove_baseline=args.remove_baseline, seed=args.seed)
        else:
            train_loader, val_loader, test_loader, class_weights = get_dataloaders(""",
"get_georgia_dataloaders")

edit("training_functions.py",
"""    if getattr(args, 'dataset', 'ptbxl') == 'chapman':
        from ecg_utils_chapman import CHAPMAN_CLASSES
        return CHAPMAN_CLASSES
    from ecg_utils import load_label_mappings""",
"""    if getattr(args, 'dataset', 'ptbxl') == 'chapman':
        from ecg_utils_chapman import CHAPMAN_CLASSES
        return CHAPMAN_CLASSES
    if getattr(args, 'dataset', 'ptbxl') == 'georgia':
        from ecg_utils_georgia import GEORGIA_CLASSES
        return GEORGIA_CLASSES
    from ecg_utils import load_label_mappings""",
"from ecg_utils_georgia import GEORGIA_CLASSES")

edit("proto_models1D.py",
'''                elif self.label_set == "chapman":
                    path = "/home/psxpk7/protoecgnet/experiments/preprocessing/label_cooccur_chapman.pt"
                elif self.label_set == "3":''',
'''                elif self.label_set == "chapman":
                    path = "/home/psxpk7/protoecgnet/experiments/preprocessing/label_cooccur_chapman.pt"
                elif self.label_set == "georgia":
                    path = "/home/psxpk7/protoecgnet/experiments/preprocessing/label_cooccur_georgia.pt"
                elif self.label_set == "3":''',
"label_cooccur_georgia")
print("done.")
