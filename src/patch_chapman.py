import os
SRC=os.path.expanduser("~/protoecgnet/src")
def edit(f, old, new):
    p=os.path.join(SRC,f); s=open(p).read()
    if old in s:
        open(p,"w").write(s.replace(old,new,1)); print("  [OK]  ",f)
    else:
        print("  [FAIL]",f,"anchor not found")

# main.py: 1) --dataset arg
edit("main.py",
"    parser.add_argument('--weight_bce', type=str2bool, default=False)",
"    parser.add_argument('--weight_bce', type=str2bool, default=False)\n    parser.add_argument('--dataset', type=str, default='ptbxl')")

# main.py: 2) guard label mappings / num_classes for chapman
edit("main.py",
"""    print("Loading label mappings...")
    label_mappings = load_label_mappings(
        custom_groups=args.custom_groups,
        prototype_category=int(args.label_set) if args.custom_groups else None
    )

    if args.custom_groups:
        num_classes = len(label_mappings["custom"])
    else:
        num_classes = len(label_mappings[args.label_set])""",
"""    if args.dataset == 'chapman':
        from ecg_utils_chapman import CHAPMAN_CLASSES
        num_classes = len(CHAPMAN_CLASSES)
    else:
        print("Loading label mappings...")
        label_mappings = load_label_mappings(
            custom_groups=args.custom_groups,
            prototype_category=int(args.label_set) if args.custom_groups else None
        )
        if args.custom_groups:
            num_classes = len(label_mappings["custom"])
        else:
            num_classes = len(label_mappings[args.label_set])""")

# main.py: 3) loader branch for chapman
edit("main.py",
"        train_loader, val_loader, test_loader, class_weights = get_dataloaders(",
"""        if args.dataset == 'chapman':
            from ecg_utils_chapman import get_chapman_dataloaders
            train_loader, val_loader, test_loader, class_weights = get_chapman_dataloaders(batch_size=args.batch_size, work_num=args.num_workers, return_sample_ids=return_sample_ids, remove_baseline=args.remove_baseline, seed=args.seed)
        else:
            train_loader, val_loader, test_loader, class_weights = get_dataloaders(""")

# training_functions.py: class names for chapman
edit("training_functions.py",
"""def get_class_names(args):
    from ecg_utils import load_label_mappings""",
"""def get_class_names(args):
    if getattr(args, 'dataset', 'ptbxl') == 'chapman':
        from ecg_utils_chapman import CHAPMAN_CLASSES
        return CHAPMAN_CLASSES
    from ecg_utils import load_label_mappings""")

# proto_models1D.py: co-occurrence path for chapman
edit("proto_models1D.py",
'                elif self.label_set == "3":',
'                elif self.label_set == "chapman":\n                    path = "/home/psxpk7/protoecgnet/experiments/preprocessing/label_cooccur_chapman.pt"\n                elif self.label_set == "3":')
print("done.")
