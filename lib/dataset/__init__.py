

def get_dataset(args, oracle):
    if args.dataset_type == "covid":
        from lib.dataset.regression import CovidAllDataset
        return CovidAllDataset(args.proxy_data_split, args.num_folds, args, oracle)
    elif args.dataset_type == "true_aff":
        from lib.dataset.regression import TrueAffDataset
        return TrueAffDataset(args.proxy_data_split, args.num_folds, args, oracle)
    elif args.dataset_type == "true_aff_hard":
        from lib.dataset.regression import TrueAffHardDataset
        return TrueAffHardDataset(args.proxy_data_split, args.num_folds, args, oracle)
    