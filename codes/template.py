def set_template(args):
    #设置模板

    # Set the templates here
    # 在此处设置模板
    if args.model == 'VDSR' or args.model == 'SRCNN' or args.model == 'LGCNET':
        args.cubic_input = True

    if 'TRANSENET' in args.model:
        args.test_block = True
        args.n_basic_modules = 3

    if 'Proposed' in args.model:
        args.test_block = True
        args.n_basic_modules = 3

    if args.dataset == 'AID':
        args.image_size = 600
    elif args.dataset == 'UCMerced':
        if args.scale[0] == 3:
            args.image_size = 255
        else:
            args.image_size = 256