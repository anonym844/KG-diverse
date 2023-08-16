import argparse


def parse_dpp_args():
    parser = argparse.ArgumentParser(description="Run DPP.")

    parser.add_argument('--seed', type=int, default=2023,
                        help='Random seed.')

    parser.add_argument('--data_name', nargs='?', default='book',
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument('--data_dir', nargs='?', default='datasets/',
                        help='Input data path.')

    parser.add_argument('--cf_batch_size', type=int, default=1024,
                        help='CF batch size.')
    parser.add_argument('--kg_batch_size', type=int, default=2048,
                        help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=10000,
                        help='Test batch size (the user number to test every batch).')

    parser.add_argument('--embed_dim', type=int, default=64,
                        help='User / entity Embedding size.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='CPU')
    parser.add_argument('--relation_dim', type=int, default=64,
                        help='Relation Embedding size.')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='number of layers.')
    parser.add_argument('--KG_layers', type=int, default=3,
                        help='number of KG layers.')
    parser.add_argument('--gamma', type=float, default= 0.1,
                        help='coefficient on KG loss.')
    parser.add_argument('--beta', type=float, default= 0.5,
                        help='Gamma on uniformity.')
    parser.add_argument('--temperature', type=float, default=1.0,
                            help='temperature of softmax.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--wd', type=float, default=0,
                        help='Weight decay.')
    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Calculate metric@K when evaluating.')

    args = parser.parse_args()

    save_dir = 'trained_model/model/{}/'.format(
        args.data_name)
    args.save_dir = save_dir

    return args


