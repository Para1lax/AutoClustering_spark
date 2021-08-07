from sys import float_info

DEBUG = True
DEBUG_PREFIX = "=====> "


class Constants:
    kmeans_algo = "KMeans"
    # affinity_algo = "Affinity_Propagation"
    # mean_shift_algo = "Mean_Shift"
    # ward_algo = "Ward"
    dbscan_algo = "DBSCAN"
    gm_algo = "Gaussian_Mixture"
    # bgm_algo = "Bayesian_Gaussian_Mixture"
    bisecting_kmeans = "BisectingKMeans"

    # num_algos = 3
    #
    # n_clusters_upper_bound = 15
    # bandit_timeout = 30  # 5 # seconds for each bandit iteration
    # bandit_iterations = 40  # 10 # iterations number
    # batch_size = 40
    # tuner_timeout = bandit_timeout * (bandit_iterations + 1) / num_algos

    max_no_improvement_iterations = 6

    smac_temp_dir = "/tmp/rm_me/"

    algos = [
        #kmeans_algo,
        # affinity_algo,
        # mean_shift_algo,
        # ward_algo,
        dbscan_algo,
        #gm_algo,
        # bgm_algo
        #bisecting_kmeans
    ]

    paused = "paused"
    resume = "resume"
    run = "run"

    n_samples = 500  # to generate data

    tau = 0.5

    proj_root = '~/WORK/MultiClustering/'
    experiment_path = 'datasets/normalized/'
    unified_data_path = 'datasets/unified/'

    bad_cluster = float_info.max
    in_reward = 1000.0
    best_init = 1000000000.0

    seeds = [1, 11, 111, 211, 311]
    noisy = False

    #algorithm = "rfrsls-ucb-SRSU-100"
    algorithm = 'softmax'
