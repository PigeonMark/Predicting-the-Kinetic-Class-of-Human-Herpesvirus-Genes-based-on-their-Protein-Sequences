import argparse
import numpy as np
from BaseData import BaseData
from PaperSelection import Selector
from Counting import Counter
from Combine import Combiner
from FeatureExtraction import FeatureExtraction
from DebugInfoCollector import DebugInfoCollector
from DataPlotter import DataPlotter
from HomologyFilter import HomologyFilter
from Classification import Classification, ClassificationPlotter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--select', action='store_true')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-c', '--count', action='store_true')
    parser.add_argument('-m', '--merge', '--combine', action='store_true')
    parser.add_argument('-e', '--extract', action='store_true')
    parser.add_argument('-d', '--debuginput', action='store_true')
    parser.add_argument('-r', '--review', action='store_true')
    parser.add_argument('--replace-debug', action='store_true')
    parser.add_argument('-p', '--plot-data', action='store_true')
    parser.add_argument('--base-data', action='store_true')
    parser.add_argument('--features', default='original')
    parser.add_argument('-f', '--homology-filter', action='store_true')
    parser.add_argument('-y', '--classify', action='store_true')
    parser.add_argument('--grid-search', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--fit', action='store_true')

    args = parser.parse_args()

    if args.select:
        if args.test:
            selector = Selector("config/Test/selection_config.json")
        else:
            selector = Selector("config/selection_config.json")
        selector.select()
        selector.selected_to_folder()

    if args.count:
        if args.test:
            counter = Counter("config/Test/counter_config.json")
        else:
            counter = Counter("config/counter_config.json")
        counter.count_all_viruses()

    if args.merge:
        if args.test:
            combiner = Combiner("config/Test/combiner_config.json")
        else:
            combiner = Combiner("config/combiner_config.json")
        combiner.combine_all_viruses()

    if args.debuginput:
        debug_input_collector = DebugInfoCollector("config/debug_info_collector_config.json")
        if args.replace_debug:
            debug_input_collector.collect(True)
        else:
            debug_input_collector.collect()

    if args.review:
        import Review
        Review.run()

    if args.plot_data:
        data_plotter = DataPlotter("config/data_plotter_config.json")
        data_plotter.plot()

    if args.base_data:
        base_data = BaseData("config/base_data_config.json")
        base_data.create_data()

    if args.homology_filter:
        homology_filter = HomologyFilter('config/homology_filter.json')
        homology_filter.filter()

    if args.extract:
        feature_extractor = FeatureExtraction("config/feature_extraction_config.json")
        feature_extractor.extract(args.features)

    if args.classify:
        if args.grid_search:
            MLgrid = [
                {
                    "booster": ["gblinear"],
                    "lambda": [0, 0.1, 0.5, 1],
                    "updater": ["shotgun"],
                    "feature_selector": ["shuffle"]
                },
                {
                    "booster": ["gbtree"],
                    # "max_depth": range(3, 10, 2),
                    # "min_child_weight": range(1, 6, 2)
                }
            ]
            _1vsAgrid = [
                {
                    "estimator__booster": ["gblinear"],
                    "estimator__lambda": [0.1],
                    "estimator__updater": ["coord_descent"],
                    "estimator__feature_selector": ["shuffle"]
                },
                # {
                #     "estimator__booster": ["gbtree"],
                #     "estimator__max_depth": range(3, 10, 2),
                #     "estimator__min_child_weight": range(1, 6, 2)
                # }

            ]
            RRgrid = [
                {
                    "estimator__booster": ["gblinear"],
                    "estimator__lambda": [0.1],
                    "estimator__updater": ["coord_descent"],
                    "estimator__feature_selector": ["shuffle"]
                },
                # {
                #     "estimator__booster": ["gbtree"]
                # #     "estimator__max_depth": range(3, 10, 2),
                # #     "estimator__min_child_weight": range(1, 6, 2)
                # }

            ]
            classification = Classification('config/classification_config.json', args.features)
            classification.grid_search('RR', 'XGBoost', RRgrid, splits=50)
        else:
            if args.fit:
                classification = Classification('config/classification_config.json', args.features)
                classification.fit_all()
                classification.save_scores()

            if args.plot:
                cp = ClassificationPlotter('config/classification_config.json', args.features)
                cp.load_results()
                cp.plot_all()
                cp.plot_feature_importance('ML', 'RF')


if __name__ == "__main__":
    main()
